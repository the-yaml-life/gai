"""
LLM client factory for routing between different backends (Groq, Ollama, Anannas, etc.)
"""

from typing import Optional, List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor
import time

from gai.ai.groq_client import GroqClient, GroqError
from gai.ai.ollama_client import OllamaClient, OllamaError
from gai.ai.anannas_client import AnannasClient, AnannasError
from gai.core.config import Config


# Base exception for all LLM errors
class LLMError(Exception):
    """Base LLM error"""
    pass


def create_client(model: str, config: Config, verbose: bool = False):
    """
    Create appropriate LLM client based on model name.

    Model naming convention:
    - "ollama/model-name" → OllamaClient
    - "anannas/model-name" → AnannasClient
    - "groq/model-name" or just "model-name" → GroqClient (default)

    Args:
        model: Model name (with optional backend prefix)
        config: Configuration object
        verbose: Enable verbose logging

    Returns:
        Either GroqClient, OllamaClient, or AnannasClient instance
    """
    if model.startswith("ollama/"):
        # Ollama model
        return OllamaClient(
            model=model,
            temperature=config.get('ai.temperature', 0.3),
            verbose=verbose,
            base_url=config.get('ollama.base_url', 'http://localhost:11434')
        )
    elif model.startswith("anannas/"):
        # Anannas model
        # Strip "anannas/" prefix for the actual model name
        anannas_model = model.replace("anannas/", "", 1)

        # Get API key from env (separate from Groq key)
        import os
        api_key = os.getenv("ANANNAS_API_KEY") or config.api_key  # Fallback to GROQ key if not set

        return AnannasClient(
            api_key=api_key,
            model=anannas_model,
            fallback_models=config.get('ai.fallback_models', []),
            temperature=config.get('ai.temperature', 0.3),
            verbose=verbose,
            api_url=config.get('anannas.api_url', 'https://api.anannas.ai/v1/chat/completions')
        )
    else:
        # Groq model (default)
        # Strip "groq/" prefix if present
        groq_model = model.replace("groq/", "")

        return GroqClient(
            api_key=config.api_key,
            model=groq_model,
            fallback_models=config.get('ai.fallback_models', []),
            temperature=config.get('ai.temperature', 0.3),
            verbose=verbose,
            api_url=config.get('ai.api_url', 'https://api.groq.com/openai/v1/chat/completions')
        )


class MultiBackendClient:
    """
    Unified client that can handle multiple backends (Groq, Ollama) with parallel execution.
    """

    def __init__(self, config: Config, verbose: bool = False):
        self.config = config
        self.verbose = verbose

        # Primary model
        primary_model = config.get('ai.model')
        self.primary_client = create_client(primary_model, config, verbose)
        self.primary_model = primary_model

        # Parallel models for multi-model generation
        self.parallel_models = config.get('ai.parallel_models', [])

    def _log(self, message: str):
        """Log if verbose"""
        if self.verbose:
            print(f"[MultiBackendClient] {message}")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 500,
        temperature: Optional[float] = None,
        retry_with_fallback: bool = True
    ) -> str:
        """
        Generate completion using primary model.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate
            temperature: Temperature override
            retry_with_fallback: Whether to try fallback (only applies to Groq)

        Returns:
            Generated text
        """
        try:
            return self.primary_client.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                retry_with_fallback=retry_with_fallback
            )
        except (GroqError, OllamaError, AnannasError) as e:
            raise LLMError(str(e))

    def generate_parallel(
        self,
        prompts: List[Tuple[str, str, int]],
        models: Optional[List[str]] = None,
        system_prompt: Optional[str] = None
    ) -> List[str]:
        """
        Generate completions in parallel using multiple models (can be mixed backends).

        Args:
            prompts: List of (prompt_text, description, max_tokens) tuples
            models: List of model names to use (if None, uses config.parallel_models)
            system_prompt: Optional system prompt for all requests

        Returns:
            List of generated texts (same order as prompts)

        Raises:
            LLMError: If generation fails
        """
        if models is None:
            models = self.parallel_models

        if not models:
            raise LLMError("No models configured for parallel generation")

        if len(prompts) > len(models):
            raise LLMError(f"Not enough models ({len(models)}) for prompts ({len(prompts)})")

        self._log(f"Parallel generation: {len(prompts)} prompts with {len(models)} models")

        # Group models by backend for efficient client creation
        clients_cache = {}

        def get_client(model: str):
            """Get or create client for model"""
            if model not in clients_cache:
                clients_cache[model] = create_client(model, self.config, self.verbose)
            return clients_cache[model]

        # Track pending tasks
        pending = {idx: (prompt_data, []) for idx, prompt_data in enumerate(prompts)}
        results = {}

        MAX_ITERATIONS = 10  # Prevent infinite loops

        iteration = 0
        while pending and iteration < MAX_ITERATIONS:
            iteration += 1
            tasks_to_run = []

            # Assign models to pending tasks
            for idx, (prompt_data, tried_models) in pending.items():
                available_models = [m for m in models if m not in tried_models]

                if not available_models:
                    # All models tried for this task - wait and retry
                    self._log(f"[{idx}] All models tried, resetting...")
                    tried_models.clear()
                    available_models = models[:]

                # Pick next model to try
                model = available_models[0]
                tasks_to_run.append((idx, prompt_data, model))

            # Execute batch in parallel
            def generate_one(idx: int, prompt_data: Tuple[str, str, int], model: str) -> Tuple[int, str, bool]:
                """Generate one completion and return (index, result, success)"""
                prompt_text, description, max_tokens = prompt_data
                self._log(f"[{idx}] {description} using {model}")

                try:
                    client = get_client(model)
                    result = client.generate(
                        prompt=prompt_text,
                        system_prompt=system_prompt,
                        max_tokens=max_tokens,
                        retry_with_fallback=False  # Don't use fallback in parallel mode
                    )

                    self._log(f"[{idx}] ✓ Complete with {model}")
                    return (idx, result, True)

                except (GroqError, OllamaError, AnannasError) as e:
                    self._log(f"[{idx}] Error with {model}: {e}")
                    return (idx, "", False)
                except Exception as e:
                    self._log(f"[{idx}] Unexpected error with {model}: {e}")
                    return (idx, "", False)

            # Run tasks in parallel
            with ThreadPoolExecutor(max_workers=len(tasks_to_run)) as executor:
                futures = []
                for idx, prompt_data, model in tasks_to_run:
                    future = executor.submit(generate_one, idx, prompt_data, model)
                    futures.append((future, idx, model))

                # Collect results
                for future, idx, model in futures:
                    result_idx, result_text, success = future.result()

                    if success:
                        results[result_idx] = result_text
                        if result_idx in pending:
                            del pending[result_idx]
                    else:
                        # Mark model as tried and retry
                        if result_idx in pending:
                            pending[result_idx][1].append(model)

            # Small delay between iterations
            if pending:
                time.sleep(1)

        if pending:
            raise LLMError(f"Failed to complete {len(pending)} prompts after {MAX_ITERATIONS} iterations")

        # Return in original order
        return [results[i] for i in range(len(prompts))]

    def list_models(self, raw=False) -> Dict[str, List[Dict[str, Any]]]:
        """
        List available models from all configured backends.

        Args:
            raw: If True, return raw API responses without modifications

        Returns:
            Dict with backend names as keys and model lists as values
        """
        all_models = {}

        # List Groq models
        try:
            groq_client = create_client("groq/dummy", self.config, verbose=False)
            groq_models = groq_client.list_models(raw=raw)
            if groq_models:
                all_models['groq'] = groq_models
        except Exception as e:
            self._log(f"Failed to list Groq models: {e}")

        # List Anannas models (only if API key is set)
        try:
            import os
            if os.getenv("ANANNAS_API_KEY"):
                anannas_client = create_client("anannas/dummy", self.config, verbose=False)
                anannas_models = anannas_client.list_models(raw=raw)
                if anannas_models:
                    all_models['anannas'] = anannas_models
        except Exception as e:
            self._log(f"Failed to list Anannas models: {e}")

        # List Ollama models
        try:
            ollama_client = create_client("ollama/dummy", self.config, verbose=False)
            ollama_models = ollama_client.list_models(raw=raw)
            if ollama_models:
                all_models['ollama'] = ollama_models
        except Exception as e:
            self._log(f"Failed to list Ollama models: {e}")

        return all_models
