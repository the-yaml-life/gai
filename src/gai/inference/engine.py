"""
Inference engine - orchestrates adapters and strategies
"""

import os
from typing import Dict, Optional
from gai.inference.models import InferenceRequest, InferenceResponse
from gai.inference.exceptions import InferenceError, AllModelsFailedError, RateLimitError, BillingError
from gai.inference.adapters import GroqAdapter, OpenRouterAdapter, AnannasAdapter, OllamaAdapter
from gai.inference.strategies import (
    SimpleRetryStrategy,
    SimpleFallbackStrategy,
    IntelligentRateLimitStrategy
)
from gai.core.config import Config


class InferenceEngine:
    """
    Main inference engine that orchestrates all components.

    Coordinates:
    - Provider adapters (HTTP requests)
    - Retry strategy (error recovery)
    - Fallback strategy (model switching)
    - Rate limit strategy (intelligent limiting)
    """

    def __init__(
        self,
        adapters: Dict[str, object],
        retry_strategy: SimpleRetryStrategy,
        fallback_strategy: SimpleFallbackStrategy,
        rate_limit_strategy: IntelligentRateLimitStrategy,
        verbose: bool = False
    ):
        self.adapters = adapters
        self.retry = retry_strategy
        self.fallback = fallback_strategy
        self.rate_limit = rate_limit_strategy
        self.verbose = verbose

    def _log(self, message: str):
        if self.verbose:
            print(f"[InferenceEngine] {message}")

    def _get_adapter(self, model: str):
        """Get adapter for model based on prefix"""
        if model.startswith("openrouter/"):
            return self.adapters["openrouter"], model.replace("openrouter/", "")
        elif model.startswith("anannas/"):
            return self.adapters["anannas"], model.replace("anannas/", "")
        elif model.startswith("ollama."):
            # ollama.endpoint/model format
            parts = model.split("/", 1)
            if len(parts) == 2:
                endpoint_name = parts[0].replace("ollama.", "")
                model_name = parts[1]
                # Use endpoint-specific adapter if available
                adapter_key = f"ollama.{endpoint_name}"
                if adapter_key in self.adapters:
                    return self.adapters[adapter_key], model_name
            # Fallback to default ollama
            return self.adapters.get("ollama"), model
        elif model.startswith("ollama/"):
            return self.adapters["ollama"], model.replace("ollama/", "")
        elif model.startswith("groq/"):
            return self.adapters["groq"], model.replace("groq/", "")
        else:
            # Default to groq
            return self.adapters["groq"], model

    def generate(self, request: InferenceRequest) -> InferenceResponse:
        """
        Generate text using inference engine.

        Orchestration flow:
        1. Get fallback chain (models to try)
        2. For each model:
           a. Check rate limit
           b. Try with retry strategy
           c. If rate limit, learn and skip to next model
           d. If other error, retry
        3. Return first successful response
        4. If all fail, raise AllModelsFailedError
        """
        # Get models to try
        models = self.fallback.get_models(request.model)
        self._log(f"Trying {len(models)} models in sequence")

        errors = {}

        for model_full in models:
            self._log(f"Attempting model: {model_full}")

            try:
                # Get adapter for this model
                adapter, model_name = self._get_adapter(model_full)

                if adapter is None:
                    self._log(f"No adapter for {model_full}, skipping")
                    errors[model_full] = "No adapter configured"
                    continue

                # Try this model with retry
                response = self.retry.execute(
                    fn=lambda: self._try_model(adapter, model_name, model_full, request),
                    on_error=self.rate_limit.handle
                )

                # Success!
                self._log(f"Success with {model_full}")
                return response

            except RateLimitError as e:
                # Rate limit - already learned, skip to next model
                self._log(f"Rate limit on {model_full}, skipping to next")
                errors[model_full] = f"Rate limit: {e.wait_time:.1f}s"
                continue

            except BillingError as e:
                # Billing error - skip to next model immediately
                self._log(f"Billing error on {model_full}, skipping to next")
                errors[model_full] = f"Billing: {str(e)}"
                continue

            except Exception as e:
                # Other error - already retried by retry strategy
                self._log(f"Failed with {model_full}: {e}")
                errors[model_full] = str(e)
                continue

        # All models failed
        raise AllModelsFailedError(errors)

    def _try_model(self, adapter, model_name: str, model_full: str, request: InferenceRequest) -> InferenceResponse:
        """Try inference with a specific model"""
        # Check rate limit before attempting
        self.rate_limit.check_can_use(model_full)

        # Make request
        response_data = adapter.request(
            messages=request.messages,
            model=model_name,
            **request.params
        )

        # Parse response
        text = adapter.parse_response(response_data)
        usage = adapter.get_usage(response_data)

        # Record usage for rate limiting
        if usage:
            self.rate_limit.record_usage(model_full, usage)

        return InferenceResponse(
            text=text.strip(),
            model=model_full,
            usage=usage
        )

    def generate_parallel(
        self,
        prompts: list,
        models: list,
        system_prompt: str = None
    ) -> list:
        """
        Generate completions in parallel using multiple models.

        Args:
            prompts: List of (prompt_text, description, max_tokens) tuples
            models: List of model names to use
            system_prompt: Optional system prompt for all requests

        Returns:
            List of generated texts (same order as prompts)
        """
        from concurrent.futures import ThreadPoolExecutor
        import time

        if not models:
            raise InferenceError("No models provided for parallel generation")

        if len(prompts) > len(models):
            raise InferenceError(f"Not enough models ({len(models)}) for prompts ({len(prompts)})")

        self._log(f"Parallel generation: {len(prompts)} prompts with {len(models)} models")

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
            def generate_one(idx: int, prompt_data: tuple, model: str) -> tuple:
                """Generate one completion and return (index, result, success)"""
                prompt_text, description, max_tokens = prompt_data
                self._log(f"[{idx}] {description} using {model}")

                try:
                    # Create request
                    messages = []
                    if system_prompt:
                        messages.append({"role": "system", "content": system_prompt})
                    messages.append({"role": "user", "content": prompt_text})

                    request = InferenceRequest(
                        messages=messages,
                        model=model,
                        params={"max_tokens": max_tokens}
                    )

                    # Generate using engine
                    response = self.generate(request)

                    self._log(f"[{idx}] ✓ Complete with {model}")
                    return (idx, response.text, True)

                except (RateLimitError, BillingError) as e:
                    self._log(f"[{idx}] Rate limit/billing error with {model}: {e}")
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
            raise InferenceError(f"Failed to complete {len(pending)} prompts after {MAX_ITERATIONS} iterations")

        # Return in original order
        return [results[i] for i in range(len(prompts))]


def get_inference_engine(config: Config, verbose: bool = False) -> InferenceEngine:
    """
    Factory function to create InferenceEngine from config.

    Args:
        config: GAI configuration
        verbose: Enable verbose logging

    Returns:
        Configured InferenceEngine instance
    """
    # Create adapters
    adapters = {}

    # Groq adapter
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        adapters["groq"] = GroqAdapter(
            api_key=groq_key,
            api_url=config.get("ai.api_url", "https://api.groq.com/openai/v1/chat/completions"),
            verbose=verbose
        )

    # OpenRouter adapter
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_key:
        adapters["openrouter"] = OpenRouterAdapter(
            api_key=openrouter_key,
            api_url=config.get("openrouter.api_url", "https://openrouter.ai/api/v1/chat/completions"),
            verbose=verbose
        )

    # Anannas adapter
    anannas_key = os.getenv("ANANNAS_API_KEY")
    if anannas_key:
        adapters["anannas"] = AnannasAdapter(
            api_key=anannas_key,
            api_url=config.get("anannas.api_url", "https://api.anannas.ai/v1/chat/completions"),
            verbose=verbose
        )

    # Ollama adapters (can have multiple endpoints)
    ollama_endpoints = config.get("ollama.endpoints", {})
    if ollama_endpoints:
        for endpoint_name, endpoint_url in ollama_endpoints.items():
            adapters[f"ollama.{endpoint_name}"] = OllamaAdapter(
                base_url=endpoint_url,
                verbose=verbose
            )
    else:
        # Default Ollama
        ollama_url = config.get("ollama.base_url", "http://localhost:11434")
        adapters["ollama"] = OllamaAdapter(base_url=ollama_url, verbose=verbose)

    # Create strategies
    retry_strategy = SimpleRetryStrategy(
        max_retries=3,
        backoff_base=2.0,
        verbose=verbose
    )

    fallback_strategy = SimpleFallbackStrategy(
        parallel_models=config.get("ai.parallel_models", []),
        verbose=verbose
    )

    rate_limit_strategy = IntelligentRateLimitStrategy(verbose=verbose)

    # Create engine
    return InferenceEngine(
        adapters=adapters,
        retry_strategy=retry_strategy,
        fallback_strategy=fallback_strategy,
        rate_limit_strategy=rate_limit_strategy,
        verbose=verbose
    )
