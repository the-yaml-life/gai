"""
List available models command
"""

import yaml
from pathlib import Path
from gai.core.config import Config
from gai.ai.groq_client import GroqClient
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt, Confirm

console = Console()


class ModelsCommand:
    """List available models from API"""

    def __init__(self, config: Config, verbose: bool = False):
        self.config = config
        self.verbose = verbose

        # Initialize AI client
        self.client = GroqClient(
            api_key=config.api_key,
            model=config.get('ai.model'),
            fallback_models=[],
            temperature=0.3,
            verbose=verbose,
            api_url=config.get('ai.api_url', 'https://api.groq.com/openai/v1/chat/completions')
        )

    def run(self, select=False):
        """
        List available models.

        Args:
            select: If True, allow interactive selection to update config
        """
        console.print("\n[bold cyan]Fetching available models...[/bold cyan]\n")

        all_models = self.client.list_models()

        if not all_models:
            console.print("[yellow]No models found or API error[/yellow]")
            return

        # Filter for text generation models only
        models = self._filter_text_models(all_models)

        if not models:
            console.print("[yellow]No text models found[/yellow]")
            return

        # Create table with index column for selection
        table = Table(show_header=True, header_style="bold magenta")
        if select:
            table.add_column("#", style="dim", width=4)
        table.add_column("Model ID", style="cyan", width=45)
        table.add_column("Context", style="yellow", justify="right", width=10)
        table.add_column("Owner", style="green", width=15)

        for idx, model in enumerate(models, 1):
            model_id = model.get("id", "unknown")
            owner = model.get("owned_by", "unknown")
            context = self._get_context_length(model)

            if select:
                table.add_row(str(idx), model_id, context, owner)
            else:
                table.add_row(model_id, context, owner)

        console.print(table)
        console.print(f"\n[dim]Total: {len(models)} text models[/dim]")

        if len(all_models) > len(models):
            filtered = len(all_models) - len(models)
            console.print(f"[dim]Filtered out {filtered} non-text models (whisper, tts, etc)[/dim]")
        console.print()

        # Interactive selection
        if select:
            self._interactive_selection(models)

    def _filter_text_models(self, models):
        """Filter models to only include text generation models"""
        text_models = []

        # Keywords that indicate non-text models
        exclude_keywords = [
            'whisper',      # Speech-to-text
            'tts',          # Text-to-speech
            'playai',       # TTS service
            'distil-whisper',
        ]

        for model in models:
            model_id = model.get("id", "").lower()

            # Skip if contains exclude keywords
            if any(keyword in model_id for keyword in exclude_keywords):
                continue

            text_models.append(model)

        return text_models

    def _get_context_length(self, model):
        """Extract context length from model data or name"""
        model_id = model.get("id", "")

        # Check context_window field if available
        context_window = model.get("context_window")
        if context_window:
            if context_window >= 100000:
                return f"{context_window // 1000}k"
            else:
                return f"{context_window // 1000}k"

        # Try to extract from model name
        model_lower = model_id.lower()

        if "128k" in model_lower or "131072" in model_lower:
            return "128k"
        elif "32k" in model_lower or "32768" in model_lower:
            return "32k"
        elif "8k" in model_lower or "8192" in model_lower:
            return "8k"
        elif "4k" in model_lower or "4096" in model_lower:
            return "4k"

        # Try to extract numeric context from model ID (e.g., mixtral-8x7b-32768)
        import re
        match = re.search(r'[-_](\d+)k?(?:[^0-9]|$)', model_lower)
        if match:
            num = int(match.group(1))
            if num > 1000:  # Likely token count, not model size
                return f"{num // 1000}k"

        return "?"

    def _interactive_selection(self, models):
        """Handle interactive model selection"""
        console.print("[bold cyan]Select models for parallel processing[/bold cyan]")
        console.print("[dim]Enter model numbers separated by spaces (e.g., 1 3 5 7)[/dim]")
        console.print("[dim]Tip: Select 3-4 different models for best parallel performance[/dim]\n")

        # Get current parallel models
        current = self.config.get('ai.parallel_models', [])
        if current:
            console.print(f"[dim]Current parallel models: {len(current)}[/dim]")
            for m in current:
                console.print(f"  [dim]- {m}[/dim]")
            console.print()

        # Prompt for selection
        selection = Prompt.ask(
            "[cyan]Enter model numbers[/cyan]",
            default=""
        )

        if not selection.strip():
            console.print("[yellow]No models selected, keeping current config[/yellow]")
            return

        # Parse selection
        try:
            indices = [int(x.strip()) for x in selection.split()]
            selected_models = []

            for idx in indices:
                if 1 <= idx <= len(models):
                    model_id = models[idx - 1].get("id")
                    selected_models.append(model_id)
                else:
                    console.print(f"[yellow]Warning: Index {idx} out of range, skipping[/yellow]")

            if not selected_models:
                console.print("[red]No valid models selected[/red]")
                return

            # Show selection
            console.print(f"\n[green]Selected {len(selected_models)} models:[/green]")
            for model in selected_models:
                console.print(f"  [cyan]✓[/cyan] {model}")
            console.print()

            # Confirm update
            if Confirm.ask("Update .gai.yaml with these models?", default=True):
                self._update_config(selected_models)
            else:
                console.print("[yellow]Config not updated[/yellow]")

        except ValueError as e:
            console.print(f"[red]Invalid input: {e}[/red]")
            console.print("[dim]Please enter numbers separated by spaces[/dim]")

    def _update_config(self, models):
        """Update .gai.yaml with selected models"""
        config_path = Path(self.config.config_path)

        if not config_path.exists():
            console.print(f"[red]Config file not found: {config_path}[/red]")
            return

        try:
            # Read current config
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)

            # Update parallel_models
            if 'ai' not in config_data:
                config_data['ai'] = {}

            config_data['ai']['parallel_models'] = models

            # Write back
            with open(config_path, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)

            console.print(f"\n[green]✓ Updated {config_path}[/green]")
            console.print(f"[dim]Set {len(models)} parallel models[/dim]\n")

        except Exception as e:
            console.print(f"[red]Error updating config: {e}[/red]")
