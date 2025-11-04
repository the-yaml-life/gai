"""
List available models command
"""

import yaml
from pathlib import Path
from gai.core.config import Config
from gai.ai.llm_factory import MultiBackendClient
from rich.console import Console
from rich.table import Table
from rich.prompt import Confirm
import questionary

console = Console()


class ModelsCommand:
    """List available models from API"""

    def __init__(self, config: Config, verbose: bool = False):
        self.config = config
        self.verbose = verbose

        # Initialize multi-backend client
        self.client = MultiBackendClient(
            config=config,
            verbose=verbose
        )

    def run(self, select=False, filter_tier=None, debug=False, filter_backend=None, filter_free=False):
        """
        List available models from all backends.

        Args:
            select: If True, allow interactive selection to update config
            filter_tier: Filter by specific tier if API provides it (e.g., flagship, fast, light, reasoning)
            debug: If True, show raw API response for debugging
            filter_backend: Filter by specific backend (groq, anannas, ollama)
            filter_free: If True, show only free models (those with :free suffix)
        """
        console.print("\n[bold cyan]Fetching available models from all backends...[/bold cyan]\n")

        # Get models from all backends (raw if debug mode)
        all_backends = self.client.list_models(raw=debug)

        if not all_backends:
            console.print("[yellow]No models found or API error[/yellow]")
            return

        # Debug mode: show raw response
        if debug:
            import json
            console.print("[bold yellow]DEBUG MODE - Raw API Responses:[/bold yellow]\n")
            console.print("[dim]Use this to see what fields the API actually returns[/dim]\n")
            for backend, models in all_backends.items():
                if filter_backend and backend != filter_backend:
                    continue
                console.print(f"[bold cyan]{backend.upper()} Backend:[/bold cyan]")
                console.print(f"[dim]Total models: {len(models)}[/dim]")
                if models:
                    console.print("\n[yellow]First model example (raw fields):[/yellow]")
                    console.print(json.dumps(models[0], indent=2))
                    console.print("\n[yellow]All unique fields across all models:[/yellow]")
                    all_fields = set()
                    for m in models:
                        all_fields.update(m.keys())
                    console.print(f"Fields: {sorted(all_fields)}")
                console.print("\n" + "="*60 + "\n")
            return

        # Flatten and filter for text generation models
        all_models = []
        for backend, models in all_backends.items():
            # Filter by backend if requested
            if filter_backend and backend != filter_backend:
                continue
            for model in models:
                model['backend'] = backend
                all_models.append(model)

        models = self._filter_text_models(all_models)

        if not models:
            console.print("[yellow]No text models found[/yellow]")
            return

        # Filter by tier if requested (only if API provides tier field)
        if filter_tier:
            models = [m for m in models if m.get("tier", "").lower() == filter_tier.lower()]
            console.print(f"[dim]Filtering by tier: {filter_tier}[/dim]\n")

            if not models:
                console.print(f"[yellow]No models found for tier: {filter_tier}[/yellow]")
                console.print(f"[dim]Note: Tier filtering only works if the API provides tier metadata[/dim]")
                return

        # Filter by free models if requested
        if filter_free:
            models = [m for m in models if ':free' in m.get("id", "").lower()]
            console.print(f"[dim]Filtering for free models only[/dim]\n")

            if not models:
                console.print(f"[yellow]No free models found[/yellow]")
                console.print(f"[dim]Free models are those with ':free' suffix in their ID[/dim]")
                return

        # Sort models: :free models first, then by backend, then by id
        # This makes free models more visible in the selection UI
        def sort_key(model):
            model_id = model.get("id", "")
            is_free = ':free' in model_id.lower()
            backend = model.get("backend", "unknown")
            # Sort order: free models first (0), then paid (1), then by backend, then by id
            return (0 if is_free else 1, backend, model_id)

        models.sort(key=sort_key)

        # Create table for display only (selection happens via checkboxes)
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Backend", style="magenta", width=8)
        table.add_column("Tier", style="green", width=10)
        table.add_column("Model ID", style="cyan", width=40)
        table.add_column("Context", style="yellow", justify="right", width=10)

        for model in models:
            backend = model.get("backend", "unknown")
            tier = model.get("tier", "standard")
            model_id = model.get("id", "unknown")
            context = self._get_context_length(model)

            # Color code tier
            tier_colored = self._format_tier(tier)
            table.add_row(backend, tier_colored, model_id, context)

        # Only show table if not in select mode (select mode shows checkboxes instead)
        if not select:
            console.print(table)
        console.print(f"\n[dim]Total: {len(models)} text models[/dim]")

        # Show count per backend
        backend_counts = {}
        tier_counts = {}
        for model in models:
            backend = model.get("backend", "unknown")
            tier = model.get("tier", "standard")
            backend_counts[backend] = backend_counts.get(backend, 0) + 1
            tier_counts[tier] = tier_counts.get(tier, 0) + 1

        for backend, count in backend_counts.items():
            console.print(f"[dim]  {backend}: {count} models[/dim]")

        console.print(f"\n[dim]By tier:[/dim]")
        for tier, count in sorted(tier_counts.items()):
            console.print(f"[dim]  {tier}: {count} models[/dim]")

        console.print()

        # Interactive selection
        if select:
            self._interactive_selection(models)

    def _filter_text_models(self, models):
        """
        Filter models to only include text generation models.

        Uses config.models.exclude_keywords if configured, otherwise uses defaults.
        """
        text_models = []

        # Get exclude keywords from config or use defaults
        exclude_keywords = self.config.get('models.exclude_keywords', [
            'whisper',         # Speech-to-text
            'tts',             # Text-to-speech
            'playai',          # TTS service
            'distil-whisper',  # Whisper variants
        ])

        for model in models:
            model_id = model.get("id", "").lower()

            # Skip if contains exclude keywords
            if any(keyword in model_id for keyword in exclude_keywords):
                continue

            text_models.append(model)

        return text_models

    def _format_tier(self, tier: str) -> str:
        """Format tier with color coding"""
        tier_colors = {
            "local": "[bright_green]local[/bright_green]",
            "flagship": "[magenta]flagship[/magenta]",
            "fast": "[cyan]fast[/cyan]",
            "light": "[yellow]light[/yellow]",
            "reasoning": "[blue]reasoning[/blue]",
            "standard": "[dim]standard[/dim]",
        }
        return tier_colors.get(tier.lower(), f"[dim]{tier}[/dim]")

    def _get_context_length(self, model):
        """Extract context length from model data or name"""
        model_id = model.get("id", "")

        # Check context_window field if available
        context_window = model.get("context_window")
        if context_window:
            if context_window >= 1000000:
                return f"{context_window // 1000000}M"
            elif context_window >= 1000:
                return f"{context_window // 1000}k"
            else:
                return str(context_window)

        # Try to extract from model name
        model_lower = model_id.lower()

        if "1m" in model_lower or "1000k" in model_lower:
            return "1M"
        elif "128k" in model_lower or "131072" in model_lower:
            return "128k"
        elif "100k" in model_lower:
            return "100k"
        elif "32k" in model_lower or "32768" in model_lower:
            return "32k"
        elif "16k" in model_lower or "16384" in model_lower:
            return "16k"
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
                if num >= 1000000:
                    return f"{num // 1000000}M"
                else:
                    return f"{num // 1000}k"

        return "?"

    def _interactive_selection(self, models):
        """Handle interactive model selection with checkboxes"""
        console.print("[bold cyan]Select models for parallel processing[/bold cyan]")
        console.print("[dim]Use ↑↓ arrows to move, SPACE to select, ENTER to confirm[/dim]")
        console.print("[dim]Tip: Select 3-4 different models for best parallel performance[/dim]")

        # Count and show free models
        free_count = sum(1 for m in models if ':free' in m.get("id", "").lower())
        if free_count > 0:
            console.print(f"[green]💰 {free_count} free models available (shown first)[/green]\n")
        else:
            console.print()

        # Get current parallel models to pre-select them
        current = self.config.get('ai.parallel_models', [])
        if current:
            console.print(f"[dim]Current parallel models: {len(current)}[/dim]")
            for m in current:
                console.print(f"  [dim]- {m}[/dim]")
            console.print()

        # Create choices with formatted display
        choices = []
        for model in models:
            backend = model.get("backend", "unknown")
            tier = model.get("tier", "standard")
            model_id = model.get("id", "unknown")
            context = self._get_context_length(model)

            # Format: "groq flagship llama-3.3-70b-versatile 128k"
            display = f"{backend:8} {tier:10} {model_id:40} {context:>6}"

            # Build full model identifier with backend prefix (for saving to config)
            # This ensures we can distinguish between groq/llama and anannas/llama
            if backend == "groq":
                full_model_id = model_id  # Groq is default, no prefix needed
            else:
                full_model_id = f"{backend}/{model_id}"

            # Check if this model is currently selected
            # Support both old format (no prefix) and new format (with prefix)
            is_selected = full_model_id in current or model_id in current

            choices.append(questionary.Choice(
                title=display,
                value=full_model_id,
                checked=is_selected
            ))

        # Show checkbox selection
        selected_models = questionary.checkbox(
            "Select models:",
            choices=choices,
            style=questionary.Style([
                ('checkbox-selected', 'fg:#00ff00 bold'),  # Green for selected
                ('checkbox', 'fg:#858585'),                # Gray for unselected
                ('selected', 'bg:#0080ff fg:#ffffff'),     # Blue background for cursor
                ('pointer', 'fg:#00ff00 bold'),            # Green pointer
            ])
        ).ask()

        if not selected_models:
            console.print("[yellow]No models selected, keeping current config[/yellow]")
            return

        # Show selection summary
        console.print(f"\n[green]Selected {len(selected_models)} models:[/green]")
        for model in selected_models:
            console.print(f"  [cyan]✓[/cyan] {model}")
        console.print()

        # Confirm update
        if Confirm.ask("Update .gai.yaml with these models?", default=True):
            self._update_config(selected_models)
        else:
            console.print("[yellow]Config not updated[/yellow]")

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
