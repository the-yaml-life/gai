"""
Diff analysis command - AI summary of changes
"""

from gai.core.git import Git, GitError
from gai.core.config import Config
from gai.core.tokens import estimate_tokens
from gai.core.stats import get_stats
from gai.ai.llm_factory import MultiBackendClient, LLMError
from gai.ai.prompts import Prompts
from gai.ui.interactive import (
    show_diff_summary,
    show_error,
    show_warning,
    show_info
)
from rich.console import Console
from rich.table import Table

console = Console()


class DiffCommand:
    """AI summary of branch differences"""

    def __init__(self, config: Config, git: Git, verbose: bool = False):
        self.config = config
        self.git = git
        self.verbose = verbose

        # Initialize AI client (multi-backend support)
        self.client = MultiBackendClient(
            config=config,
            verbose=verbose
        )

    def run(
        self,
        branch: str = None,
        show_stat: bool = False,
        show_files: bool = False
    ):
        """
        Run diff command.

        Args:
            branch: Branch to compare (default: current vs HEAD)
            show_stat: Show file statistics
            show_files: List changed files
        """
        # Record command usage
        get_stats().record_command("diff")

        try:
            current_branch = self.git.get_current_branch()

            # Determine comparison
            if branch:
                branch1 = branch
                branch2 = current_branch
                title = f"{branch} vs {current_branch}"
            else:
                # Compare staged changes
                branch1 = "HEAD"
                branch2 = "working tree"
                title = "Staged changes"

            console.print()
            console.print(f"[bold cyan]Analyzing diff:[/bold cyan] {title}")
            console.print()

            # Get diff
            if branch:
                try:
                    merge_base = self.git.get_merge_base(branch1, branch2)
                except GitError:
                    show_error(f"Cannot compare {branch1} and {branch2}")
                    return

                diff = self.git.get_branch_diff(branch1, branch2)
                diff_stat = self.git.get_file_changes_stat(branch1, branch2)
                changed_files = self.git.get_changed_files(branch1, branch2)
            else:
                # Staged changes
                diff = self.git.get_diff(cached=True)
                diff_stat = self.git.get_diff_stat(cached=True)

                status = self.git.get_status()
                changed_files = [
                    line.split()[1]
                    for line in status.split('\n')
                    if line.strip()
                ]

            if not diff.strip():
                show_info("No changes to analyze")
                return

            # Show files if requested
            if show_files:
                self._show_files(changed_files)

            # Show stats if requested
            if show_stat:
                console.print(diff_stat)
                console.print()

            # Check size
            tokens = estimate_tokens(diff)
            max_tokens = self.config.get('diff.max_tokens', 30000)

            if self.verbose:
                show_info(f"Estimated tokens: {tokens:,} / {max_tokens:,}")

            if tokens > max_tokens:
                show_warning(f"Diff too large ({tokens:,} tokens)")
                show_info("Analyzing from stats only...")
                diff = "... diff too large ..."

            # Generate AI summary
            if self.verbose:
                show_info("Generating AI summary...")

            summary = self._generate_summary(
                branch1=branch1,
                branch2=branch2,
                diff_stat=diff_stat,
                diff=diff,
                changed_files=changed_files
            )

            if summary:
                show_diff_summary(summary, title=f"Diff Summary: {title}")
            else:
                show_error("Failed to generate summary")

        except GitError as e:
            show_error(f"Git error: {e}")
        except LLMError as e:
            show_error(f"AI analysis failed: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
        except Exception as e:
            show_error(f"Unexpected error: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()

    def _show_files(self, files: list[str]):
        """Show changed files"""
        console.print("[bold]Changed files:[/bold]")
        for f in files[:20]:
            console.print(f"  • {f}")

        if len(files) > 20:
            console.print(f"  ... and {len(files) - 20} more")

        console.print()

    def _generate_summary(
        self,
        branch1: str,
        branch2: str,
        diff_stat: str,
        diff: str,
        changed_files: list[str]
    ) -> str:
        """Generate AI summary of diff"""

        style = self.config.get('diff.summary_style', 'detailed')

        system_prompt, user_prompt = Prompts.diff_analysis(
            branch1=branch1,
            branch2=branch2,
            diff_stat=diff_stat,
            diff=diff,
            changed_files=changed_files,
            style=style
        )

        try:
            summary = self.client.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                max_tokens=600
            )

            return summary

        except Exception as e:
            if self.verbose:
                show_error(f"Generation failed: {e}")
            raise
