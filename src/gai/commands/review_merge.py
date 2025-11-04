"""
Review merge command - AI-powered merge preview
"""

from gai.core.git import Git, GitError
from gai.core.config import Config
from gai.core.tokens import estimate_tokens, chunk_by_files
from gai.core.stats import get_stats
from gai.ai.groq_client import GroqClient, GroqError
from gai.ai.prompts import Prompts
from gai.ui.interactive import (
    show_merge_analysis,
    show_error,
    show_warning,
    show_info,
    show_success,
    confirm
)
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


class ReviewMergeCommand:
    """Preview and analyze merge with AI"""

    def __init__(self, config: Config, git: Git, verbose: bool = False):
        self.config = config
        self.git = git
        self.verbose = verbose

        # Initialize AI client
        self.client = GroqClient(
            api_key=config.api_key,
            model=config.get('ai.model'),
            fallback_models=config.get('ai.fallback_models', []),
            temperature=config.get('ai.temperature', 0.3),
            verbose=verbose,
            api_url=config.get('ai.api_url', 'https://api.groq.com/openai/v1/chat/completions')
        )

    def run(self, base_branch: str, show_verbose: bool = False):
        """
        Run review-merge command.

        Args:
            base_branch: Base branch to merge into
            show_verbose: Show verbose output
        """
        # Record command usage
        get_stats().record_command("review_merge")

        try:
            current_branch = self.git.get_current_branch()

            if current_branch == base_branch:
                show_error(f"Already on {base_branch}")
                return

            console.print()
            console.print(f"[bold cyan]Analyzing merge:[/bold cyan] {current_branch} → {base_branch}")
            console.print()

            # Get merge base
            if self.verbose or show_verbose:
                show_info(f"Finding merge base between {current_branch} and {base_branch}...")

            try:
                merge_base = self.git.get_merge_base(current_branch, base_branch)
            except GitError:
                show_error(f"Cannot find merge base with {base_branch}")
                show_info(f"Make sure {base_branch} exists and has common history")
                return

            # Get commits ahead
            commits = self.git.get_commits_between(merge_base, current_branch)

            if not commits:
                show_info(f"No commits ahead of {base_branch}")
                return

            # Get changed files
            changed_files = self.git.get_changed_files(base_branch)

            # Get diff
            diff = self.git.get_branch_diff(base_branch)
            diff_stat = self.git.get_file_changes_stat(base_branch)

            # Show basic stats
            self._show_stats(
                current_branch=current_branch,
                base_branch=base_branch,
                num_commits=len(commits),
                num_files=len(changed_files),
                diff_stat=diff_stat
            )

            # Check size
            tokens = estimate_tokens(diff)
            max_tokens = self.config.get('review_merge.max_tokens', 30000)

            if self.verbose or show_verbose:
                show_info(f"Estimated tokens: {tokens:,} / {max_tokens:,}")

            # Handle large diffs
            if tokens > max_tokens:
                show_warning(f"Diff too large ({tokens:,} tokens)")

                if not confirm("Generate summary from file stats only?", default=True):
                    return

                # Use only stats, not full diff
                diff = "... diff too large, analyzing from stats only ..."

            # Get commit messages
            commit_messages = []
            for commit_hash in commits[:50]:  # Limit to 50 commits
                msg = self.git.get_commit_message(commit_hash)
                # Take first line only
                first_line = msg.split('\n')[0]
                commit_messages.append(first_line)

            # Generate AI analysis
            if self.verbose or show_verbose:
                show_info("Generating AI analysis...")

            analysis = self._generate_analysis(
                current_branch=current_branch,
                base_branch=base_branch,
                commits=commit_messages,
                diff_stat=diff_stat,
                diff=diff,
                changed_files=changed_files
            )

            if analysis:
                show_merge_analysis(analysis, title=f"Merge Analysis: {current_branch} → {base_branch}")
            else:
                show_error("Failed to generate analysis")

            # Check for conflicts
            if self.config.get('review_merge.check_conflicts', True):
                conflicts = self.git.get_conflicts(base_branch)
                if conflicts:
                    console.print()
                    console.print("[bold red]⚠ Potential Conflicts Detected:[/bold red]")
                    for file, status in conflicts:
                        console.print(f"  - {file}: {status}")
                    console.print()

        except GitError as e:
            show_error(f"Git error: {e}")
        except GroqError as e:
            show_error(f"AI analysis failed: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
        except Exception as e:
            show_error(f"Unexpected error: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()

    def _show_stats(
        self,
        current_branch: str,
        base_branch: str,
        num_commits: int,
        num_files: int,
        diff_stat: str
    ):
        """Show merge statistics"""

        # Parse diff stat for additions/deletions
        additions = 0
        deletions = 0

        for line in diff_stat.split('\n'):
            if 'insertion' in line or 'deletion' in line:
                parts = line.split(',')
                for part in parts:
                    if 'insertion' in part:
                        try:
                            additions = int(part.split()[0])
                        except:
                            pass
                    if 'deletion' in part:
                        try:
                            deletions = int(part.split()[0])
                        except:
                            pass

        # Create stats table
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column(style="cyan")
        table.add_column(style="white")

        table.add_row("📊 Commits ahead:", f"{num_commits:,}")
        table.add_row("📁 Files changed:", f"{num_files:,}")
        table.add_row("➕ Lines added:", f"{additions:,}")
        table.add_row("➖ Lines deleted:", f"{deletions:,}")

        console.print(table)
        console.print()

    def _generate_analysis(
        self,
        current_branch: str,
        base_branch: str,
        commits: list[str],
        diff_stat: str,
        diff: str,
        changed_files: list[str]
    ) -> str:
        """Generate AI analysis of merge"""

        system_prompt, user_prompt = Prompts.review_merge(
            current_branch=current_branch,
            base_branch=base_branch,
            commits=commits,
            diff_stat=diff_stat,
            diff=diff,
            changed_files=changed_files
        )

        try:
            analysis = self.client.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                max_tokens=800,  # More tokens for detailed analysis
            )

            return analysis

        except Exception as e:
            if self.verbose:
                show_error(f"Generation failed: {e}")
            raise
