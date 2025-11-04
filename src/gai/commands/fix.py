"""
Fix command - Execute git problem solutions
"""

from typing import Optional
from rich.console import Console
from rich.panel import Panel
import click

from gai.core.config import Config
from gai.core.git import Git, GitError


class FixCommand:
    """Execute solutions for git problems"""

    def __init__(self, config: Config, git: Git, verbose: bool = False):
        self.config = config
        self.git = git
        self.verbose = verbose
        self.console = Console()

    def run(self, rebase: bool = False, merge: bool = False, reset: bool = False,
            pull: bool = False, auto: bool = False, force: bool = False):
        """
        Main fix flow.

        Args:
            rebase: Rebase local commits on top of remote
            merge: Merge remote changes with local
            reset: Reset to match remote (discard local)
            pull: Simple pull (for when just behind)
            auto: Automatically choose best solution
            force: Skip confirmation prompts
        """
        self.console.print()

        # Detect problem
        problem = self._detect_problem()

        if not problem:
            self.console.print("[green]✓ No issues detected[/green]")
            self.console.print()
            return

        problem_type = problem['type']
        data = problem['data']

        # Auto mode: choose best solution
        if auto:
            rebase, merge, reset, pull = self._auto_choose(problem_type, data)
            force = True  # Auto implies no prompts

        # Route to appropriate fix
        if problem_type == 'divergent_branches':
            success = self._fix_divergent_branches(data, rebase, merge, reset, force, auto)

            # If auto mode failed, offer manual options
            if auto and not success:
                self.console.print()
                self.console.print("[yellow]⚠️  Automatic fix failed (likely conflicts)[/yellow]")
                self.console.print()
                self.console.print("[bold]What would you like to do?[/bold]")
                self.console.print("  1. View conflicts and resolve manually")
                self.console.print("  2. Abort and keep current state")
                self.console.print("  3. Reset to remote (discard local changes)")
                self.console.print()

                choice = click.prompt("Choice", type=click.Choice(['1', '2', '3']), default='1')

                if choice == '1':
                    # Use gai resolve
                    self.console.print()
                    self.console.print("[cyan]→ Launching conflict resolver...[/cyan]")
                    from gai.commands.resolve import ResolveCommand
                    resolve_cmd = ResolveCommand(self.config, self.git, self.verbose)
                    resolve_cmd.run(auto=False)
                    return
                elif choice == '2':
                    self.console.print("[cyan]→ Aborting...[/cyan]")
                    try:
                        self.git._run("merge", "--abort", check=False)
                        self.git._run("rebase", "--abort", check=False)
                        self.console.print("[green]✓ Aborted[/green]")
                    except:
                        pass
                elif choice == '3':
                    self._execute_reset(data['ahead'], data['behind'], False)

        elif problem_type == 'behind_remote':
            self._fix_behind_remote(data, force)
        elif problem_type == 'ahead_remote':
            self._fix_ahead_remote(data, force)
        else:
            self.console.print(f"[yellow]No automatic fix available for: {problem_type}[/yellow]")
            self.console.print()

    def _detect_problem(self) -> Optional[dict]:
        """Detect git problems (same as debug command)"""
        try:
            div = self.git.get_divergence_info()
            if div['diverged']:
                return {
                    'type': 'divergent_branches',
                    'severity': 'warning',
                    'data': div
                }
            elif div['behind'] > 0 and div['ahead'] == 0:
                return {
                    'type': 'behind_remote',
                    'severity': 'info',
                    'data': div
                }
            elif div['ahead'] > 0 and div['behind'] == 0:
                return {
                    'type': 'ahead_remote',
                    'severity': 'info',
                    'data': div
                }
        except GitError as e:
            if self.verbose:
                self.console.print(f"[dim]Error: {e}[/dim]")

        return None

    def _auto_choose(self, problem_type: str, data: dict) -> tuple:
        """
        Automatically choose best solution based on problem type.

        Returns: (rebase, merge, reset, pull, try_fallback)
        """
        if problem_type == 'divergent_branches':
            # Divergent: always try merge for auto mode (safer)
            # Rebase can have conflicts that require manual intervention
            self.console.print("[cyan]🤖 Auto-choosing: Merge (safest for auto mode)[/cyan]")
            return (False, True, False, False)

        elif problem_type == 'behind_remote':
            # Just behind: simple pull
            self.console.print("[cyan]🤖 Auto-choosing: Pull[/cyan]")
            return (False, False, False, True)

        elif problem_type == 'ahead_remote':
            # Just ahead: push (handled separately)
            self.console.print("[cyan]🤖 Auto-choosing: Push[/cyan]")
            return (False, False, False, False)

        # Default: do nothing
        return (False, False, False, False)

    def _fix_divergent_branches(self, data: dict, rebase: bool, merge: bool,
                                 reset: bool, force: bool, auto: bool = False) -> bool:
        """
        Fix divergent branches.

        Returns: True if successful, False if failed
        """
        branch = self.git.get_current_branch()
        ahead = data['ahead']
        behind = data['behind']

        # Show status
        self.console.print(f"[yellow]⚠️  Divergent branches detected[/yellow]")
        self.console.print(f"  Branch: {branch}")
        self.console.print(f"  Ahead:  {ahead} commit{'s' if ahead != 1 else ''}")
        self.console.print(f"  Behind: {behind} commit{'s' if behind != 1 else ''}")
        self.console.print()

        # Determine action
        if not any([rebase, merge, reset]):
            # No flag specified, ask user
            self.console.print("[bold]Choose a solution:[/bold]")
            self.console.print("  1. Rebase  - Apply your commits on top of remote (clean history)")
            self.console.print("  2. Merge   - Combine with merge commit (safer)")
            self.console.print("  3. Reset   - Discard local commits, match remote")
            self.console.print()

            choice = click.prompt("Choice", type=click.Choice(['1', '2', '3', 'q']), default='1')

            if choice == 'q':
                self.console.print("[dim]Cancelled[/dim]")
                return False
            elif choice == '1':
                rebase = True
            elif choice == '2':
                merge = True
            elif choice == '3':
                reset = True

        # Execute chosen action
        if rebase:
            return self._execute_rebase(branch, ahead, behind, force)
        elif merge:
            return self._execute_merge(branch, ahead, behind, force)
        elif reset:
            return self._execute_reset(branch, ahead, behind, force)

        return False

    def _execute_rebase(self, branch: str, ahead: int, behind: int, force: bool) -> bool:
        """Execute rebase. Returns True if successful."""
        self.console.print("[cyan]→ Rebasing...[/cyan]")
        self.console.print()

        # Check for uncommitted changes
        if self.git.has_uncommitted_changes():
            self.console.print("[yellow]⚠️  You have uncommitted changes[/yellow]")
            self.console.print()

            if not force and not click.confirm("Stash changes and continue?"):
                self.console.print("[dim]Cancelled[/dim]")
                return

            try:
                self.git._run("stash", "push", "-m", "gai fix: auto-stash before rebase")
                self.console.print("[green]✓[/green] Changes stashed")
                stashed = True
            except GitError as e:
                self.console.print(f"[red]Error stashing: {e}[/red]")
                return False
        else:
            stashed = False

        # Confirm if not forced
        if not force:
            self.console.print(f"[bold]This will:[/bold]")
            self.console.print(f"  • Fetch latest changes from remote")
            self.console.print(f"  • Reapply your {ahead} commit{'s' if ahead != 1 else ''} on top")
            self.console.print()

            if not click.confirm("Continue?", default=True):
                if stashed:
                    self.git._run("stash", "pop")
                    self.console.print("[green]✓[/green] Changes restored")
                self.console.print("[dim]Cancelled[/dim]")
                return False

        # Execute rebase
        try:
            self.git.pull_rebase()
            self.console.print()
            self.console.print("[green]✓ Rebase successful![/green]")
            self.console.print()
            self.console.print("[bold]Next steps:[/bold]")
            self.console.print(f"  • Review: [cyan]git log --oneline -10[/cyan]")
            self.console.print(f"  • Push:   [cyan]git push origin {branch}[/cyan]")

            if stashed:
                self.console.print()
                self.console.print("[yellow]Don't forget to restore your stashed changes:[/yellow]")
                self.console.print("  [cyan]git stash pop[/cyan]")

            self.console.print()
            return True

        except GitError as e:
            self.console.print()
            self.console.print(f"[red]✗ Rebase failed: {e}[/red]")
            self.console.print()
            self.console.print("[bold]To abort the rebase:[/bold]")
            self.console.print("  [cyan]git rebase --abort[/cyan]")

            if stashed:
                self.console.print()
                self.console.print("[bold]To restore your changes:[/bold]")
                self.console.print("  [cyan]git stash pop[/cyan]")

            self.console.print()
            return False

    def _execute_merge(self, branch: str, ahead: int, behind: int, force: bool) -> bool:
        """Execute merge. Returns True if successful."""
        self.console.print("[cyan]→ Merging...[/cyan]")
        self.console.print()

        # Confirm if not forced
        if not force:
            self.console.print(f"[bold]This will:[/bold]")
            self.console.print(f"  • Pull and merge remote changes")
            self.console.print(f"  • Create a merge commit")
            self.console.print(f"  • Preserve all history")
            self.console.print()

            if not click.confirm("Continue?", default=True):
                self.console.print("[dim]Cancelled[/dim]")
                return False

        # Execute merge
        try:
            self.git.pull_merge()
            self.console.print()
            self.console.print("[green]✓ Merge successful![/green]")
            self.console.print()
            self.console.print("[bold]Next steps:[/bold]")
            self.console.print(f"  • Review: [cyan]git log --oneline -10[/cyan]")
            self.console.print(f"  • Push:   [cyan]git push origin {branch}[/cyan]")
            self.console.print()
            return True

        except GitError as e:
            self.console.print()
            self.console.print(f"[red]✗ Merge failed: {e}[/red]")
            self.console.print()
            self.console.print("[bold]To abort the merge:[/bold]")
            self.console.print("  [cyan]git merge --abort[/cyan]")
            self.console.print()
            return False

    def _execute_reset(self, ahead: int, behind: int, force: bool) -> bool:
        """Execute reset (DANGEROUS). Returns True if successful."""
        branch = self.git.get_current_branch()
        self.console.print("[red]⚠️  WARNING: DANGEROUS OPERATION[/red]")
        self.console.print()
        self.console.print(f"[bold]This will:[/bold]")
        self.console.print(f"  • [red]DELETE your {ahead} local commit{'s' if ahead != 1 else ''}[/red]")
        self.console.print(f"  • [red]LOSE all local changes[/red]")
        self.console.print(f"  • Reset to match remote exactly")
        self.console.print()

        # Always confirm for reset, even with force
        if not click.confirm("Are you ABSOLUTELY SURE?", default=False):
            self.console.print("[dim]Cancelled (good choice)[/dim]")
            return False

        self.console.print()
        self.console.print("[cyan]→ Resetting...[/cyan]")

        try:
            self.git.fetch()
            self.git.reset_hard(f"origin/{branch}")
            self.console.print()
            self.console.print("[green]✓ Reset complete[/green]")
            self.console.print()
            self.console.print(f"Your branch now matches origin/{branch}")
            self.console.print()
            return True

        except GitError as e:
            self.console.print()
            self.console.print(f"[red]✗ Reset failed: {e}[/red]")
            self.console.print()
            return False

    def _fix_behind_remote(self, data: dict, force: bool):
        """Fix when behind remote (simple pull)"""
        branch = self.git.get_current_branch()
        behind = data['behind']

        self.console.print(f"[blue]ℹ️  Behind remote by {behind} commit{'s' if behind != 1 else ''}[/blue]")
        self.console.print()

        if not force and not click.confirm("Pull changes from remote?", default=True):
            self.console.print("[dim]Cancelled[/dim]")
            return

        self.console.print("[cyan]→ Pulling...[/cyan]")

        try:
            self.git._run("pull")
            self.console.print()
            self.console.print("[green]✓ Pull successful![/green]")
            self.console.print()
            self.console.print("Your branch is now up to date")

        except GitError as e:
            self.console.print()
            self.console.print(f"[red]✗ Pull failed: {e}[/red]")

        self.console.print()

    def _fix_ahead_remote(self, data: dict, force: bool):
        """Fix when ahead of remote (push)"""
        branch = self.git.get_current_branch()
        ahead = data['ahead']

        self.console.print(f"[blue]ℹ️  Ahead of remote by {ahead} commit{'s' if ahead != 1 else ''}[/blue]")
        self.console.print()

        if not force and not click.confirm("Push commits to remote?", default=True):
            self.console.print("[dim]Cancelled[/dim]")
            return

        self.console.print("[cyan]→ Pushing...[/cyan]")

        try:
            self.git.push()
            self.console.print()
            self.console.print("[green]✓ Push successful![/green]")
            self.console.print()
            self.console.print(f"Your commits are now on origin/{branch}")

        except GitError as e:
            self.console.print()
            self.console.print(f"[red]✗ Push failed: {e}[/red]")

        self.console.print()

    def _show_conflict_help(self):
        """Show help for resolving conflicts"""
        self.console.print()
        self.console.print("[bold cyan]How to resolve conflicts:[/bold cyan]")
        self.console.print()
        self.console.print("1. Check conflicted files:")
        self.console.print("   [cyan]git status[/cyan]")
        self.console.print()
        self.console.print("2. Open each file and look for conflict markers:")
        self.console.print("   [dim]<<<<<<< HEAD[/dim]")
        self.console.print("   [dim]your changes[/dim]")
        self.console.print("   [dim]=======[/dim]")
        self.console.print("   [dim]remote changes[/dim]")
        self.console.print("   [dim]>>>>>>> origin/master[/dim]")
        self.console.print()
        self.console.print("3. Edit files to resolve conflicts")
        self.console.print()
        self.console.print("4. Mark as resolved:")
        self.console.print("   [cyan]git add <file>[/cyan]")
        self.console.print()
        self.console.print("5. Continue merge:")
        self.console.print("   [cyan]git merge --continue[/cyan]")
        self.console.print()
        self.console.print("Or to abort:")
        self.console.print("   [cyan]git merge --abort[/cyan]")
        self.console.print()
