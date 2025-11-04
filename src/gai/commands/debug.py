"""
Debug command - Explain git problems without fixing them
"""

from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from gai.core.config import Config
from gai.core.git import Git, GitError
from gai.ai.llm_factory import MultiBackendClient, LLMError


class DebugCommand:
    """Detect and explain git problems"""

    def __init__(self, config: Config, git: Git, verbose: bool = False):
        self.config = config
        self.git = git
        self.verbose = verbose
        self.console = Console()
        self.client = MultiBackendClient(config, verbose)

    def run(self):
        """Main debug flow"""
        self.console.print()
        self.console.print("[cyan]🔍 Analyzing repository...[/cyan]")
        self.console.print()

        # Detect problems
        problem = self._detect_problem()

        if not problem:
            self.console.print(Panel(
                "[green]✓ No issues detected[/green]\n\n"
                "Your repository looks healthy!",
                title="Status",
                border_style="green"
            ))
            return

        # Show problem and get AI analysis
        self._show_problem(problem)

    def _detect_problem(self) -> Optional[dict]:
        """
        Detect common git problems.

        Returns dict with:
            - type: str (divergent_branches, merge_conflict, etc)
            - severity: str (info, warning, error)
            - data: dict (problem-specific data)
        """
        # Check for divergent branches
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
                self.console.print(f"[dim]Error checking divergence: {e}[/dim]")

        # Check for uncommitted changes
        if self.git.has_uncommitted_changes():
            return {
                'type': 'uncommitted_changes',
                'severity': 'info',
                'data': {}
            }

        # No problems detected
        return None

    def _show_problem(self, problem: dict):
        """Show problem details and AI analysis"""
        problem_type = problem['type']
        severity = problem['severity']
        data = problem['data']

        # Icon and color based on severity
        if severity == 'error':
            icon = "❌"
            color = "red"
        elif severity == 'warning':
            icon = "⚠️"
            color = "yellow"
        else:
            icon = "ℹ️"
            color = "blue"

        # Show problem based on type
        if problem_type == 'divergent_branches':
            self._show_divergent_branches(data, icon, color)
        elif problem_type == 'behind_remote':
            self._show_behind_remote(data, icon, color)
        elif problem_type == 'ahead_remote':
            self._show_ahead_remote(data, icon, color)
        elif problem_type == 'uncommitted_changes':
            self._show_uncommitted_changes(icon, color)

    def _show_divergent_branches(self, data: dict, icon: str, color: str):
        """Show divergent branches problem"""
        branch = self.git.get_current_branch()
        ahead = data['ahead']
        behind = data['behind']

        # Build header
        header = f"{icon} Issue detected: Divergent Branches"

        # Get commit summaries
        ahead_commits = []
        for commit in data['ahead_commits'][:5]:  # Max 5
            summary = self.git.get_commit_summary(commit)
            ahead_commits.append(f"  {summary}")

        behind_commits = []
        for commit in data['behind_commits'][:5]:  # Max 5
            summary = self.git.get_commit_summary(commit)
            behind_commits.append(f"  {summary}")

        # Build content
        content = f"Your branch '[cyan]{branch}[/cyan]' has diverged from '[cyan]origin/{branch}[/cyan]'\n"
        content += f"  • Local:  {ahead} commit{'s' if ahead != 1 else ''} ahead\n"
        content += f"  • Remote: {behind} commit{'s' if behind != 1 else ''} behind\n\n"

        if ahead_commits:
            content += "[bold]Local commits (not on remote):[/bold]\n"
            content += "\n".join(ahead_commits[:3])
            if ahead > 3:
                content += f"\n  ... and {ahead - 3} more"
            content += "\n\n"

        if behind_commits:
            content += "[bold]Remote commits (not local):[/bold]\n"
            content += "\n".join(behind_commits[:3])
            if behind > 3:
                content += f"\n  ... and {behind - 3} more"
            content += "\n"

        self.console.print(Panel(content, title=header, border_style=color))
        self.console.print()

        # Get AI analysis
        self._show_ai_analysis('divergent_branches', data)

    def _show_behind_remote(self, data: dict, icon: str, color: str):
        """Show behind remote problem"""
        branch = self.git.get_current_branch()
        behind = data['behind']

        behind_commits = []
        for commit in data['behind_commits'][:5]:
            summary = self.git.get_commit_summary(commit)
            behind_commits.append(f"  {summary}")

        content = f"Your branch '[cyan]{branch}[/cyan]' is behind '[cyan]origin/{branch}[/cyan]'\n"
        content += f"  • Remote has {behind} commit{'s' if behind != 1 else ''} you don't have\n\n"

        if behind_commits:
            content += "[bold]Remote commits:[/bold]\n"
            content += "\n".join(behind_commits[:3])
            if behind > 3:
                content += f"\n  ... and {behind - 3} more"

        self.console.print(Panel(content, title=f"{icon} Info: Behind Remote", border_style=color))
        self.console.print()

        # Simple solution
        self.console.print("[bold]Solution:[/bold]")
        self.console.print(f"  Run: [cyan]gai fix --pull[/cyan]")
        self.console.print(f"  Or:  [cyan]git pull origin {branch}[/cyan]")
        self.console.print()

    def _show_ahead_remote(self, data: dict, icon: str, color: str):
        """Show ahead of remote problem"""
        branch = self.git.get_current_branch()
        ahead = data['ahead']

        ahead_commits = []
        for commit in data['ahead_commits'][:5]:
            summary = self.git.get_commit_summary(commit)
            ahead_commits.append(f"  {summary}")

        content = f"Your branch '[cyan]{branch}[/cyan]' is ahead of '[cyan]origin/{branch}[/cyan]'\n"
        content += f"  • You have {ahead} commit{'s' if ahead != 1 else ''} not pushed\n\n"

        if ahead_commits:
            content += "[bold]Local commits:[/bold]\n"
            content += "\n".join(ahead_commits[:3])
            if ahead > 3:
                content += f"\n  ... and {ahead - 3} more"

        self.console.print(Panel(content, title=f"{icon} Info: Ahead of Remote", border_style=color))
        self.console.print()

        # Simple solution
        self.console.print("[bold]Solution:[/bold]")
        self.console.print(f"  Run: [cyan]git push origin {branch}[/cyan]")
        self.console.print()

    def _show_uncommitted_changes(self, icon: str, color: str):
        """Show uncommitted changes info"""
        status = self.git.get_status()
        lines = status.strip().split('\n')
        count = len(lines)

        content = f"You have {count} uncommitted change{'s' if count != 1 else ''}\n\n"
        content += "[bold]Changed files:[/bold]\n"

        for line in lines[:10]:
            content += f"  {line}\n"

        if count > 10:
            content += f"  ... and {count - 10} more\n"

        self.console.print(Panel(content, title=f"{icon} Info: Uncommitted Changes", border_style=color))
        self.console.print()

        self.console.print("[bold]Next steps:[/bold]")
        self.console.print("  • Commit: [cyan]gai commit[/cyan]")
        self.console.print("  • Stash:  [cyan]git stash[/cyan]")
        self.console.print("  • Discard: [cyan]git restore .[/cyan]")
        self.console.print()

    def _show_ai_analysis(self, problem_type: str, data: dict):
        """Get and show AI analysis of the problem"""
        self.console.print("[cyan]🤖 AI Analysis:[/cyan]")
        self.console.print()

        # Build prompt based on problem type
        if problem_type == 'divergent_branches':
            prompt = self._build_divergent_prompt(data)
        else:
            return  # No AI analysis for other types yet

        try:
            # Get AI response
            system_prompt = "You are a git expert helping developers understand and resolve git problems. Be concise and practical."

            response = self.client.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=800,
                temperature=0.3
            )

            # Show analysis
            self.console.print(Panel(
                response.strip(),
                title="Analysis",
                border_style="cyan"
            ))
            self.console.print()

            # Show next steps
            self.console.print("[bold]To fix this issue, run:[/bold]")
            self.console.print("  [cyan]gai fix --auto[/cyan]    (let AI decide)")
            self.console.print("  [cyan]gai fix --rebase[/cyan]  (recommended)")
            self.console.print("  [cyan]gai fix --merge[/cyan]   (safer)")
            self.console.print("  [cyan]gai fix --reset[/cyan]   (discard local changes)")
            self.console.print()

        except LLMError as e:
            self.console.print(f"[yellow]Could not get AI analysis: {e}[/yellow]")
            self.console.print()

    def _build_divergent_prompt(self, data: dict) -> str:
        """Build prompt for divergent branches analysis"""
        branch = self.git.get_current_branch()
        ahead = data['ahead']
        behind = data['behind']

        # Get commit summaries
        local_commits = []
        for commit in data['ahead_commits'][:3]:
            summary = self.git.get_commit_summary(commit)
            local_commits.append(summary)

        remote_commits = []
        for commit in data['behind_commits'][:3]:
            summary = self.git.get_commit_summary(commit)
            remote_commits.append(summary)

        prompt = f"""Analyze this git situation:

Branch: {branch}
Local commits (ahead): {ahead}
Remote commits (behind): {behind}

Local commits:
{chr(10).join(local_commits) if local_commits else '(none)'}

Remote commits:
{chr(10).join(remote_commits) if remote_commits else '(none)'}

Explain in 2-3 sentences:
1. What happened (why did this divergence occur?)
2. Which option is best (rebase, merge, or reset) and why
3. Any risks to be aware of

Keep it practical and concise."""

        return prompt
