"""
Version command - Show current version and suggest next
"""

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from gai.core.config import Config
from gai.core.git import Git
from gai.core.versioning import VersionManager, BumpType


class VersionCommand:
    """Show version information"""

    def __init__(self, config: Config, git: Git, verbose: bool = False):
        self.config = config
        self.git = git
        self.verbose = verbose
        self.console = Console()
        # version_manager will be created after project selection

    def run(self, show_commits: bool = False):
        """
        Run version command.

        Args:
            show_commits: Show commits since last version
        """
        self.console.print()
        self.console.print("[cyan]Version Information[/cyan]")
        self.console.print()

        # Detect projects from tags
        temp_vm = VersionManager(self.git)
        projects = temp_vm.list_projects()

        # Select project (interactive if multiple)
        selected_project = None
        if projects:
            if len(projects) == 1:
                selected_project = projects[0]
                self.console.print(f"[dim]Project:[/dim] {selected_project}")
                self.console.print()
            else:
                # Multiple projects - show menu
                self.console.print("[cyan]Multiple projects detected:[/cyan]")
                self.console.print()

                for idx, proj in enumerate(projects, 1):
                    # Get current version for this project
                    proj_vm = VersionManager(self.git, project_name=proj)
                    proj_version = proj_vm.get_current_version()
                    version_str = str(proj_version) if proj_version else "[dim]no version[/dim]"
                    self.console.print(f"  [{idx}] {proj} ({version_str})")

                self.console.print()

                # Ask user to select
                import click
                while True:
                    choice = click.prompt(
                        "Select project number",
                        type=int,
                        default=1
                    )
                    if 1 <= choice <= len(projects):
                        selected_project = projects[choice - 1]
                        break
                    else:
                        self.console.print(f"[red]Invalid choice. Please enter 1-{len(projects)}[/red]")

                self.console.print()
                self.console.print(f"[cyan]Selected:[/cyan] {selected_project}")
                self.console.print()

        # Create version manager with selected project
        version_manager = VersionManager(self.git, project_name=selected_project)

        # Get current version
        current = version_manager.get_current_version()

        if current is None:
            self.console.print(Panel(
                "[yellow]No version tags found[/yellow]\n\n"
                "This repository doesn't have any version tags yet.\n"
                "Use [cyan]gai release[/cyan] to create your first release (v0.1.0).",
                title="Status",
                border_style="yellow"
            ))
            self.console.print()
            return

        # Get commits since last version
        commits = version_manager.get_commits_since_tag()

        # Detect suggested bump
        suggested_bump = version_manager.detect_bump_type(commits)
        next_version = current.bump(suggested_bump)

        # Create version table
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("", style="dim")
        table.add_column("Version", style="green")
        table.add_column("Details")

        table.add_row(
            "Current",
            str(current),
            "Latest git tag"
        )

        if commits:
            table.add_row(
                "Suggested",
                str(next_version),
                f"{suggested_bump.value} bump ({len(commits)} commits)"
            )
        else:
            table.add_row(
                "Status",
                "—",
                "[dim]No commits since last version[/dim]"
            )

        self.console.print(table)
        self.console.print()

        # Show commit breakdown if requested
        if show_commits and commits:
            self.console.print("[cyan]Commits since last version:[/cyan]")
            self.console.print()

            groups = version_manager.group_commits_by_type(commits)

            # Count by type
            type_counts = {}
            for commit_type, commit_list in groups.items():
                count = len(commit_list)
                if count > 0:
                    type_counts[commit_type] = count

            # Display counts
            for commit_type, count in type_counts.items():
                emoji = self._get_type_emoji(commit_type)
                self.console.print(f"  {emoji} [cyan]{commit_type}[/cyan]: {count}")

            self.console.print()

        # Show next steps
        if commits:
            self.console.print("[dim]Next steps:[/dim]")
            self.console.print(f"  [cyan]gai release[/cyan]                # Interactive release")
            self.console.print(f"  [cyan]gai release --auto[/cyan]         # Automatic release")
            self.console.print()

    def _get_type_emoji(self, commit_type: str) -> str:
        """Get prefix for commit type"""
        prefixes = {
            'breaking': '[!]',
            'feat': '[+]',
            'fix': '[*]',
            'docs': '[d]',
            'style': '[s]',
            'refactor': '[r]',
            'perf': '[p]',
            'test': '[t]',
            'build': '[b]',
            'ci': '[c]',
            'chore': '[~]',
            'other': '[-]'
        }
        return prefixes.get(commit_type, '[-]')
