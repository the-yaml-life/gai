"""
Release command - Create semantic version releases
"""

from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import click

from gai.core.config import Config
from gai.core.git import Git, GitError
from gai.core.versioning import VersionManager, BumpType


class ReleaseCommand:
    """Create semantic version release"""

    def __init__(self, config: Config, git: Git, verbose: bool = False):
        self.config = config
        self.git = git
        self.verbose = verbose
        self.console = Console()
        self.version_manager = VersionManager(git)

    def run(
        self,
        bump_type: str = None,
        auto: bool = False,
        dry_run: bool = False,
        skip_changelog: bool = False
    ):
        """
        Run release command.

        Args:
            bump_type: Type of bump (major, minor, patch) - auto-detect if None
            auto: Automatically execute without confirmation
            dry_run: Show what would happen without executing
            skip_changelog: Skip CHANGELOG.md generation
        """
        self.console.print()
        self.console.print("[cyan]Creating Release[/cyan]")
        self.console.print()

        try:
            # Check for uncommitted changes
            status = self.git.get_status()
            if status.strip():
                self.console.print("[yellow]Warning: You have uncommitted changes[/yellow]")
                self.console.print()
                if not auto and not dry_run:
                    if not click.confirm("Continue anyway?", default=False):
                        self.console.print("[dim]Release cancelled[/dim]")
                        return

            # Get current version
            current = self.version_manager.get_current_version()

            # Get commits since last version
            commits = self.version_manager.get_commits_since_tag()

            if not commits and current is not None:
                self.console.print(Panel(
                    "[yellow]No commits since last version[/yellow]\n\n"
                    f"Current version: {current}\n"
                    "There are no new commits to release.",
                    title="Nothing to Release",
                    border_style="yellow"
                ))
                self.console.print()
                return

            # Determine bump type
            if bump_type:
                try:
                    selected_bump = BumpType(bump_type)
                except ValueError:
                    self.console.print(f"[red]Invalid bump type: {bump_type}[/red]")
                    self.console.print("[dim]Valid types: major, minor, patch[/dim]")
                    return
            else:
                selected_bump = self.version_manager.detect_bump_type(commits)

            # Calculate next version
            if current is None:
                # First version
                next_version = self.version_manager.get_next_version(BumpType.MINOR)
                self.console.print("[cyan]First release detected[/cyan]")
            else:
                next_version = current.bump(selected_bump)

            # Show version info
            version_table = Table(show_header=False, box=None)
            version_table.add_column("", style="dim")
            version_table.add_column("", style="bold")

            if current:
                version_table.add_row("Current version:", str(current))
            version_table.add_row("New version:", f"[green]{next_version}[/green]")
            version_table.add_row("Bump type:", selected_bump.value)
            version_table.add_row("Commits:", str(len(commits)))

            self.console.print(version_table)
            self.console.print()

            # Show commit summary
            if commits:
                groups = self.version_manager.group_commits_by_type(commits)
                self.console.print("[cyan]Changes:[/cyan]")
                for commit_type, commit_list in groups.items():
                    emoji = self._get_type_emoji(commit_type)
                    self.console.print(f"  {emoji} {commit_type}: {len(commit_list)}")
                self.console.print()

            # Dry run mode
            if dry_run:
                self.console.print(Panel(
                    "[yellow]DRY RUN[/yellow]\n\n"
                    "The following actions would be performed:\n"
                    f"1. Update VERSION file to {next_version}\n"
                    f"2. Update pyproject.toml version to {next_version}\n" +
                    ("3. Generate CHANGELOG.md entry\n" if not skip_changelog else "") +
                    f"4. Commit changes (chore: bump version to {next_version})\n"
                    f"5. Create git tag {next_version}\n"
                    f"6. Push tag to remote",
                    title="Dry Run",
                    border_style="yellow"
                ))
                self.console.print()
                return

            # Confirmation
            if not auto:
                self.console.print()
                if not click.confirm(f"Create release {next_version}?", default=True):
                    self.console.print("[dim]Release cancelled[/dim]")
                    return

            self.console.print()
            self.console.print("[cyan]Creating release...[/cyan]")
            self.console.print()

            # Get repo root
            repo_root = Path(self.git.repo_root())

            # 1. Update version files
            self.console.print("  [dim]1.[/dim] Updating version files...")
            self.version_manager.update_version_files(next_version, repo_root)

            # 2. Generate CHANGELOG
            if not skip_changelog:
                self.console.print("  [dim]2.[/dim] Generating CHANGELOG.md...")
                self._update_changelog(next_version, commits, repo_root)

            # 3. Stage changes
            self.console.print("  [dim]3.[/dim] Staging changes...")
            self.git.run(['add', 'VERSION'])
            self.git.run(['add', 'pyproject.toml'])
            if not skip_changelog:
                self.git.run(['add', 'CHANGELOG.md'])

            # 4. Commit
            commit_msg = f"chore: bump version to {next_version}\n\nGenerated with gai"
            self.console.print("  [dim]4.[/dim] Creating commit...")
            self.git.commit(commit_msg)

            # 5. Create tag
            self.console.print("  [dim]5.[/dim] Creating git tag...")
            tag_message = f"Release {next_version}\n\n{len(commits)} commits"
            self.git.run(['tag', '-a', str(next_version), '-m', tag_message])

            # 6. Push (if not auto, ask)
            should_push = auto
            if not auto:
                self.console.print()
                should_push = click.confirm("Push to remote?", default=True)

            if should_push:
                self.console.print("  [dim]6.[/dim] Pushing to remote...")
                try:
                    self.git.push()
                    self.git.run(['push', '--tags'])
                except GitError as e:
                    self.console.print(f"[yellow]Push failed: {e}[/yellow]")
                    self.console.print("[dim]You can push manually with: git push && git push --tags[/dim]")

            # Success!
            self.console.print()
            self.console.print(Panel(
                f"[green]Release {next_version} created successfully[/green]\n\n"
                f"Tag: {next_version}\n"
                f"Commits: {len(commits)}",
                title="Success",
                border_style="green"
            ))
            self.console.print()

        except GitError as e:
            self.console.print(f"[red]Git error: {e}[/red]")
            if self.verbose:
                import traceback
                traceback.print_exc()
        except Exception as e:
            self.console.print(f"[red]Unexpected error: {e}[/red]")
            if self.verbose:
                import traceback
                traceback.print_exc()

    def _update_changelog(self, version, commits, repo_root: Path):
        """
        Update CHANGELOG.md with new version entry.

        Args:
            version: New version
            commits: List of commits
            repo_root: Repository root path
        """
        changelog_file = repo_root / "CHANGELOG.md"

        # Generate new entry
        new_entry = self.version_manager.generate_changelog_entry(version, commits)

        # Read existing changelog or create new
        if changelog_file.exists():
            existing_content = changelog_file.read_text()

            # Find where to insert (after header, before first version)
            lines = existing_content.split('\n')
            insert_idx = 0

            # Skip header lines (# Changelog, blank lines, description)
            for i, line in enumerate(lines):
                if line.startswith('##'):
                    insert_idx = i
                    break
            else:
                # No existing versions, append at end
                insert_idx = len(lines)

            # Insert new entry
            lines.insert(insert_idx, new_entry)
            if insert_idx < len(lines) - 1:
                lines.insert(insert_idx + 1, '')

            new_content = '\n'.join(lines)
        else:
            # Create new changelog
            new_content = f"""# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

{new_entry}
"""

        changelog_file.write_text(new_content)

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
