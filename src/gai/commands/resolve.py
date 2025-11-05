"""
Resolve command - AI-assisted conflict resolution
"""

import re
from typing import List, Optional, Tuple
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
import click

from gai.core.config import Config
from gai.core.git import Git, GitError
from gai.inference import get_inference_engine, InferenceRequest, InferenceError


class ResolveCommand:
    """AI-assisted conflict resolution"""

    def __init__(self, config: Config, git: Git, verbose: bool = False):
        self.config = config
        self.git = git
        self.verbose = verbose
        self.console = Console()
        self.engine = get_inference_engine(config=config, verbose=verbose)

    def run(self, auto: bool = False):
        """
        Main resolve flow.

        Args:
            auto: Automatically apply AI suggestions without confirmation
        """
        self.console.print()
        self.console.print("[cyan]🔍 Checking for conflicts...[/cyan]")
        self.console.print()

        # Get conflicted files
        conflicted_files = self._get_conflicted_files()

        if not conflicted_files:
            self.console.print("[green]✓ No conflicts detected[/green]")
            self.console.print()
            return

        self.console.print(f"[yellow]Found {len(conflicted_files)} file(s) with conflicts:[/yellow]")
        for f in conflicted_files:
            self.console.print(f"  • {f}")
        self.console.print()

        # Resolve each file
        resolved_count = 0
        for file_path in conflicted_files:
            if self._resolve_file(file_path, auto):
                resolved_count += 1

        # Summary
        self.console.print()
        if resolved_count == len(conflicted_files):
            self.console.print(f"[green]✓ All {resolved_count} file(s) resolved![/green]")
            self.console.print()
            self.console.print("[bold]Next steps:[/bold]")
            self.console.print("  1. Review changes: [cyan]git diff --staged[/cyan]")
            self.console.print("  2. Continue merge: [cyan]git merge --continue[/cyan]")
            self.console.print("     or rebase: [cyan]git rebase --continue[/cyan]")
        else:
            self.console.print(f"[yellow]Resolved {resolved_count}/{len(conflicted_files)} files[/yellow]")
            self.console.print()
            self.console.print("Some files still have conflicts. Resolve them manually or run [cyan]gai resolve[/cyan] again.")

        self.console.print()

    def _get_conflicted_files(self) -> List[str]:
        """Get list of files with merge conflicts"""
        try:
            result = self.git._run("diff", "--name-only", "--diff-filter=U")
            files = [f.strip() for f in result.stdout.strip().split('\n') if f.strip()]
            return files
        except GitError:
            return []

    def _resolve_file(self, file_path: str, auto: bool) -> bool:
        """
        Resolve conflicts in a single file.

        Returns: True if resolved, False if skipped/failed
        """
        self.console.print()
        self.console.print(f"[bold cyan]━━━ {file_path} ━━━[/bold cyan]")
        self.console.print()

        # Read file content
        try:
            repo_root = Path(self.git.repo_root())
            full_path = repo_root / file_path
            content = full_path.read_text()
        except Exception as e:
            self.console.print(f"[red]Error reading file: {e}[/red]")
            return False

        # Parse conflicts
        conflicts = self._parse_conflicts(content)

        if not conflicts:
            self.console.print("[yellow]No conflict markers found (already resolved?)[/yellow]")
            return False

        # Show conflicts
        self.console.print(f"Found {len(conflicts)} conflict(s) in this file")
        self.console.print()

        # Resolve each conflict
        resolved_content = content
        for idx, conflict in enumerate(conflicts, 1):
            self.console.print(f"[bold cyan]Conflict {idx}/{len(conflicts)}:[/bold cyan]")
            self.console.print()

            # Show both versions
            self.console.print("[bold yellow]LOCAL (your changes):[/bold yellow]")
            local_syntax = Syntax(
                conflict['ours'],
                self._get_file_language(file_path),
                theme="monokai",
                line_numbers=False
            )
            self.console.print(Panel(local_syntax, border_style="yellow"))
            self.console.print()

            self.console.print("[bold blue]REMOTE (their changes):[/bold blue]")
            remote_syntax = Syntax(
                conflict['theirs'],
                self._get_file_language(file_path),
                theme="monokai",
                line_numbers=False
            )
            self.console.print(Panel(remote_syntax, border_style="blue"))
            self.console.print()

            # Get AI analysis
            if not auto:
                self.console.print("[cyan]🤖 Getting AI analysis...[/cyan]")
                try:
                    analysis = self._analyze_conflict(file_path, conflict)
                    self.console.print()
                    self.console.print(Panel(
                        analysis,
                        title="AI Analysis",
                        border_style="cyan"
                    ))
                    self.console.print()
                except InferenceError:
                    pass  # Continue without AI

            # Ask user choice
            if auto:
                # Auto mode: prefer local by default
                choice = '1'
                self.console.print("[cyan]🤖 Auto-choosing: Keep local (--auto mode)[/cyan]")
            else:
                self.console.print("[bold]Choose resolution:[/bold]")
                self.console.print("  1. Keep LOCAL (your version)")
                self.console.print("  2. Keep REMOTE (their version)")
                self.console.print("  3. Keep BOTH (yours then theirs)")
                self.console.print("  4. Edit manually")
                self.console.print("  5. Skip this conflict")
                self.console.print()

                choice = click.prompt("Choice", type=click.Choice(['1', '2', '3', '4', '5']), default='1')

            # Apply choice
            if choice == '1':
                # Keep local
                resolution = conflict['ours']
            elif choice == '2':
                # Keep remote
                resolution = conflict['theirs']
            elif choice == '3':
                # Keep both
                resolution = conflict['ours'] + '\n' + conflict['theirs']
            elif choice == '4':
                # Edit manually
                self.console.print(f"[cyan]Opening in $EDITOR...[/cyan]")
                import os
                editor = os.getenv('EDITOR', 'vim')
                os.system(f"{editor} {full_path}")
                if click.confirm("Continue resolving other conflicts?", default=True):
                    continue
                else:
                    return False
            elif choice == '5':
                # Skip
                self.console.print("[dim]Skipped this conflict[/dim]")
                continue

            # Replace conflict markers with resolution
            conflict_pattern = self._build_conflict_pattern(conflict)
            resolved_content = resolved_content.replace(conflict_pattern, resolution, 1)

            self.console.print()

        # Write resolved content
        try:
            full_path.write_text(resolved_content)
            self.git._run("add", file_path)
            self.console.print(f"[green]✓ {file_path} resolved and staged[/green]")
            return True
        except Exception as e:
            self.console.print(f"[red]Error applying resolution: {e}[/red]")
            return False

    def _parse_conflicts(self, content: str) -> List[dict]:
        """
        Parse conflict markers in file content.

        Returns list of conflicts with:
            - ours: our version
            - theirs: their version
            - base: common ancestor (if available)
        """
        conflicts = []
        lines = content.split('\n')

        i = 0
        while i < len(lines):
            if lines[i].startswith('<<<<<<<'):
                # Start of conflict
                conflict = {
                    'ours': [],
                    'theirs': [],
                    'base': None
                }

                i += 1
                # Read "ours" section
                while i < len(lines) and not lines[i].startswith('======='):
                    conflict['ours'].append(lines[i])
                    i += 1

                i += 1  # Skip =======

                # Read "theirs" section
                while i < len(lines) and not lines[i].startswith('>>>>>>>'):
                    conflict['theirs'].append(lines[i])
                    i += 1

                conflict['ours'] = '\n'.join(conflict['ours'])
                conflict['theirs'] = '\n'.join(conflict['theirs'])

                conflicts.append(conflict)

            i += 1

        return conflicts

    def _build_conflict_pattern(self, conflict: dict) -> str:
        """Build the original conflict pattern to replace"""
        return f"<<<<<<< HEAD\n{conflict['ours']}\n=======\n{conflict['theirs']}\n>>>>>>>"

    def _analyze_conflict(self, file_path: str, conflict: dict) -> str:
        """Get AI analysis of a single conflict"""
        prompt = f"""Analyze this merge conflict in {file_path}:

LOCAL version (current):
```
{conflict['ours']}
```

REMOTE version (incoming):
```
{conflict['theirs']}
```

Provide a brief analysis (2-3 sentences):
1. What's the difference between the two versions?
2. Which version should probably be kept and why?
3. Any risks to consider?

Be concise and practical."""

        system_prompt = "You are a code review expert analyzing merge conflicts."

        request = InferenceRequest.from_prompt(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=300
        )
        response = self.engine.generate(request)
        analysis = response.text

        return analysis.strip()

    def _get_ai_resolution(self, file_path: str, content: str, conflicts: List[dict]) -> dict:
        """
        Get AI suggestion for resolving conflicts.

        Returns:
            - explanation: why AI chose this resolution
            - resolved_content: full file content with conflicts resolved
        """
        # Build prompt
        prompt = f"""You are resolving a git merge conflict in file: {file_path}

File content with conflicts:
```
{content}
```

There are {len(conflicts)} conflict(s) in this file.

Your task:
1. Analyze each conflict carefully
2. Choose the best resolution (keep ours, keep theirs, or combine both)
3. Provide the COMPLETE resolved file content
4. Explain your reasoning

Respond in this format:

EXPLANATION:
[Brief explanation of how you resolved each conflict and why]

RESOLVED_CONTENT:
[The complete file content with all conflicts resolved]
"""

        system_prompt = """You are an expert at resolving git merge conflicts.
You understand code semantics and can make intelligent decisions about how to combine changes.
Be conservative - when in doubt, try to preserve both changes if they can coexist."""

        # Get AI response
        request = InferenceRequest.from_prompt(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=4000
        )
        inference_response = self.engine.generate(request)
        response = inference_response.text

        # Parse response
        parts = response.split('RESOLVED_CONTENT:')

        if len(parts) != 2:
            raise InferenceError("AI response format invalid")

        explanation = parts[0].replace('EXPLANATION:', '').strip()
        resolved_content = parts[1].strip()

        # Remove markdown code blocks if present
        resolved_content = re.sub(r'^```\w*\n', '', resolved_content)
        resolved_content = re.sub(r'\n```$', '', resolved_content)

        return {
            'explanation': explanation,
            'resolved_content': resolved_content
        }

    def _get_file_language(self, file_path: str) -> str:
        """Get language for syntax highlighting"""
        ext = Path(file_path).suffix.lower()

        lang_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.rs': 'rust',
            '.go': 'go',
            '.java': 'java',
            '.rb': 'ruby',
            '.php': 'php',
            '.c': 'c',
            '.cpp': 'cpp',
            '.h': 'c',
            '.hpp': 'cpp',
            '.sh': 'bash',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.json': 'json',
            '.md': 'markdown',
            '.html': 'html',
            '.css': 'css',
        }

        return lang_map.get(ext, 'text')
