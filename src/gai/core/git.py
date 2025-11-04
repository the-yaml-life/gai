"""
Git operations wrapper
"""

import subprocess
from typing import Optional, List, Tuple
from pathlib import Path


class GitError(Exception):
    """Git operation error"""
    pass


class Git:
    """Git operations wrapper"""

    def __init__(self, repo_path: Optional[Path] = None):
        self.repo_path = repo_path or Path.cwd()

    def _run(self, *args, check: bool = True) -> subprocess.CompletedProcess:
        """Run git command"""
        cmd = ["git", "-C", str(self.repo_path)] + list(args)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=check
            )
            return result
        except subprocess.CalledProcessError as e:
            raise GitError(f"Git command failed: {e.stderr}") from e

    def is_repo(self) -> bool:
        """Check if current dir is a git repo"""
        result = self._run("rev-parse", "--git-dir", check=False)
        return result.returncode == 0

    def get_status(self) -> str:
        """Get git status --porcelain"""
        result = self._run("status", "--porcelain")
        return result.stdout

    def get_diff(self, cached: bool = True, target: Optional[str] = None) -> str:
        """Get git diff"""
        args = ["diff"]

        if cached:
            args.append("--cached")

        if target:
            args.append(target)

        result = self._run(*args)
        return result.stdout

    def get_diff_stat(self, cached: bool = True, target: Optional[str] = None) -> str:
        """Get git diff --stat"""
        args = ["diff", "--stat"]

        if cached:
            args.append("--cached")

        if target:
            args.append(target)

        result = self._run(*args)
        return result.stdout

    def add_all(self):
        """Run git add -A"""
        self._run("add", "-A")

    def commit(self, message: str):
        """Create commit with message"""
        self._run("commit", "-m", message)

    def get_current_branch(self) -> str:
        """Get current branch name"""
        try:
            result = self._run("rev-parse", "--abbrev-ref", "HEAD")
            return result.stdout.strip()
        except GitError:
            # For empty repos (no commits yet), try symbolic-ref
            try:
                result = self._run("symbolic-ref", "--short", "HEAD")
                return result.stdout.strip()
            except GitError:
                # Fallback to 'main' if nothing works
                return "main"

    def get_merge_base(self, branch1: str, branch2: str) -> str:
        """Get merge base between two branches"""
        result = self._run("merge-base", branch1, branch2)
        return result.stdout.strip()

    def get_commits_between(self, base: str, head: str) -> List[str]:
        """Get commit hashes between base and head"""
        result = self._run("rev-list", f"{base}..{head}")
        commits = result.stdout.strip().split('\n')
        return [c for c in commits if c]

    def get_commit_message(self, commit: str) -> str:
        """Get commit message"""
        result = self._run("log", "-1", "--pretty=%B", commit)
        return result.stdout.strip()

    def get_branch_diff(self, base: str, head: Optional[str] = None) -> str:
        """Get diff between branches"""
        if head is None:
            head = self.get_current_branch()

        merge_base = self.get_merge_base(base, head)
        return self.get_diff(cached=False, target=f"{merge_base}..{head}")

    def get_conflicts(self, branch: str) -> List[Tuple[str, str]]:
        """
        Check for potential merge conflicts.
        Returns list of (file, status) tuples.
        """
        # Dry run merge to check conflicts
        current = self.get_current_branch()

        # This is a simple check - real conflict detection needs merge simulation
        # For MVP, we'll return empty and improve later
        # TODO: Implement proper conflict detection
        return []

    def get_changed_files(self, base: str, head: Optional[str] = None) -> List[str]:
        """Get list of changed files between base and head"""
        if head is None:
            head = self.get_current_branch()

        merge_base = self.get_merge_base(base, head)
        result = self._run("diff", "--name-only", f"{merge_base}..{head}")

        files = result.stdout.strip().split('\n')
        return [f for f in files if f]

    def get_file_changes_stat(self, base: str, head: Optional[str] = None) -> str:
        """Get detailed stats of file changes"""
        if head is None:
            head = self.get_current_branch()

        merge_base = self.get_merge_base(base, head)
        result = self._run("diff", "--stat", f"{merge_base}..{head}")
        return result.stdout

    def repo_root(self) -> str:
        """Get repository root path"""
        result = self._run("rev-parse", "--show-toplevel")
        return result.stdout.strip()

    def get_last_commit_hash(self) -> str:
        """Get the hash of the last commit"""
        try:
            result = self._run("rev-parse", "HEAD")
            return result.stdout.strip()
        except GitError:
            # Empty repo, no commits yet
            return None

    def push(self, remote: str = "origin", branch: Optional[str] = None) -> bool:
        """
        Push commits to remote.

        Args:
            remote: Remote name (default: origin)
            branch: Branch name (default: current branch)

        Returns:
            True if push succeeded

        Raises:
            GitError: If push fails
        """
        if branch is None:
            branch = self.get_current_branch()

        result = self._run("push", remote, branch)
        return result.returncode == 0
