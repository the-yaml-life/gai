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

    def run(self, args: List[str], check: bool = True) -> subprocess.CompletedProcess:
        """
        Run git command (public interface).

        Args:
            args: List of git command arguments
            check: Raise exception on error

        Returns:
            CompletedProcess result
        """
        return self._run(*args, check=check)

    def repo_root(self) -> str:
        """Get repository root path"""
        result = self._run("rev-parse", "--show-toplevel")
        return result.stdout.strip()

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

    def add_files(self, files: List[str]):
        """Add specific files to staging"""
        if files:
            self._run("add", "--", *files)

    def get_unstaged_files(self) -> List[str]:
        """
        Get list of unstaged files (modified, deleted, or untracked).

        Returns files that have changes in working tree (not yet staged).

        Git status porcelain format: XY filename
        - X = staged status (index)
        - Y = unstaged status (working tree)

        Examples:
        - ' M file.txt' = modified, not staged
        - 'MM file.txt' = modified, staged, then modified again (includes both)
        - '?? file.txt' = untracked
        - 'M  file.txt' = modified and staged (not included, no working tree changes)
        """
        result = self._run("status", "--porcelain")
        files = []

        for line in result.stdout.split('\n'):
            # Strip only trailing whitespace, keep leading structure
            line = line.rstrip()
            if not line or len(line) < 3:
                continue

            # Git porcelain format is exactly: XY<space>filename
            # where X and Y are each one character
            # Example: ' M file.txt' or '?? file.txt'
            staged_status = line[0]    # X
            unstaged_status = line[1]  # Y
            # Filename starts at position 3 (after XY and the space)
            filename = line[3:] if len(line) > 3 else ""

            if not filename:
                continue

            # Include files with any unstaged changes (Y is not space)
            # This includes: modified, deleted, untracked
            if unstaged_status != ' ':
                files.append(filename)
            # Also explicitly include untracked files (status ??)
            elif staged_status == '?' and unstaged_status == '?':
                files.append(filename)

        return files

    def has_staged_files(self) -> bool:
        """Check if there are any staged files"""
        result = self._run("diff", "--cached", "--name-only")
        return bool(result.stdout.strip())

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

    def get_status_full(self) -> str:
        """Get full git status output (human readable)"""
        result = self._run("status")
        return result.stdout

    def get_divergence_info(self, remote: str = "origin") -> dict:
        """
        Get divergence information between local and remote branch.

        Returns dict with:
            - diverged: bool
            - ahead: int (commits ahead)
            - behind: int (commits behind)
            - ahead_commits: list of commit hashes
            - behind_commits: list of commit hashes
            - remote_exists: bool
        """
        branch = self.get_current_branch()
        remote_branch = f"{remote}/{branch}"

        # Check if remote branch exists
        result = self._run("rev-parse", "--verify", remote_branch, check=False)
        if result.returncode != 0:
            return {
                'diverged': False,
                'ahead': 0,
                'behind': 0,
                'ahead_commits': [],
                'behind_commits': [],
                'remote_exists': False
            }

        # Get commits ahead
        result_ahead = self._run("rev-list", f"{remote_branch}..{branch}")
        ahead_commits = [c for c in result_ahead.stdout.strip().split('\n') if c]

        # Get commits behind
        result_behind = self._run("rev-list", f"{branch}..{remote_branch}")
        behind_commits = [c for c in result_behind.stdout.strip().split('\n') if c]

        ahead = len(ahead_commits)
        behind = len(behind_commits)

        return {
            'diverged': ahead > 0 and behind > 0,
            'ahead': ahead,
            'behind': behind,
            'ahead_commits': ahead_commits,
            'behind_commits': behind_commits,
            'remote_exists': True
        }

    def get_commit_summary(self, commit_hash: str) -> str:
        """Get one-line commit summary"""
        result = self._run("log", "-1", "--oneline", commit_hash)
        return result.stdout.strip()

    def has_uncommitted_changes(self) -> bool:
        """Check if there are uncommitted changes"""
        status = self.get_status()
        return bool(status.strip())

    def pull_rebase(self, remote: str = "origin", branch: Optional[str] = None) -> bool:
        """Pull with rebase"""
        if branch is None:
            branch = self.get_current_branch()

        result = self._run("pull", "--rebase", remote, branch)
        return result.returncode == 0

    def pull_merge(self, remote: str = "origin", branch: Optional[str] = None) -> bool:
        """Pull with merge"""
        if branch is None:
            branch = self.get_current_branch()

        result = self._run("pull", "--no-rebase", remote, branch)
        return result.returncode == 0

    def reset_hard(self, target: str) -> bool:
        """Reset hard to target"""
        result = self._run("reset", "--hard", target)
        return result.returncode == 0

    def fetch(self, remote: str = "origin") -> bool:
        """Fetch from remote"""
        result = self._run("fetch", remote)
        return result.returncode == 0
