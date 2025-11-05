"""
Semantic versioning management for gai

Handles:
- Version detection from git tags
- Semantic version bumping (major.minor.patch)
- Conventional commit analysis
- Branch name parsing
"""

import re
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class BumpType(Enum):
    """Version bump types"""
    MAJOR = "major"
    MINOR = "minor"
    PATCH = "patch"


@dataclass
class Version:
    """Semantic version"""
    major: int
    minor: int
    patch: int

    def __str__(self) -> str:
        return f"v{self.major}.{self.minor}.{self.patch}"

    def bump(self, bump_type: BumpType) -> 'Version':
        """Return new version with bump applied"""
        if bump_type == BumpType.MAJOR:
            return Version(self.major + 1, 0, 0)
        elif bump_type == BumpType.MINOR:
            return Version(self.major, self.minor + 1, 0)
        else:  # PATCH
            return Version(self.major, self.minor, self.patch + 1)

    @classmethod
    def from_string(cls, version_str: str) -> 'Version':
        """Parse version from string (v1.2.3 or 1.2.3)"""
        # Remove 'v' prefix if present
        version_str = version_str.lstrip('v')

        # Parse major.minor.patch
        match = re.match(r'^(\d+)\.(\d+)\.(\d+)', version_str)
        if not match:
            raise ValueError(f"Invalid version format: {version_str}")

        major, minor, patch = match.groups()
        return cls(int(major), int(minor), int(patch))


@dataclass
class CommitInfo:
    """Information about a commit"""
    hash: str
    message: str
    type: Optional[str] = None
    scope: Optional[str] = None
    breaking: bool = False
    issue: Optional[str] = None


class VersionManager:
    """Manages semantic versioning for git repository"""

    def __init__(self, git):
        """
        Initialize version manager.

        Args:
            git: Git instance from gai.core.git
        """
        self.git = git

    def get_current_version(self) -> Optional[Version]:
        """
        Get current version from latest git tag.

        Returns:
            Current Version or None if no version tags exist
        """
        try:
            # Get latest tag matching v*.*.* pattern
            result = self.git.run(['describe', '--tags', '--abbrev=0', '--match', 'v*.*.*'])

            if result.returncode == 0:
                tag = result.stdout.strip()
                return Version.from_string(tag)

            return None
        except Exception:
            return None

    def get_commits_since_tag(self, tag: Optional[str] = None) -> List[CommitInfo]:
        """
        Get commits since a specific tag (or all commits if no tag).

        Args:
            tag: Git tag to start from (default: latest version tag)

        Returns:
            List of CommitInfo objects
        """
        if tag is None:
            current_version = self.get_current_version()
            tag = str(current_version) if current_version else None

        # Get commit hashes
        if tag:
            # Commits since tag
            result = self.git.run(['log', f'{tag}..HEAD', '--format=%H|||%s'])
        else:
            # All commits
            result = self.git.run(['log', '--format=%H|||%s'])

        if result.returncode != 0:
            return []

        commits = []
        for line in result.stdout.strip().split('\n'):
            if not line:
                continue

            parts = line.split('|||', 1)
            if len(parts) != 2:
                continue

            commit_hash, message = parts
            commits.append(self._parse_commit(commit_hash, message))

        return commits

    def _parse_commit(self, commit_hash: str, message: str) -> CommitInfo:
        """
        Parse conventional commit message.

        Format: type(scope): subject
        Or: type(scope)!: subject (breaking)

        Args:
            commit_hash: Git commit hash
            message: Commit message

        Returns:
            CommitInfo with parsed data
        """
        # Pattern: type(scope)!: subject or type!: subject or type(scope): subject
        pattern = r'^(\w+)(?:\(([^)]+)\))?(!)?: (.+)$'
        match = re.match(pattern, message)

        commit_type = None
        scope = None
        breaking = False

        if match:
            commit_type = match.group(1)
            scope = match.group(2)
            breaking = match.group(3) == '!'
            subject = match.group(4)
        else:
            subject = message

        # Check for BREAKING CHANGE in message body
        if 'BREAKING CHANGE:' in message.upper():
            breaking = True

        # Extract issue reference
        issue = self._extract_issue(message)

        return CommitInfo(
            hash=commit_hash,
            message=subject if match else message,
            type=commit_type,
            scope=scope,
            breaking=breaking,
            issue=issue
        )

    def _extract_issue(self, message: str) -> Optional[str]:
        """
        Extract issue reference from commit message.

        Looks for patterns like:
        - JIRA-123
        - GAI-456
        - #123

        Args:
            message: Commit message

        Returns:
            Issue reference or None
        """
        # Pattern: WORD-NUMBER or #NUMBER
        patterns = [
            r'([A-Z]+-\d+)',  # JIRA-123
            r'#(\d+)'          # #123
        ]

        for pattern in patterns:
            match = re.search(pattern, message)
            if match:
                return match.group(1) if '-' in match.group(0) else f"#{match.group(1)}"

        return None

    def detect_bump_type(self, commits: Optional[List[CommitInfo]] = None) -> BumpType:
        """
        Detect recommended bump type from commits.

        Rules:
        - Any BREAKING CHANGE → MAJOR
        - Any feat → MINOR
        - Any fix → PATCH
        - Default → PATCH

        Args:
            commits: List of commits to analyze (default: commits since last tag)

        Returns:
            Recommended BumpType
        """
        if commits is None:
            commits = self.get_commits_since_tag()

        has_breaking = False
        has_feat = False
        has_fix = False

        for commit in commits:
            if commit.breaking:
                has_breaking = True
            if commit.type == 'feat':
                has_feat = True
            if commit.type == 'fix':
                has_fix = True

        if has_breaking:
            return BumpType.MAJOR
        elif has_feat:
            return BumpType.MINOR
        elif has_fix:
            return BumpType.PATCH
        else:
            # Default to patch for any changes
            return BumpType.PATCH

    def get_next_version(self, bump_type: Optional[BumpType] = None) -> Version:
        """
        Get next version based on bump type.

        Args:
            bump_type: Type of bump (auto-detect if None)

        Returns:
            Next Version
        """
        current = self.get_current_version()

        # If no version exists, start at 0.1.0
        if current is None:
            current = Version(0, 0, 0)

        # Auto-detect bump type if not specified
        if bump_type is None:
            bump_type = self.detect_bump_type()

        return current.bump(bump_type)

    def update_version_files(self, version: Version, repo_root: Path):
        """
        Update version in project files.

        Updates:
        - VERSION file
        - pyproject.toml (if exists)

        Args:
            version: New version to write
            repo_root: Repository root path
        """
        # Update VERSION file
        version_file = repo_root / "VERSION"
        with open(version_file, 'w') as f:
            f.write(f"{version}\n")

        # Update pyproject.toml if it exists
        pyproject_file = repo_root / "pyproject.toml"
        if pyproject_file.exists():
            content = pyproject_file.read_text()

            # Replace version = "x.y.z" with new version
            version_str = str(version).lstrip('v')
            new_content = re.sub(
                r'version\s*=\s*"[^"]+"',
                f'version = "{version_str}"',
                content
            )

            pyproject_file.write_text(new_content)

    def group_commits_by_type(self, commits: List[CommitInfo]) -> Dict[str, List[CommitInfo]]:
        """
        Group commits by type for changelog generation.

        Args:
            commits: List of commits

        Returns:
            Dict mapping commit type to list of commits
        """
        groups = {
            'breaking': [],
            'feat': [],
            'fix': [],
            'docs': [],
            'style': [],
            'refactor': [],
            'perf': [],
            'test': [],
            'build': [],
            'ci': [],
            'chore': [],
            'other': []
        }

        for commit in commits:
            if commit.breaking:
                groups['breaking'].append(commit)
            elif commit.type and commit.type in groups:
                groups[commit.type].append(commit)
            else:
                groups['other'].append(commit)

        # Remove empty groups
        return {k: v for k, v in groups.items() if v}

    def generate_changelog_entry(self, version: Version, commits: List[CommitInfo]) -> str:
        """
        Generate changelog entry for a version.

        Format follows Keep a Changelog style.

        Args:
            version: Version for this entry
            commits: Commits included in this version

        Returns:
            Changelog entry as markdown
        """
        from datetime import datetime

        lines = []
        lines.append(f"## [{version}] - {datetime.now().strftime('%Y-%m-%d')}")
        lines.append("")

        # Group commits
        groups = self.group_commits_by_type(commits)

        # Type to section name mapping
        section_names = {
            'breaking': 'BREAKING CHANGES',
            'feat': 'Features',
            'fix': 'Bug Fixes',
            'docs': 'Documentation',
            'style': 'Styling',
            'refactor': 'Refactoring',
            'perf': 'Performance',
            'test': 'Tests',
            'build': 'Build',
            'ci': 'CI',
            'chore': 'Chore',
            'other': 'Other'
        }

        # Write sections
        for commit_type, section_name in section_names.items():
            if commit_type not in groups:
                continue

            lines.append(f"### {section_name}")
            lines.append("")

            for commit in groups[commit_type]:
                # Format: - message (scope) [hash] (#issue)
                scope_str = f"({commit.scope})" if commit.scope else ""
                issue_str = f" ({commit.issue})" if commit.issue else ""
                hash_short = commit.hash[:7]

                lines.append(f"- {commit.message} {scope_str} [`{hash_short}`] {issue_str}")

            lines.append("")

        return "\n".join(lines)
