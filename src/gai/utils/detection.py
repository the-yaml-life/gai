"""
Utilities for detecting commit type, scope, and issues
"""

import re
from pathlib import Path
from typing import Optional, List, Tuple


class Detection:
    """Detect scope, type, and issue references"""

    # Conventional commit types with patterns
    TYPE_PATTERNS = {
        "feat": [
            r"\+.*class\s+\w+",  # New class
            r"\+.*def\s+\w+",    # New function
            r"\+.*fn\s+\w+",     # New Rust function
            r"\+.*interface\s+\w+",  # New interface
        ],
        "fix": [
            r"[-+].*bug",
            r"[-+].*fix",
            r"[-+].*error",
            r"[-+].*issue",
        ],
        "docs": [],  # Detected by file extensions
        "style": [
            r"[-+].*format",
            r"[-+].*whitespace",
        ],
        "refactor": [
            r"[-+].*refactor",
            r"[-+].*rename",
            r"[-+].*move",
        ],
        "test": [],  # Detected by file extensions
        "chore": [],  # Default fallback
    }

    # File patterns for type detection
    FILE_PATTERNS = {
        "docs": [".md", ".txt", ".rst", "README", "CHANGELOG"],
        "test": ["test_", "_test.", "spec.", ".test.", ".spec."],
        "style": [".css", ".scss", ".sass", ".less"],
    }

    @staticmethod
    def detect_scope(changed_files: List[str]) -> Optional[str]:
        """
        Detect scope from changed files.

        Args:
            changed_files: List of changed file paths

        Returns:
            Detected scope or None
        """
        if not changed_files:
            return None

        # Extract directories
        dirs = set()
        for f in changed_files:
            parts = Path(f).parts
            if len(parts) > 1:
                # Use first meaningful directory (skip common roots)
                for part in parts[:-1]:
                    if part not in ["src", "lib", "app", "pkg"]:
                        dirs.add(part)
                        break

        # If only one directory, use it
        if len(dirs) == 1:
            return dirs.pop()

        # If 2-3 directories, combine
        if 2 <= len(dirs) <= 3:
            return ",".join(sorted(dirs))

        # Too many or none, return None
        return None

    @staticmethod
    def detect_type(diff: str, changed_files: List[str]) -> str:
        """
        Detect commit type from diff and files.

        Args:
            diff: Git diff content
            changed_files: List of changed files

        Returns:
            Detected type (defaults to 'chore')
        """
        # Check file-based patterns first
        for file_path in changed_files:
            # Documentation files
            if any(pattern in file_path for pattern in Detection.FILE_PATTERNS["docs"]):
                return "docs"

            # Test files
            if any(pattern in file_path for pattern in Detection.FILE_PATTERNS["test"]):
                return "test"

            # Style files
            if any(file_path.endswith(ext) for ext in Detection.FILE_PATTERNS["style"]):
                return "style"

        # Check diff patterns
        diff_lower = diff.lower()

        # Count pattern matches for each type
        scores = {}
        for type_name, patterns in Detection.TYPE_PATTERNS.items():
            score = 0
            for pattern in patterns:
                score += len(re.findall(pattern, diff_lower, re.MULTILINE))
            if score > 0:
                scores[type_name] = score

        # Return type with highest score
        if scores:
            return max(scores.items(), key=lambda x: x[1])[0]

        # Default to chore
        return "chore"

    @staticmethod
    def detect_issue(branch_name: str) -> Optional[str]:
        """
        Detect issue/ticket reference from branch name.

        Patterns supported:
        - feature/JIRA-123-description
        - fix/AUTH-456-bug-name
        - PROJ-789-feature

        Args:
            branch_name: Git branch name

        Returns:
            Issue reference or None
        """
        # Common patterns: JIRA-123, ABC-456, PROJ-789
        patterns = [
            r'([A-Z]+-\d+)',  # JIRA-123
            r'([A-Z]{2,}-\d+)',  # ABC-123
            r'#(\d+)',  # #123 (GitHub style)
        ]

        for pattern in patterns:
            match = re.search(pattern, branch_name)
            if match:
                return match.group(1)

        return None

    @staticmethod
    def detect_breaking_changes(diff: str) -> bool:
        """
        Detect potential breaking changes in diff.

        Args:
            diff: Git diff content

        Returns:
            True if breaking changes detected
        """
        breaking_patterns = [
            r'-\s*(public|export|def|fn|function)\s+\w+',  # Removed public API
            r'BREAKING[:\s]',  # Explicit marker
            r'breaking[:\s]change',  # Explicit marker
        ]

        diff_content = diff.lower()
        for pattern in breaking_patterns:
            if re.search(pattern, diff_content, re.MULTILINE):
                return True

        return False
