"""
Prompt templates for AI generation
"""

from typing import Optional


class Prompts:
    """Prompt templates for different commands"""

    @staticmethod
    def commit_message(
        status: str,
        diff_stat: str,
        diff: str,
        context: Optional[str] = None,
        breaking: bool = False,
        conventional: bool = True
    ) -> tuple[str, str]:
        """
        Generate prompt for commit message.

        Returns:
            (system_prompt, user_prompt)
        """
        system_prompt = """You are an expert at writing clear, concise git commit messages.
Follow these rules:
- Use conventional commits format: <type>(<scope>): <description>
- Types: feat, fix, docs, style, refactor, test, chore, perf
- Scope: optional, based on files changed
- Description: imperative mood, lowercase, no period
- Body: optional, explain what and why vs how
- Keep title under 72 characters
- Be specific and descriptive"""

        if not conventional:
            system_prompt = """You are an expert at writing clear, concise git commit messages.
- Use imperative mood (\"Add feature\" not \"Added feature\")
- Be specific and descriptive
- Keep first line under 72 characters
- Optionally add body for complex changes"""

        user_prompt = f"""Generate a commit message for these changes:

Status:
{status}

Stats:
{diff_stat}

Changes:
{diff}
"""

        if context:
            user_prompt += f"\nAdditional context: {context}\n"

        if breaking:
            user_prompt += "\nNOTE: This is a BREAKING CHANGE. Include BREAKING CHANGE footer.\n"

        user_prompt += "\nGenerate the commit message (title and optional body):"

        return system_prompt, user_prompt

    @staticmethod
    def review_merge(
        current_branch: str,
        base_branch: str,
        commits: list[str],
        diff_stat: str,
        diff: str,
        changed_files: list[str]
    ) -> tuple[str, str]:
        """
        Generate prompt for merge review.

        Returns:
            (system_prompt, user_prompt)
        """
        system_prompt = """You are an expert code reviewer analyzing a merge.
Provide:
1. High-level summary of changes
2. Potential conflicts or issues
3. Breaking changes detected
4. Recommendations before merging

Be concise but thorough."""

        commits_text = "\n".join(f"- {msg}" for msg in commits[:10])
        if len(commits) > 10:
            commits_text += f"\n... and {len(commits) - 10} more commits"

        files_text = "\n".join(f"- {f}" for f in changed_files[:20])
        if len(changed_files) > 20:
            files_text += f"\n... and {len(changed_files) - 20} more files"

        user_prompt = f"""Analyze this merge:

Merging: {current_branch} → {base_branch}

Commits ({len(commits)}):
{commits_text}

Files changed ({len(changed_files)}):
{files_text}

Stats:
{diff_stat}

Changes:
{diff}

Provide merge analysis:"""

        return system_prompt, user_prompt

    @staticmethod
    def diff_analysis(
        branch1: str,
        branch2: str,
        diff_stat: str,
        diff: str,
        changed_files: list[str],
        style: str = "detailed"
    ) -> tuple[str, str]:
        """
        Generate prompt for diff analysis.

        Returns:
            (system_prompt, user_prompt)
        """
        if style == "concise":
            system_prompt = "You are a code analyst. Provide a concise 2-3 sentence summary of changes."
        elif style == "minimal":
            system_prompt = "You are a code analyst. Provide a one-sentence summary of changes."
        else:  # detailed
            system_prompt = """You are a code analyst. Analyze the diff and provide:
1. What changed (high-level overview)
2. Why these changes might have been made
3. Potential impact
4. Notable patterns or concerns"""

        files_text = "\n".join(f"- {f}" for f in changed_files[:15])
        if len(changed_files) > 15:
            files_text += f"\n... and {len(changed_files) - 15} more files"

        user_prompt = f"""Analyze differences between {branch1} and {branch2}:

Files changed ({len(changed_files)}):
{files_text}

Stats:
{diff_stat}

Changes:
{diff}

Analysis:"""

        return system_prompt, user_prompt
