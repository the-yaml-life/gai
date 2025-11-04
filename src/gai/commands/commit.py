"""
Auto-generate commit messages command
"""

from gai.core.git import Git, GitError
from gai.core.config import Config
from gai.core.tokens import estimate_tokens, chunk_by_files
from gai.core.stats import get_stats
from gai.ai.groq_client import GroqClient, GroqError
from gai.ai.prompts import Prompts
from gai.ui.interactive import (
    show_commit_message,
    show_error,
    show_warning,
    show_info,
    show_success,
    commit_confirm,
    edit_text,
    confirm
)
from gai.utils.detection import Detection


class CommitCommand:
    """Auto-generate commit messages"""

    def __init__(self, config: Config, git: Git, verbose: bool = False):
        self.config = config
        self.git = git
        self.verbose = verbose

        # Initialize AI client
        self.client = GroqClient(
            api_key=config.api_key,
            model=config.get('ai.model'),
            fallback_models=config.get('ai.fallback_models', []),
            temperature=config.get('ai.temperature', 0.3),
            verbose=verbose,
            api_url=config.get('ai.api_url', 'https://api.groq.com/openai/v1/chat/completions')
        )

    def run(
        self,
        auto: bool = False,
        amend: bool = False,
        context: str = None,
        breaking: bool = False,
        dry_run: bool = False
    ):
        """
        Run commit command.

        Args:
            auto: Auto-commit without confirmation
            amend: Amend last commit
            context: Additional context
            breaking: Mark as breaking change
            dry_run: Show message without committing
        """
        # Record command usage
        get_stats().record_command("commit")

        try:
            # Get git status and diff
            if self.verbose:
                show_info("Gathering git information...")

            status = self.git.get_status()
            if not status.strip():
                show_warning("No changes to commit")
                return

            # Add all changes
            if self.verbose:
                show_info("Running git add -A...")
            self.git.add_all()

            # Get diff after staging
            diff = self.git.get_diff(cached=True)
            diff_stat = self.git.get_diff_stat(cached=True)

            if not diff.strip():
                show_warning("No staged changes to commit")
                return

            # Check size
            tokens = estimate_tokens(diff)
            max_tokens = self.config.get('commit.max_diff_tokens', 30000)

            if self.verbose:
                show_info(f"Estimated tokens: {tokens:,} / {max_tokens:,}")

            if tokens > max_tokens:
                show_warning(f"Diff too large ({tokens:,} tokens)")

                # Check if parallel models are configured
                parallel_models = self.config.get('ai.parallel_models', [])

                if len(parallel_models) >= 2:
                    # Multi-model parallel processing (automatic)
                    show_info(f"Processing with {len(parallel_models)} models in parallel...")

                    # Split diff into chunks and process in parallel
                    message = self._generate_parallel_commit(
                        diff=diff,
                        diff_stat=diff_stat,
                        status=status,
                        max_tokens=max_tokens,
                        parallel_models=parallel_models
                    )

                    # Interactive confirmation (commit_confirm shows message internally)
                    if not dry_run:
                        should_commit, final_message = commit_confirm(message, allow_edit=True)
                        if should_commit and final_message:
                            # Save to DB before committing
                            scope = Detection.detect_scope([line.split()[1] for line in status.split('\n') if line.strip()])
                            commit_type = "feat"  # Default for large commits
                            get_stats().record_commit(commit_type, scope)

                            self.git.commit(final_message)
                            show_success("Commit created!")
                        else:
                            show_info("Commit aborted")
                    else:
                        # In dry-run, just show the message
                        show_commit_message(message)

                    return

                # Fallback to smart sampling
                if dry_run or confirm("Generate commit with intelligent sampling?", default=True):
                    show_info("Using intelligent sampling strategy...")

                    # Get changed files
                    changed_files_list = [
                        line.split()[1]
                        for line in status.split('\n')
                        if line.strip()
                    ]

                    # Build smart diff with prioritized content
                    diff = self._build_smart_diff(
                        full_diff=diff,
                        changed_files=changed_files_list,
                        diff_stat=diff_stat,
                        max_tokens=max_tokens
                    )
                else:
                    show_info("Aborted. Consider making smaller commits.")
                    return

            # Detect scope and type for DB storage
            changed_files = [
                line.split()[1]
                for line in status.split('\n')
                if line.strip()
            ]

            scope = None
            commit_type = "chore"

            if self.config.get('commit.scope_detection', True):
                scope = Detection.detect_scope(changed_files)

            if self.config.get('commit.type_detection', True):
                commit_type = Detection.detect_type(diff, changed_files)

            # Generate commit message
            if self.verbose:
                show_info("Generating commit message...")

            message = self._generate_message(
                status=status,
                diff_stat=diff_stat,
                diff=diff,
                context=context,
                breaking=breaking
            )

            if not message:
                show_error("Failed to generate commit message")
                return

            # Handle dry-run
            if dry_run:
                show_commit_message(message, title="Generated Commit Message (dry-run)")
                get_stats().record_commit(dry_run=True)
                # Save to DB if enabled
                self._save_to_db(status, diff, diff_stat, message, commit_type, scope, breaking, committed=False, success=True)
                return

            # Handle auto-commit
            if auto:
                show_commit_message(message)
                self.git.commit(message)
                get_stats().record_commit(auto=True)
                # Save to DB if enabled
                commit_hash = self.git.get_last_commit_hash()
                self._save_to_db(status, diff, diff_stat, message, commit_type, scope, breaking, committed=True, commit_hash=commit_hash, success=True)
                show_success("Committed successfully")
                return

            # Interactive confirmation
            should_commit, final_message = commit_confirm(message, allow_edit=True)

            if should_commit and final_message:
                edited = final_message != message
                self.git.commit(final_message)
                get_stats().record_commit(edited=edited)
                # Save to DB if enabled
                commit_hash = self.git.get_last_commit_hash()
                self._save_to_db(status, diff, diff_stat, final_message, commit_type, scope, breaking, committed=True, commit_hash=commit_hash, success=True)
                show_success("Committed successfully")
            else:
                show_info("Commit cancelled")
                # Save to DB if enabled (cancelled commit)
                self._save_to_db(status, diff, diff_stat, message, commit_type, scope, breaking, committed=False, success=True)

        except GitError as e:
            show_error(f"Git error: {e}")
            # Save error to DB if enabled
            if 'scope' in locals() and 'commit_type' in locals():
                self._save_to_db(
                    status=status if 'status' in locals() else "",
                    diff=diff if 'diff' in locals() else "",
                    diff_stat=diff_stat if 'diff_stat' in locals() else "",
                    message="",
                    commit_type=commit_type,
                    scope=scope,
                    breaking=breaking,
                    committed=False,
                    success=False,
                    error_message=str(e)
                )
        except GroqError as e:
            show_error(f"AI generation failed: {e}")
            show_info("Opening editor for manual commit message...")

            manual = edit_text("", "txt")
            if manual:
                self.git.commit(manual)
                commit_hash = self.git.get_last_commit_hash()
                # Save manual commit to DB if enabled
                if 'scope' in locals() and 'commit_type' in locals():
                    self._save_to_db(
                        status=status if 'status' in locals() else "",
                        diff=diff if 'diff' in locals() else "",
                        diff_stat=diff_stat if 'diff_stat' in locals() else "",
                        message=manual,
                        commit_type=commit_type,
                        scope=scope,
                        breaking=breaking,
                        committed=True,
                        commit_hash=commit_hash,
                        success=True
                    )
                show_success("Committed successfully (manual)")
        except Exception as e:
            show_error(f"Unexpected error: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            # Save error to DB if enabled
            if 'scope' in locals() and 'commit_type' in locals():
                self._save_to_db(
                    status=status if 'status' in locals() else "",
                    diff=diff if 'diff' in locals() else "",
                    diff_stat=diff_stat if 'diff_stat' in locals() else "",
                    message="",
                    commit_type=commit_type if 'commit_type' in locals() else "chore",
                    scope=scope if 'scope' in locals() else None,
                    breaking=breaking,
                    committed=False,
                    success=False,
                    error_message=str(e)
                )

    def _save_to_db(
        self,
        status: str,
        diff: str,
        diff_stat: str,
        message: str,
        commit_type: str,
        scope: str,
        breaking: bool,
        committed: bool,
        commit_hash: str = None,
        success: bool = True,
        error_message: str = None
    ):
        """Save commit record to database if DB backend is enabled"""
        stats = get_stats()

        # Only save if using DB backend
        if hasattr(stats, 'use_db') and stats.use_db:
            stats.save_record(
                command_type="commit",
                generated_output=message,
                repo_path=self.git.repo_root(),
                branch_name=self.git.get_current_branch(),
                commit_hash=commit_hash,
                diff_content=diff[:5000] if diff else None,  # Limit size
                status_content=status[:5000] if status else None,
                stats_content=diff_stat[:2000] if diff_stat else None,
                commit_type=commit_type,
                commit_scope=scope,
                breaking_change=breaking,
                success=success,
                error_message=error_message
            )

    def _generate_message(
        self,
        status: str,
        diff_stat: str,
        diff: str,
        context: str = None,
        breaking: bool = False
    ) -> str:
        """Generate commit message using AI"""

        # Detect scope and type
        changed_files = [
            line.split()[1]
            for line in status.split('\n')
            if line.strip()
        ]

        scope = None
        commit_type = "chore"

        if self.config.get('commit.scope_detection', True):
            scope = Detection.detect_scope(changed_files)

        if self.config.get('commit.type_detection', True):
            commit_type = Detection.detect_type(diff, changed_files)

        # Detect issue from branch
        issue = None
        if self.config.get('commit.issue_detection', True):
            branch = self.git.get_current_branch()
            issue = Detection.detect_issue(branch)

        # Auto-detect breaking changes
        if not breaking and self.config.get('commit.auto_detect_breaking', False):
            breaking = Detection.detect_breaking_changes(diff)

        # Build prompt
        conventional = self.config.get('commit.format') == 'conventional'

        system_prompt, user_prompt = Prompts.commit_message(
            status=status,
            diff_stat=diff_stat,
            diff=diff,
            context=context,
            breaking=breaking,
            conventional=conventional
        )

        # Generate
        try:
            message = self.client.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                max_tokens=500
            )

            # Add issue reference if detected
            if issue and issue not in message:
                message += f"\n\nRefs: {issue}"

            return message

        except Exception as e:
            if self.verbose:
                show_error(f"Generation failed: {e}")
            raise

    def _group_files_by_category(self, files: list) -> dict:
        """
        Group files by category for intelligent summarization.

        Categories:
        - core: Core application code (src/, lib/, app/)
        - commands: CLI commands
        - config: Configuration files
        - tests: Test files
        - docs: Documentation
        - build: Build/packaging files
        - other: Everything else
        """
        categories = {
            'core': [],
            'commands': [],
            'api': [],
            'ui': [],
            'tests': [],
            'config': [],
            'docs': [],
            'build': [],
            'other': []
        }

        for file in files:
            file_lower = file.lower()

            # Core application code
            if any(x in file_lower for x in ['core/', 'lib/', 'app/', 'src/']):
                categories['core'].append(file)
            # Commands
            elif 'command' in file_lower:
                categories['commands'].append(file)
            # API/Client code
            elif any(x in file_lower for x in ['api/', 'client', 'ai/']):
                categories['api'].append(file)
            # UI code
            elif 'ui/' in file_lower:
                categories['ui'].append(file)
            # Tests
            elif any(x in file_lower for x in ['test', 'spec', '__test__']):
                categories['tests'].append(file)
            # Config
            elif any(file_lower.endswith(x) for x in ['.yaml', '.yml', '.json', '.toml', '.ini', '.env']):
                categories['config'].append(file)
            # Docs
            elif any(file_lower.endswith(x) for x in ['.md', '.rst', '.txt']) or 'doc' in file_lower:
                categories['docs'].append(file)
            # Build/packaging
            elif any(x in file_lower for x in ['setup.py', 'pyproject.toml', 'package.json', 'cargo.toml', 'makefile', '.lock']):
                categories['build'].append(file)
            else:
                categories['other'].append(file)

        # Remove empty categories
        return {k: v for k, v in categories.items() if v}

    def _build_smart_diff(
        self,
        full_diff: str,
        changed_files: list,
        diff_stat: str,
        max_tokens: int
    ) -> str:
        """
        Build smart diff with prioritized content sampling.

        Priority levels:
        1. Critical files (README, LICENSE, package files) - include full diff
        2. Core code - include as much as budget allows
        3. Tests/other - stats only
        """
        # Split diff by file
        file_diffs = {}
        current_file = None
        current_lines = []

        for line in full_diff.split('\n'):
            if line.startswith('diff --git'):
                # Save previous file
                if current_file:
                    file_diffs[current_file] = '\n'.join(current_lines)

                # Extract filename: diff --git a/path/file b/path/file
                parts = line.split()
                if len(parts) >= 3:
                    current_file = parts[2][2:]  # Remove 'a/' prefix
                current_lines = [line]
            else:
                current_lines.append(line)

        # Save last file
        if current_file:
            file_diffs[current_file] = '\n'.join(current_lines)

        # Categorize files by priority
        high_priority = []
        medium_priority = []
        low_priority = []

        for file in changed_files:
            file_lower = file.lower()

            # High priority: documentation and package files
            if any(x in file_lower for x in ['readme', 'license', 'pyproject.toml', 'package.json', 'cargo.toml', 'setup.py']):
                high_priority.append(file)
            # Low priority: tests, generated files
            elif any(x in file_lower for x in ['test', 'spec', '__test__', '.lock', 'generated']):
                low_priority.append(file)
            # Medium priority: everything else
            else:
                medium_priority.append(file)

        # Build budget-aware diff
        budget = max_tokens * 3  # chars
        result = ["[SMART DIFF - Prioritized content sampling]\n"]
        remaining_budget = budget

        # Include high priority files fully
        if high_priority:
            result.append("\n## HIGH PRIORITY FILES (full diff)")
            for file in high_priority:
                if file in file_diffs:
                    file_diff = file_diffs[file]
                    if len(file_diff) < remaining_budget:
                        result.append(f"\n{file_diff}")
                        remaining_budget -= len(file_diff)

        # Include medium priority files with budget
        if medium_priority and remaining_budget > 1000:
            result.append("\n## CORE CODE (sampled)")
            for file in medium_priority:
                if file in file_diffs and remaining_budget > 1000:
                    file_diff = file_diffs[file]
                    # Take what fits
                    sample = file_diff[:min(len(file_diff), remaining_budget // 2)]
                    result.append(f"\n### {file}")
                    result.append(sample)
                    if len(file_diff) > len(sample):
                        result.append("\n... (truncated)")
                    remaining_budget -= len(sample)

        # Stats for low priority
        if low_priority:
            result.append(f"\n## TESTS/OTHER ({len(low_priority)} files)")
            for file in low_priority[:5]:
                result.append(f"  - {file}")
            if len(low_priority) > 5:
                result.append(f"  ... and {len(low_priority) - 5} more")

        result.append(f"\n## OVERALL STATISTICS\n{diff_stat}")

        return '\n'.join(result)

    def _build_grouped_summary(self, categories: dict, diff_stat: str) -> str:
        """
        Build an intelligent summary from grouped files.

        Instead of sending full diff, send structured summary by category.
        """
        lines = ["[INTELLIGENT GROUPING - Large commit summary]\n"]

        # Priority order for categories
        priority = ['core', 'api', 'commands', 'ui', 'config', 'tests', 'docs', 'build', 'other']

        for category in priority:
            if category not in categories:
                continue

            files = categories[category]
            lines.append(f"\n## {category.upper()} ({len(files)} files)")

            # Show first 5 files as examples
            for file in files[:5]:
                lines.append(f"  - {file}")

            if len(files) > 5:
                lines.append(f"  ... and {len(files) - 5} more {category} files")

        lines.append(f"\n## STATISTICS\n{diff_stat}")

        return '\n'.join(lines)

    def _generate_parallel_commit(
        self,
        diff: str,
        diff_stat: str,
        status: str,
        max_tokens: int,
        parallel_models: list
    ) -> str:
        """
        Generate commit message using parallel multi-model processing.

        Strategy:
        1. Apply smart sampling to reduce diff size
        2. Split reduced diff into chunks by file
        3. Send each chunk to a different model in parallel
        4. Each model summarizes its chunk
        5. Combine summaries into final commit message
        """
        from gai.core.tokens import chunk_by_files, estimate_tokens

        # Get changed files for smart sampling
        changed_files_list = [
            line.split()[1]
            for line in status.split('\n')
            if line.strip()
        ]

        # Apply smart sampling first to reduce diff size
        if self.verbose:
            show_info("Applying smart sampling to reduce diff size...")

        sampled_diff = self._build_smart_diff(
            full_diff=diff,
            changed_files=changed_files_list,
            diff_stat=diff_stat,
            max_tokens=max_tokens
        )

        sampled_tokens = estimate_tokens(sampled_diff)
        if self.verbose:
            show_info(f"Sampled diff: {sampled_tokens:,} tokens (from {estimate_tokens(diff):,})")

        # Now split sampled diff into chunks for parallel processing
        # Use smaller chunk size to avoid 413 errors (max 8k tokens per chunk)
        chunk_size = min(8000, max_tokens // 2)
        chunks = chunk_by_files(sampled_diff, chunk_size)

        if self.verbose:
            show_info(f"Split into {len(chunks)} chunks for parallel processing")

        # Limit to available models
        num_workers = min(len(chunks), len(parallel_models))
        chunks = chunks[:num_workers]
        models = parallel_models[:num_workers]

        # Create prompts for each chunk
        prompts = []
        for i, chunk in enumerate(chunks):
            prompt = f"""Analyze these code changes and provide a brief summary (2-3 sentences max):

CHUNK {i+1}/{len(chunks)}:
{chunk}

Focus on: what changed, why it matters, key functionality added/modified."""
            prompts.append((prompt, f"Chunk {i+1}/{len(chunks)}", 200))

        # Generate summaries in parallel
        if self.verbose:
            show_info("Generating summaries in parallel...")

        try:
            summaries = self.client.generate_parallel(
                prompts=prompts,
                models=models,
                system_prompt="You are analyzing code changes for a git commit message."
            )
        except Exception as e:
            show_error(f"Parallel generation failed: {e}")
            raise

        # Combine summaries into final commit message
        combined_summary = "\n\n".join([f"Part {i+1}: {s}" for i, s in enumerate(summaries)])

        # Generate final commit message from combined summaries
        final_prompt = f"""Based on these change summaries, generate a conventional commit message.

CHANGE SUMMARIES:
{combined_summary}

STATISTICS:
{diff_stat}

Format:
<type>(<scope>): <subject>

<body with key changes>

Rules:
- Use conventional commit types (feat, fix, refactor, docs, etc)
- Subject max 72 chars
- Body explains WHAT and WHY
- Be specific and concise"""

        if self.verbose:
            show_info("Generating final commit message...")

        final_message = self.client.generate(
            prompt=final_prompt,
            system_prompt="You are an expert at writing clear, informative git commit messages.",
            max_tokens=500
        )

        return final_message
