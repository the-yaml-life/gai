"""
Auto-generate commit messages command
"""

from gai.core.git import Git, GitError
from gai.core.config import Config
from gai.core.tokens import estimate_tokens, chunk_by_files
from gai.core.stats import get_stats
from gai.inference import get_inference_engine, InferenceRequest, InferenceError
from gai.ai.prompts import Prompts
from gai.ui.interactive import (
    show_commit_message,
    show_error,
    show_warning,
    show_info,
    show_success,
    commit_confirm,
    edit_text,
    confirm,
    ask_add_mode,
    select_files_to_add
)
from gai.utils.detection import Detection


class CommitCommand:
    """Auto-generate commit messages"""

    def __init__(self, config: Config, git: Git, verbose: bool = False):
        self.config = config
        self.git = git
        self.verbose = verbose

        # Initialize inference engine
        self.engine = get_inference_engine(config=config, verbose=verbose)

    def _get_model_token_limit(self, model: str) -> int:
        """
        Get TPM (tokens per minute) limit for a model.

        Args:
            model: Model name (with or without backend prefix)

        Returns:
            Token limit per minute for the model
        """
        # Import here to avoid circular dependency
        from gai.ai.rate_limiter import RateLimiter

        # Strip backend prefixes
        clean_model = model.replace("groq/", "").replace("ollama/", "").replace("anannas/", "")

        # Get limit from RateLimiter's MODEL_LIMITS
        return RateLimiter.MODEL_LIMITS.get(clean_model, 30_000)

    def run(
        self,
        auto: bool = False,
        amend: bool = False,
        context: str = None,
        breaking: bool = False,
        dry_run: bool = False,
        no_add: bool = False
    ):
        """
        Run commit command.

        Args:
            auto: Auto-commit without confirmation
            amend: Amend last commit
            context: Additional context
            breaking: Mark as breaking change
            dry_run: Show message without committing
            no_add: Skip git add, use only staged changes
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

            # Handle adding files (interactive mode if not --auto and not --no-add)
            if not no_add:
                # Check if there are unstaged files
                unstaged_files = self.git.get_unstaged_files()
                has_staged = self.git.has_staged_files()

                # Interactive mode: ask how to add files if not in auto mode
                if not auto and unstaged_files:
                    add_mode = ask_add_mode()

                    if add_mode is None:
                        show_info("Commit cancelled")
                        return
                    elif add_mode == "all":
                        if self.verbose:
                            show_info("Running git add -A...")
                        self.git.add_all()
                        # Show files that will be committed
                        show_info("\nFiles selected:")
                        for f in unstaged_files:
                            show_info(f"  - {f}")
                        show_info("")
                    elif add_mode == "select":
                        selected_files = select_files_to_add(unstaged_files)
                        if selected_files is None:
                            show_info("Commit cancelled")
                            return
                        elif selected_files:
                            show_info(f"\nFiles selected ({len(selected_files)}):")
                            for f in selected_files:
                                show_info(f"  - {f}")
                            show_info("")
                            self.git.add_files(selected_files)
                        # If empty list, use only staged files (do nothing)
                    elif add_mode == "staged":
                        # Use only staged files, do nothing
                        if self.verbose:
                            show_info("Using only staged files...")
                else:
                    # Auto mode or no unstaged files: add all
                    if unstaged_files:
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
                    show_info("Generating commit message...")

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
                        should_commit, final_message, should_push = commit_confirm(message, allow_edit=True)
                        if should_commit and final_message:
                            # Save to DB before committing
                            scope = Detection.detect_scope([line.split()[1] for line in status.split('\n') if line.strip()])
                            commit_type = "feat"  # Default for large commits
                            get_stats().record_commit(commit_type, scope)

                            self.git.commit(final_message)
                            show_success("Commit created!")

                            # Push if requested
                            if should_push:
                                self._do_push()
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

                    # Validate sampled diff fits in model's token limit
                    sampled_tokens = estimate_tokens(diff)
                    primary_model = self.config.get('ai.model')
                    model_limit = self._get_model_token_limit(primary_model)

                    if self.verbose:
                        show_info(f"Sampled diff: {sampled_tokens:,} tokens (model limit: {model_limit:,})")

                    # Check if sampled diff still exceeds model limit
                    if sampled_tokens > model_limit:
                        show_warning(f"Even with sampling, diff is {sampled_tokens:,} tokens (model limit: {model_limit:,})")

                        # Try automatic fallback to parallel processing if configured
                        if len(parallel_models) >= 2:
                            show_info(f"Automatically using {len(parallel_models)} models in parallel...")
                            show_info("Generating commit message...")

                            # Use parallel processing
                            message = self._generate_parallel_commit(
                                diff=diff,
                                diff_stat=diff_stat,
                                status=status,
                                max_tokens=model_limit,
                                parallel_models=parallel_models
                            )

                            # Interactive confirmation
                            if not dry_run:
                                should_commit, final_message, should_push = commit_confirm(message, allow_edit=True)
                                if should_commit and final_message:
                                    scope = Detection.detect_scope([line.split()[1] for line in status.split('\n') if line.strip()])
                                    commit_type = "feat"
                                    get_stats().record_commit(commit_type, scope)

                                    self.git.commit(final_message)
                                    show_success("Commit created!")

                                    if should_push:
                                        self._do_push()
                                else:
                                    show_info("Commit aborted")
                            else:
                                show_commit_message(message)

                            return
                        else:
                            # No parallel models configured - fail with clear message
                            show_error(f"Diff too large even with sampling ({sampled_tokens:,} tokens > {model_limit:,} limit)")
                            show_info("Solutions:")
                            show_info("  1. Make smaller commits")
                            show_info("  2. Configure ai.parallel_models in .gai.yaml")
                            show_info("  3. Use a model with higher token limit")
                            return

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

                # Auto push when using --auto
                self._do_push()
                return

            # Interactive confirmation
            should_commit, final_message, should_push = commit_confirm(message, allow_edit=True)

            if should_commit and final_message:
                edited = final_message != message
                self.git.commit(final_message)
                get_stats().record_commit(edited=edited)
                # Save to DB if enabled
                commit_hash = self.git.get_last_commit_hash()
                self._save_to_db(status, diff, diff_stat, final_message, commit_type, scope, breaking, committed=True, commit_hash=commit_hash, success=True)
                show_success("Committed successfully")

                # Push if requested
                if should_push:
                    self._do_push()

                # Suggest release if appropriate
                self._suggest_release_if_needed()
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
        except InferenceError as e:
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

                # Suggest release if appropriate
                self._suggest_release_if_needed()
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
            request = InferenceRequest.from_prompt(
                prompt=user_prompt,
                system_prompt=system_prompt,
                max_tokens=50000
            )
            response = self.engine.generate(request)
            message = response.text

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
            summaries = self.engine.generate_parallel(
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

        request = InferenceRequest.from_prompt(
            prompt=final_prompt,
            system_prompt="You are an expert at writing clear, informative git commit messages.",
            max_tokens=50000
        )
        response = self.engine.generate(request)
        final_message = response.text

        return final_message

    def _do_push(self):
        """Push the commit to remote"""
        try:
            show_info("Pushing to remote...")
            branch = self.git.get_current_branch()
            result = self.git.push()

            if result:
                show_success(f"Pushed to origin/{branch}")
            else:
                show_warning("Push completed with warnings")

        except GitError as e:
            show_error(f"Push failed: {e}")
            show_info("You can push manually with: git push")

    def _suggest_release_if_needed(self):
        """
        Suggest creating a release if there are enough unreleased commits.

        Smart logic:
        - Only suggest if version detection is enabled in config
        - Only suggest if there are 3+ commits OR a breaking change
        - Only suggest if last tag was more than 1 day ago (avoid spam)
        """
        # Check if version suggestions are enabled
        if not self.config.get('commit.suggest_release', True):
            return

        try:
            from gai.core.versioning import VersionManager

            version_manager = VersionManager(self.git)

            # Get current version
            current_version = version_manager.get_current_version()

            # Get commits since last version
            commits = version_manager.get_commits_since_tag()

            if not commits:
                return

            # Smart threshold: only suggest if meaningful
            has_breaking = any(c.breaking for c in commits)
            has_feat = any(c.type == 'feat' for c in commits)
            commit_count = len(commits)

            # Suggest if:
            # - Breaking change (always)
            # - 5+ commits (significant batch)
            # - 3+ commits with at least one feature
            should_suggest = (
                has_breaking or
                commit_count >= 5 or
                (commit_count >= 3 and has_feat)
            )

            if not should_suggest:
                return

            # Detect suggested bump
            suggested_bump = version_manager.detect_bump_type(commits)
            next_version = version_manager.get_next_version(suggested_bump)

            # Show suggestion
            show_info("")
            show_info(f"Suggestion: {commit_count} unreleased commit{'s' if commit_count > 1 else ''}")

            if current_version:
                show_info(f"  Current version: {current_version}")
            show_info(f"  Suggested release: {next_version} ({suggested_bump.value} bump)")

            if has_breaking:
                show_warning("  Contains BREAKING CHANGES")

            show_info("")
            show_info("Create release:")
            show_info("  gai release          # Interactive")
            show_info("  gai release --auto   # Automatic")
            show_info("  gai version          # View details")

        except Exception:
            # Silently fail - this is just a suggestion
            pass
