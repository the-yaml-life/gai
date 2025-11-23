# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [v0.3.1] - 2025-11-23

### Bug Fixes

- only stage pyproject.toml if it exists (release) [`6558bd0`] 


## [v0.3.0] - 2025-11-23

### Features

- improve error handling and logging docs(inference/adapters): add skip retry flag to error handling docs(inference/engine): add max_tokens parameter to request creation perf(inference/engine): reduce max_tokens value to avoid payload errors docs(inference/models): reduce max_tokens value to avoid payload errors docs(inference/adapters/anannas): add skip retry flag to error handling docs(inference/adapters/groq): add skip retry flag to error handling docs(inference/adapters/openrouter): add skip retry flag to error handling refactor(commands/release): improve error handling and logging refactor(commands/version): improve error handling and logging refactor(inference/engine): improve error handling and logging refactor(inference/models): reduce max_tokens value to avoid payload errors (commands) [`fd93ba7`] 


## [v0.2.0] - 2025-11-23

### Features

- bump version to 0.2.0 (version) [`2c0ba12`] 
- add project configuration to .serena/project.yml docs(project): update .serena/project.yml documentation feat(commands/release): add project selection to release command feat(commands/version): add project selection to version command feat(versioning): add support for monorepo projects feat(inference/adapters/anannas): add support for anannas API feat(inference/adapters/groq): add support for groq API feat(inference/adapters/openrouter): add support for openrouter API feat(inference/engine): add support for parallel task execution docs(inference/adapters/anannas): add anannas API documentation docs(inference/adapters/groq): add groq API documentation docs(inference/adapters/openrouter): add openrouter API documentation docs(inference/engine): add inference engine documentation docs(inference/exceptions): add inference exceptions documentation refactor(commands/release): refactor release command to use version manager refactor(commands/version): refactor version command to use version manager refactor(versioning): refactor versioning to use version manager refactor(inference/adapters/anannas): refactor anannas adapter to use requests refactor(inference/adapters/groq): refactor groq adapter to use requests refactor(inference/adapters/openrouter): refactor openrouter adapter to use requests refactor(inference/engine): refactor inference engine to use parallel task execution test(commands/release): add test for release command test(commands/version): add test for version command test(versioning): add test for versioning test(inference/adapters/anannas): add test for anannas adapter test(inference/adapters/groq): add test for groq adapter test(inference/adapters/openrouter): add test for openrouter adapter test(inference/engine): add test for inference engine test(inference/exceptions): add test for inference exceptions perf(commands/release): improve performance of release command perf(commands/version): improve performance of version command perf(versioning): improve performance of versioning perf(inference/adapters/anannas): improve performance of anannas adapter perf(inference/adapters/groq): improve performance of groq adapter perf(inference/adapters/openrouter): improve performance of openrouter adapter perf(inference/engine): improve performance of inference engine perf(inference/exceptions): improve performance of inference exceptions chore(project): update .serena/project.yml to use new configuration chore(commands/release): update release command to use new version manager chore(commands/version): update version command to use new version manager chore(versioning): update versioning to use new version manager chore(inference/adapters/anannas): update anannas adapter to use new requests chore(inference/adapters/groq): update groq adapter to use new requests chore(inference/adapters/openrouter): update openrouter adapter to use new requests chore(inference/engine): update inference engine to use new parallel task execution chore(inference/exceptions): update inference exceptions to use new error handling chore(commands/release): update release command to use new version manager chore(commands/version): update version command to use new version manager chore(versioning): update versioning to use new version manager chore(inference/adapters/anannas): update anannas adapter to use new requests chore(inference/adapters/groq): update groq adapter to use new requests chore(inference/adapters/openrouter): update openrouter adapter to use new requests chore(inference/engine): update inference engine to use new parallel task execution chore(inference/exceptions): update inference exceptions to use new error handling (project) [`32e4edc`] 
- get unstaged files including deleted and untracked (git) [`faf9506`] 
- improve ai commit message cleaning (commit) [`549b47c`] 
- add interactive commit modes and flags (readme) [`100fd7e`] 
- simplify file addition logic in interactive mode (commands/commit) [`55639ed`] 
- add --no-add option to skip git add (commit) [`d3cee95`] 
- show files that will be committed and selected files in commit message (commands/commit) [`e35a9d8`] 
- simplify inference engine params (gai/inference) [`6d53dc0`] 
- add suggestion to create release after commits (commands/commit) [`a9fd2d9`] 

### Bug Fixes

- improve git status parsing for porcelain format (core/git.py) [`8cdcd7b`] 
- add interactive options for file addition (ui) [`527dc04`] 

### Documentation

- update readme with new features and usage  [`76aee35`] 

### Other

- Here is a commit message that follows the conventional commits format:  [`8122feb`] 


## [v0.1.0] - 2025-11-04

### Features

- Implement release command with bump type detection and version file updates (commands/release) [`1f25216`] 
- parallel generation of summaries and analysis (inference/engine) [`9250a7d`] 
- add inference packages to package-data (pyproject.toml) [`252b38c`] 
- Update API configuration and add billing error handling (gai) [`d4088f1`] 
- add ollama endpoint support (llm_factory) [`33d2d8a`] 
- ignore db files (db) [`9865abe`] 
- add parallel model support and free model filtering (ai) [`b130ad5`] 
- add number input to select option (ui) [`c0a65e1`] 
- add push after commit option (commit) [`60c857e`] 
- add init command and config cascading (config) [`c5631b0`] 
- introduce AI-powered Git assistant (gai) [`f9084cb`] 

### Other

- **Title:** feat(inference): add OpenRouter backend support  [`ace7caa`] 
- Here is a conventional commit message based on the provided change summaries:  [`92d8e97`] 

