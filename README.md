# gai - Git AI Assistant

AI-powered git assistant for commit messages, merge reviews, and intelligent diff analysis.

## Features

- **Smart Commit Messages**: Generate conventional commit messages from your changes
- **Multi-Model Parallel Processing**: Handle large diffs by distributing work across multiple models
- **Merge Preview**: AI-powered analysis of merge conflicts and changes
- **Intelligent Diff Analysis**: Get summaries of code changes between branches
- **Usage Stats**: Track AI usage with JSON or database backend (SQLite/PostgreSQL)
- **Smart Detection**: Automatic scope, type, and issue detection from files and branch names
- **OpenAI-Compatible**: Works with Groq, Ollama, OpenAI, or any compatible API

## Installation

### From Source

```bash
git clone https://github.com/yourusername/gai.git
cd gai
pip install -e .
```

### Using Container

```bash
podman build -t gai .
podman run -v /path/to/repo:/workspace -e GROQ_API_KEY=key gai commit --dry-run
```

## Quick Start

1. Get API key from [console.groq.com](https://console.groq.com)
2. Create `.env` file:
   ```bash
   GROQ_API_KEY=your_key_here
   ```
3. Generate commit:
   ```bash
   gai commit --dry-run  # Preview
   gai commit           # Create commit
   ```

## Usage

```bash
# Commit messages
gai commit                              # Interactive mode
gai commit --dry-run                    # Preview only
gai commit --auto                       # Auto-commit
gai commit --context "Fix login bug"   # Add context

# Merge preview
gai review-merge main                   # Analyze merge

# Diff analysis
gai diff                                # Current branch vs HEAD
gai diff main                           # Compare with main
gai diff --stat                         # File statistics

# Model management
gai models                              # List available models
gai models --select                     # Interactive selection

# Stats
gai stats                               # View usage stats
gai stats --reset                       # Reset stats
```

## Configuration

Edit `.gai.yaml`:

```yaml
ai:
  api_url: https://api.groq.com/openai/v1/chat/completions
  model: llama-3.3-70b-versatile
  max_tokens: 30000
  temperature: 0.3
  parallel_models:              # For large diffs
    - llama-3.3-70b-versatile
    - meta-llama/llama-4-scout-17b-16e-instruct
    - qwen/qwen3-32b

commit:
  format: conventional          # conventional, free, minimal
  scope_detection: true
  issue_detection: true
  max_diff_tokens: 25000

stats:
  use_db: false                 # true for database backend
  backend: sqlite               # sqlite or postgresql
  db_path: ~/.gai.db
```

## Multi-Model Parallel Processing

For large diffs that exceed token limits, gai can distribute work across multiple models:

1. Smart sampling prioritizes critical files (README, LICENSE, package files)
2. Chunks diff into pieces and sends to different models in parallel
3. Each model analyzes its chunk independently
4. Combines all analyses into final commit message

Configure parallel models in `.gai.yaml` or use `gai models --select` for interactive setup.

## Database Backend

Enable persistent storage for team collaboration:

```yaml
# SQLite (single user)
stats:
  use_db: true
  backend: sqlite
  db_path: ~/.gai.db

# PostgreSQL (team/shared)
stats:
  use_db: true
  backend: postgresql
  connection_string: postgresql://user:pass@localhost/gai_stats
```

Query examples:

```bash
# Recent commits
sqlite3 ~/.gai.db "SELECT * FROM gai_records ORDER BY timestamp DESC LIMIT 10"

# Token usage by model
sqlite3 ~/.gai.db "SELECT model_used, SUM(total_tokens) FROM gai_records GROUP BY model_used"
```

## Project Structure

```
src/gai/
├── ai/                 # AI client and rate limiting
├── commands/           # CLI commands (commit, review-merge, diff, models)
├── core/              # Git, config, stats, tokens
├── ui/                # Interactive UI components
└── utils/             # Detection utilities

tests/
├── unit/              # Unit tests
├── integration/       # Integration tests
└── conftest.py        # Pytest fixtures
```

## Smart Detection

- **Commit Type**: Detects feat, fix, refactor, docs, test, chore from changes
- **Scope**: Extracts scope from file paths (e.g., `src/auth/` → `auth`)
- **Issues**: Parses issue numbers from branch names (e.g., `feature/JIRA-123` → `JIRA-123`)

## Development

```bash
# Run tests
pytest

# With coverage
pytest --cov=gai --cov-report=html

# Test specific markers
pytest -m unit
pytest -m integration
```

## License

GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later)

See LICENSE file for details.
