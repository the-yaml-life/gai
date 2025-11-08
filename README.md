# gai

**TL;DR:** AI git assistant for commits, diffs, merges, releases, and conflict resolution.
```bash
git clone https://github.com/the-yaml-life/gai.git && cd gai && pip install -e . && gai init
```

## Commands

```
commit        Generate conventional commit messages
              --auto: add all, commit, and push automatically
              --no-add: use only staged files
              Interactive file selection with checkbox UI
diff          Analyze branch differences
review-merge  Preview merge with AI analysis
resolve       AI-assisted conflict resolution
fix           Auto-fix git problems (diverged, conflicts)
debug         Explain git issues without fixing
version       Show version and suggest next bump
release       Create semantic releases with changelog
models        List/filter available models
stats         Track API usage and tokens
config-show   Display current configuration
init          Initialize config in ~/.config/gai/
```

## Installation

```bash
# From source
git clone <repo-url>
cd gai
pip install -e .

# Initialize configuration
gai init

# Edit config and add API keys
vim ~/.config/gai/.env
vim ~/.config/gai/config.yaml
```

## Configuration

Config cascade: `~/.config/gai/config.yaml` → `.gai.yaml` (project-local)

### Backends

**Groq** (requires `GROQ_API_KEY`)
```yaml
ai:
  api_url: https://api.groq.com/openai/v1/chat/completions
  parallel_models:
    - llama-3.3-70b-versatile
    - groq/compound
```

**Ollama** (multiple endpoints)
```yaml
ollama:
  base_url: http://localhost:11434
  endpoints:
    local: http://localhost:11434
    remote: https://ollama.example.com

ai:
  parallel_models:
    - ollama.local/qwen2.5-coder:7b
    - ollama.remote/deepseek-r1:1.5b
```

**Anannas** (requires `ANANNAS_API_KEY`)
```yaml
anannas:
  api_url: https://api.anannas.ai/v1/chat/completions
```

**OpenRouter** (requires `OPENROUTER_API_KEY`)
```yaml
openrouter:
  api_url: https://openrouter.ai/api/v1/chat/completions

ai:
  parallel_models:
    - openrouter/google/gemini-2.0-flash-exp:free
    - openrouter/meta-llama/llama-3.3-8b-instruct:free
```

### Key Options

```yaml
commit:
  format: conventional           # conventional, free, minimal
  scope_detection: true          # Auto-detect scope from paths
  issue_detection: true          # Parse issue refs from branch names
  max_diff_tokens: 30000
  suggest_release: true          # Suggest release after commits

stats:
  use_db: true
  backend: sqlite                # sqlite or postgresql
  db_path: ~/.config/gai/gai.db

models:
  exclude_keywords:              # Filter out unwanted models
    - whisper
    - tts
```

## Usage

### Commits
```bash
# Interactive mode - ask which files to add
gai commit

# Auto mode - add all, commit, and push automatically
gai commit --auto

# Use only staged files (skip git add)
gai commit --no-add

# Preview without committing
gai commit --dry-run

# Add context to generation
gai commit --context "Fix bug"

# Amend last commit
gai commit --amend

# Combine flags
gai commit --auto --no-add      # Auto-commit staged files only + push
```

**File Selection Modes:**

When running `gai commit` interactively, you'll be prompted:
- **Add all files** - `git add -A` (all changes)
- **Select files** - Interactive checkbox to pick specific files
- **Use only staged** - Skip adding, use what's already staged
- **Cancel** - Abort commit

With `--auto`, files are always added automatically unless `--no-add` is used.

### Diffs and Merges
```bash
gai diff                        # Current branch vs HEAD
gai diff main                   # Compare with main branch
gai diff --stat                 # Show file statistics

gai review-merge main           # Analyze merge before merging
```

### Conflict Resolution
```bash
gai debug                       # Explain current git problem
gai fix --auto                  # Auto-fix diverged/conflicts
gai resolve                     # AI-assisted conflict resolution
```

### Releases
```bash
gai version                     # Show current version
gai version --commits           # Show commits since last release

gai release                     # Auto-detect bump type
gai release --major             # Force major bump
gai release --dry-run           # Preview release
```

### Models
```bash
gai models                      # List configured models
gai models --edit               # Interactive model selection
gai models --backend groq       # Filter by backend
gai models --free               # Show only free models
gai models --tier flagship      # Filter by tier
```

### Stats
```bash
gai stats                       # View usage statistics
gai stats --reset               # Reset statistics
```

## Multi-Model Parallel Processing

For large diffs exceeding token limits, gai distributes work across multiple models:

1. Prioritizes critical files (README, LICENSE, package files)
2. Chunks diff and sends to parallel models
3. Combines analyses into final result

Configure in `parallel_models` list. Use `gai models --edit` for interactive setup.

## Database Backend

```yaml
# SQLite (single user)
stats:
  use_db: true
  backend: sqlite
  db_path: ~/.config/gai/gai.db

# PostgreSQL (team/shared)
stats:
  use_db: true
  backend: postgresql
  connection_string: postgresql://user:pass@localhost/gai_stats
```

Query examples:
```bash
sqlite3 ~/.config/gai/gai.db "SELECT * FROM gai_records ORDER BY timestamp DESC LIMIT 10"
sqlite3 ~/.config/gai/gai.db "SELECT model_used, SUM(total_tokens) FROM gai_records GROUP BY model_used"
```

## Project Structure

```
src/gai/
├── inference/          # Inference engine (adapters + strategies)
│   ├── adapters/       # Groq, Ollama, Anannas, OpenRouter
│   └── strategies/     # Fallback, RateLimit, Retry
├── commands/           # CLI commands
├── core/              # Git, config, stats, tokens, versioning
├── ui/                # Interactive prompts
└── utils/             # Detection utilities
```

## Development

```bash
pytest                          # Run all tests
pytest -m unit                  # Unit tests only
pytest -m integration           # Integration tests only
pytest --cov=gai --cov-report=html
```

## License

GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later)
