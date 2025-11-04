#!/usr/bin/env python3
"""
Simulate a real commit with DB backend to test end-to-end
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from gai.core.stats import get_stats
from gai.core.config import get_config

def simulate_commit():
    """Simulate a complete commit workflow"""
    print("Simulating real commit with DB backend...")
    print()

    # Load config (should have use_db: true)
    config = get_config()
    print(f"1. Config loaded: use_db = {config.get('stats.use_db', False)}")
    print()

    # Get stats instance (will auto-create DB if needed)
    stats = get_stats(config)
    print(f"2. Stats initialized with DB: {stats.use_db}")
    print()

    # Simulate command usage
    print("3. Recording command: commit")
    stats.record_command("commit")

    # Simulate API call (like groq_client would do)
    print("4. Recording API call...")
    stats.record_api_call(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        tokens_used=2500,
        prompt_tokens=2000,
        completion_tokens=500
    )

    # Simulate commit info
    print("5. Recording commit metadata...")
    stats.record_commit(auto=False, dry_run=False, edited=True)

    # Save complete record (like CommitCommand would do)
    print("6. Saving complete record to DB...")
    stats.save_record(
        command_type="commit",
        generated_output="feat(storage): add SQLite/PostgreSQL backend for stats tracking\n\nImplement dual-mode stats system supporting both JSON (legacy) and\ndatabase backends. Adds automatic schema creation and migration support.",
        repo_path="/home/namen/_home/the.yaml.life/tylting/gai",
        branch_name="feature/db-backend",
        commit_hash="a1b2c3d4e5f6",
        diff_content="diff --git a/core/db_storage.py b/core/db_storage.py\nnew file mode 100644\n...",
        status_content="A  core/db_storage.py\nM  core/stats.py\nM  .gai.yaml",
        stats_content="3 files changed, 522 insertions(+), 15 deletions(-)",
        commit_type="feat",
        commit_scope="storage",
        breaking_change=False,
        success=True,
        error_message=None
    )

    print("   ✓ Record saved!")
    print()

    # Get summary
    print("7. Getting stats summary...")
    summary = stats.get_summary()
    print(f"   Commands total: {summary['commands']['total']}")
    print(f"   API calls: {summary['api']['total_calls']}")
    print(f"   Tokens used: {summary['api']['total_tokens']}")
    print(f"   Commits: {summary['commits']['total']}")
    print()

    # Query the record
    print("8. Querying last commit from DB...")
    history = stats.storage.get_history(limit=1)
    if history:
        record = history[0]
        print(f"   Timestamp: {record['timestamp']}")
        print(f"   Command: {record['command_type']}")
        print(f"   Type: {record['commit_type']}({record['commit_scope']})")
        print(f"   Message: {record['generated_output'][:80]}...")
        print(f"   Model: {record['model_used']}")
        print(f"   Tokens: {record['total_tokens']}")
        print(f"   Committed: {bool(record['committed'])}")
    print()

    print("✅ End-to-end test complete!")
    print()
    print("You can query the DB with:")
    print("  sqlite3 ~/.gai.db 'SELECT * FROM gai_records ORDER BY timestamp DESC LIMIT 1;'")
    print()

if __name__ == "__main__":
    simulate_commit()
