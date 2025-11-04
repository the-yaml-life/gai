#!/usr/bin/env python3
"""
Test script for DB backend
Creates a test commit with DB backend enabled
"""

import os
import sys
from pathlib import Path

# Add gai to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gai.core.config import Config
from gai.core.stats import Stats
from gai.core.git import Git

def test_db_backend():
    """Test that DB backend works"""
    print("Testing DB backend for gai stats...")
    print()

    # Create stats instance with DB backend
    print("1. Creating Stats instance with SQLite backend...")
    stats = Stats(use_db=True, db_backend="sqlite", db_config={"db_path": "/tmp/gai_test.db"})
    print(f"   ✓ Stats initialized with DB backend")
    print()

    # Test record_command
    print("2. Recording command usage...")
    stats.record_command("commit")
    print("   ✓ Command recorded")
    print()

    # Test record_api_call
    print("3. Recording API call...")
    stats.record_api_call(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        tokens_used=1000,
        prompt_tokens=800,
        completion_tokens=200
    )
    print("   ✓ API call recorded")
    print()

    # Test record_commit
    print("4. Recording commit info...")
    stats.record_commit(auto=False, dry_run=False, edited=False)
    print("   ✓ Commit info recorded")
    print()

    # Test save_record
    print("5. Saving complete record to database...")
    stats.save_record(
        command_type="commit",
        generated_output="feat(test): test commit message for DB backend",
        repo_path="/tmp/test_repo",
        branch_name="test/db-backend",
        commit_hash="abc123def456",
        diff_content="diff --git a/test.txt b/test.txt\n+test",
        status_content="M test.txt",
        stats_content="1 file changed, 1 insertion(+)",
        commit_type="feat",
        commit_scope="test",
        breaking_change=False,
        success=True,
        error_message=None
    )
    print("   ✓ Record saved to database")
    print()

    # Test get_summary
    print("6. Getting stats summary...")
    summary = stats.get_summary()
    print(f"   ✓ Summary retrieved:")
    print(f"      - Total commands: {summary['commands']['total']}")
    print(f"      - Total API calls: {summary['api']['total_calls']}")
    print(f"      - Total tokens: {summary['api']['total_tokens']}")
    print(f"      - Commits generated: {summary['commits']['total']}")
    print()

    # Test get_history
    print("7. Getting history...")
    history = stats.storage.get_history(limit=5)
    print(f"   ✓ History retrieved: {len(history)} records")
    for record in history:
        print(f"      - {record['timestamp']}: {record['command_type']} - {record['generated_output'][:50]}...")
    print()

    # Clean up
    stats.storage.close()
    print("✅ All tests passed!")
    print()
    print(f"Database created at: /tmp/gai_test.db")
    print("You can inspect it with: sqlite3 /tmp/gai_test.db")
    print()

if __name__ == "__main__":
    test_db_backend()
