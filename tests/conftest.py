"""
Pytest configuration and fixtures for gai tests
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import subprocess


@pytest.fixture
def temp_dir():
    """Create a temporary directory"""
    tmpdir = tempfile.mkdtemp()
    yield Path(tmpdir)
    shutil.rmtree(tmpdir)


@pytest.fixture
def git_repo(temp_dir):
    """Create a temporary git repository"""
    subprocess.run(["git", "init"], cwd=temp_dir, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=temp_dir, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=temp_dir, check=True)

    # Create initial commit
    readme = temp_dir / "README.md"
    readme.write_text("# Test Repo\n")
    subprocess.run(["git", "add", "README.md"], cwd=temp_dir, check=True)
    subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=temp_dir, check=True)

    yield temp_dir


@pytest.fixture
def mock_config():
    """Mock configuration for testing"""
    class MockConfig:
        def __init__(self):
            self.data = {
                'ai': {
                    'model': 'test-model',
                    'max_tokens': 1000,
                    'temperature': 0.3,
                    'api_url': 'http://localhost/test'
                },
                'commit': {
                    'format': 'conventional',
                    'max_diff_tokens': 5000
                }
            }
            self.api_key = 'test-key'
            self.config_path = Path('/tmp/test.yaml')

        def get(self, key, default=None):
            keys = key.split('.')
            val = self.data
            for k in keys:
                val = val.get(k, {})
                if not isinstance(val, dict):
                    return val
            return default or val

        def set(self, key, value):
            keys = key.split('.')
            val = self.data
            for k in keys[:-1]:
                if k not in val:
                    val[k] = {}
                val = val[k]
            val[keys[-1]] = value

    return MockConfig()


@pytest.fixture
def sample_diff():
    """Sample git diff for testing"""
    return """diff --git a/test.py b/test.py
new file mode 100644
index 0000000..1234567
--- /dev/null
+++ b/test.py
@@ -0,0 +1,10 @@
+def hello():
+    print("Hello, World!")
+
+if __name__ == "__main__":
+    hello()
"""


@pytest.fixture
def sample_diff_stat():
    """Sample git diff --stat output"""
    return " test.py | 10 ++++++++++\n 1 file changed, 10 insertions(+)\n"
