"""
Unit tests for token estimation and chunking
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from gai.core.tokens import estimate_tokens, chunk_text, chunk_by_files


class TestEstimateTokens:
    """Test token estimation"""

    def test_empty_string(self):
        assert estimate_tokens("") == 0

    def test_simple_text(self):
        # Roughly 3 chars = 1 token
        text = "Hello, World!"  # 13 chars
        tokens = estimate_tokens(text)
        assert tokens == 4  # 13 // 3

    def test_code(self):
        code = """def hello():
    print("Hello")"""
        tokens = estimate_tokens(code)
        assert tokens > 0
        assert tokens < len(code)  # Tokens < characters


class TestChunkText:
    """Test text chunking"""

    def test_no_chunking_needed(self):
        text = "Short text"
        chunks = chunk_text(text, max_tokens=1000)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chunking_by_lines(self):
        text = "\n".join([f"Line {i}" for i in range(100)])
        chunks = chunk_text(text, max_tokens=50, overlap=10)
        assert len(chunks) > 1
        # Each chunk should be list of lines joined
        for chunk in chunks:
            assert isinstance(chunk, str)
            assert len(chunk) > 0


class TestChunkByFiles:
    """Test file-based chunking"""

    def test_single_file_diff(self, sample_diff):
        chunks = chunk_by_files(sample_diff, max_tokens=1000)
        assert len(chunks) == 1

    def test_multiple_files_diff(self):
        diff = """diff --git a/file1.py b/file1.py
new file mode 100644
--- /dev/null
+++ b/file1.py
@@ -0,0 +1,5 @@
+print("file1")
diff --git a/file2.py b/file2.py
new file mode 100644
--- /dev/null
+++ b/file2.py
@@ -0,0 +1,5 @@
+print("file2")
"""
        # With very small max_tokens, should split into 2 chunks
        chunks = chunk_by_files(diff, max_tokens=20)
        assert len(chunks) >= 2
