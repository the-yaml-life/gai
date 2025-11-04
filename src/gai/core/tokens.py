"""
Token estimation and chunking utilities
"""

from typing import List


def estimate_tokens(text: str) -> int:
    """
    Estimate number of tokens in text.

    Rule of thumb for code/diffs:
    - ~3.5 chars = 1 token
    - More conservative: ~3 chars = 1 token
    """
    if not text:
        return 0

    # Conservative estimation
    chars = len(text)
    return chars // 3


def chunk_text(text: str, max_tokens: int, overlap: int = 100) -> List[str]:
    """
    Chunk text into pieces of max_tokens size.

    Args:
        text: Text to chunk
        max_tokens: Maximum tokens per chunk
        overlap: Number of tokens to overlap between chunks

    Returns:
        List of text chunks
    """
    if estimate_tokens(text) <= max_tokens:
        return [text]

    # Convert tokens to approximate chars
    max_chars = max_tokens * 3
    overlap_chars = overlap * 3

    chunks = []
    lines = text.split('\n')
    current_chunk = []
    current_size = 0

    for line in lines:
        line_size = len(line) + 1  # +1 for newline

        if current_size + line_size > max_chars and current_chunk:
            # Save current chunk
            chunks.append('\n'.join(current_chunk))

            # Start new chunk with overlap (last few lines)
            overlap_lines = []
            overlap_size = 0

            for prev_line in reversed(current_chunk):
                if overlap_size + len(prev_line) > overlap_chars:
                    break
                overlap_lines.insert(0, prev_line)
                overlap_size += len(prev_line)

            current_chunk = overlap_lines
            current_size = overlap_size

        current_chunk.append(line)
        current_size += line_size

    # Add final chunk
    if current_chunk:
        chunks.append('\n'.join(current_chunk))

    return chunks


def chunk_by_files(diff: str, max_tokens: int) -> List[str]:
    """
    Chunk diff by files, keeping file diffs together when possible.

    Args:
        diff: Git diff output
        max_tokens: Maximum tokens per chunk

    Returns:
        List of diff chunks
    """
    if estimate_tokens(diff) <= max_tokens:
        return [diff]

    # Split by file
    file_diffs = []
    current_file = []

    for line in diff.split('\n'):
        if line.startswith('diff --git'):
            if current_file:
                file_diffs.append('\n'.join(current_file))
            current_file = [line]
        else:
            current_file.append(line)

    if current_file:
        file_diffs.append('\n'.join(current_file))

    # Group files into chunks
    chunks = []
    current_chunk = []
    current_size = 0

    for file_diff in file_diffs:
        file_size = estimate_tokens(file_diff)

        if current_size + file_size > max_tokens and current_chunk:
            chunks.append('\n'.join(current_chunk))
            current_chunk = []
            current_size = 0

        current_chunk.append(file_diff)
        current_size += file_size

    if current_chunk:
        chunks.append('\n'.join(current_chunk))

    return chunks
