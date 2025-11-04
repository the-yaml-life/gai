# Multi-stage build for gai - Git AI Assistant
#
# Usage:
#   Build:  podman build -t gai .
#   Run:    podman run -v /path/to/repo:/workspace -e GROQ_API_KEY=your_key gai commit --dry-run

# Stage 1: Builder
FROM python:3.11-alpine AS builder

WORKDIR /build

# Install build dependencies
RUN apk add --no-cache \
    gcc \
    musl-dev \
    libffi-dev \
    git

# Copy only requirements first for layer caching
COPY pyproject.toml ./

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# Stage 2: Runtime
FROM python:3.11-alpine

# Install runtime dependencies
RUN apk add --no-cache \
    git \
    && adduser -D -u 1000 gai

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=gai:gai src/ ./src/
COPY --chown=gai:gai gai.py ./
COPY --chown=gai:gai .gai.yaml ./.gai.yaml.example

# Switch to non-root user
USER gai

# Set working directory for git operations
WORKDIR /workspace

# Entry point
ENTRYPOINT ["python", "/app/gai.py"]
CMD ["--help"]

# Labels
LABEL org.opencontainers.image.title="gai"
LABEL org.opencontainers.image.description="AI-powered Git assistant for commit messages, merge reviews, and diffs"
LABEL org.opencontainers.image.authors="gai contributors"
LABEL org.opencontainers.image.source="https://github.com/your-org/gai"
LABEL org.opencontainers.image.licenses="AGPL-3.0-or-later"
