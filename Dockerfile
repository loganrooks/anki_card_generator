# Anki Forge Docker Image
# Multi-stage build for smaller final image

# Build stage
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install package and dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .[all]


# Runtime stage
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # For PDF processing
    libmupdf-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin/anki-forge /usr/local/bin/anki-forge

# Copy source (for development/debugging)
COPY src/ ./src/

# Create non-root user for security
RUN useradd -m -s /bin/bash anki && \
    mkdir -p /data /output && \
    chown -R anki:anki /app /data /output

USER anki

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Default data directories
VOLUME ["/data", "/output"]

# Set entrypoint
ENTRYPOINT ["anki-forge"]
CMD ["--help"]

# Example usage:
# docker build -t anki-forge .
# docker run -v ~/books:/data -v ~/output:/output -e GOOGLE_API_KEY=$GOOGLE_API_KEY anki-forge generate /data/book.epub -o /output/cards.csv
