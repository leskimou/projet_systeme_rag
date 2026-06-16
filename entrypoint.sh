#!/bin/sh
set -e

if [ ! -f "vector_db/index.faiss" ]; then
    echo "vector_db/ absent, construction de l'index initial..."
    uv run python -m utils.indexer
fi

exec "$@"
