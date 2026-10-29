#!/usr/bin/env bash
# Source after setting your own endpoint and benchmark installation paths.
export AZURE_OPENAI_API_VERSION="${AZURE_OPENAI_API_VERSION:-2024-12-01-preview}"
export WM_MODEL="${WM_MODEL:-gpt-5.4}"
export PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
