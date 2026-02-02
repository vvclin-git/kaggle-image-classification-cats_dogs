Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

uv sync
uv pip install -e .
