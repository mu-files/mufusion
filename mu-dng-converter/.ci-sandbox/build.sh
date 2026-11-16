#!/bin/bash
# Local CI sandbox - runs the exact commands from desktop-app.yml on a fresh copy
# Usage: ./build.sh [branch]
#   no branch: copy this checkout, uncommitted changes included
#   branch:    clone that branch from GitHub (e.g. "main")
# The macOS build is unsigned; signing and notarizing need CI's secrets.
set -e

BRANCH="$1"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CHECKOUT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_DIR="$SCRIPT_DIR/repo"
VENV_DIR="$SCRIPT_DIR/venv"

# Always start fresh: remove the old copy, build, and dist
echo "=== Cleaning previous build ==="
rm -rf "$REPO_DIR"

if [ -n "$BRANCH" ]; then
    # Like actions/checkout@v4
    echo "=== Cloning mufusion from GitHub (branch: $BRANCH) ==="
    git clone --depth 1 --branch "$BRANCH" https://github.com/mu-files/mufusion.git "$REPO_DIR"
else
    # A copy, so the workflow's version sed does not edit the checkout
    echo "=== Copying $CHECKOUT_DIR ==="
    rsync -a \
        --exclude .git --exclude .ci-sandbox --exclude venv --exclude .venv \
        --exclude build --exclude dist --exclude '*.egg-info' --exclude __pycache__ \
        --exclude tests \
        "$CHECKOUT_DIR/" "$REPO_DIR/"
fi

WORKFLOW="$REPO_DIR/.github/workflows/desktop-app.yml"
PROJECT_DIR="$REPO_DIR/mu-dng-converter"

# One-time venv setup (Python 3.12 like CI)
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating Python 3.12 venv..."
    python3.12 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
pip install -q pyyaml

# Print the run script of the workflow step with this name
step() {
    python -c '
import sys, yaml
steps = yaml.safe_load(open(sys.argv[1]))["jobs"]["build"]["steps"]
print(next(s["run"] for s in steps if s.get("name") == sys.argv[2]))
' "$WORKFLOW" "$1"
}

# The version the build step reads; the workflow's plan job sets it
export APP_VERSION="${APP_VERSION:-0.0.0}"

# Extract and run steps from desktop-app.yml
cd "$PROJECT_DIR"

echo "=== Install dependencies (from desktop-app.yml) ==="
step "Install dependencies" | bash -e

echo "=== Install macOS dependencies (from desktop-app.yml) ==="
step "Install macOS dependencies" | bash -e

echo "=== Build app (macOS) (from desktop-app.yml) ==="
step "Build app (macOS)" | PYTHONIOENCODING=utf-8 PYTHONUTF8=1 bash -e

echo ""
echo "Done. Test with:"
echo "  open $PROJECT_DIR/dist/mu-dng-converter.app"
