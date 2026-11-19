#!/usr/bin/env bash
# Publish the documentation to GitHub Pages by pushing the built site to the gh-pages branch.
#
# Usage:
#   ./scripts/publish_docs.sh [ref]   # tag, branch or SHA to build the docs from; defaults to HEAD
#
# Publishing is manual because GitHub Actions IPs are blocked by the org IP allowlist:
# run this from a machine that is allowed to push to the gh-pages branch.
set -euo pipefail

REF="${1:-HEAD}"

repo_root="$(git rev-parse --show-toplevel)"
src_wt="$(mktemp -d)"
pages_wt="$(mktemp -d)"
cleanup() {
  git -C "$repo_root" worktree remove --force "$src_wt" || true
  git -C "$repo_root" worktree remove --force "$pages_wt" || true
}
trap cleanup EXIT

git -C "$repo_root" fetch origin gh-pages
git -C "$repo_root" worktree add --detach "$src_wt" "$REF"

(
  cd "$src_wt"
  uv sync --frozen --all-extras --group docs
  uv run python docs/gen_api_pages.py
  uv run zensical build
)

git -C "$repo_root" worktree add --detach "$pages_wt" origin/gh-pages
rsync -a --delete --exclude=.git "$src_wt/site/" "$pages_wt/"

(
  cd "$pages_wt"
  git add -A
  if ! git diff --cached --quiet; then
    git commit -m "deploy: $(git -C "$src_wt" rev-parse HEAD)"
  fi
  git push origin HEAD:gh-pages
)

echo "Published docs built from $REF to the gh-pages branch."
