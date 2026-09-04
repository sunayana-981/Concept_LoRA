#!/usr/bin/env bash
# =============================================================================
# sync_overleaf.sh
#
# Pushes the contents of Arxiv-version/ to the Overleaf project at
# https://www.overleaf.com/project/6a5669a6dee30e702b95e527
#
# Why a script rather than a second git remote on this repo: Overleaf expects
# the main .tex at the *project root*, but our sources live one level down in
# Arxiv-version/. Adding git.overleaf.com as a remote here would push the whole
# Concept_LoRA tree -- code, weights config, everything -- and Overleaf would
# find no main document. So this clones the Overleaf project separately and
# copies Arxiv-version/'s contents in at top level.
#
# Authentication: Overleaf git uses a token, not your account password.
# Generate one at Overleaf -> Account Settings -> Git integration, then when
# git prompts:
#     Username: your Overleaf email
#     Password: the token
# Run `git config --global credential.helper store` first if you'd rather not
# be asked every time.
#
# Usage:
#   bash sync_overleaf.sh                 # add/update files, leave others alone
#   bash sync_overleaf.sh --prune         # also delete Overleaf files we no
#                                         # longer have (destructive -- only do
#                                         # this if nobody edits on Overleaf)
# =============================================================================

set -uo pipefail

PROJECT_ID="6a5669a6dee30e702b95e527"
REMOTE="https://git.overleaf.com/${PROJECT_ID}"
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="$(mktemp -d)"
PRUNE=0
[ "${1:-}" = "--prune" ] && PRUNE=1

cleanup() { rm -rf "$WORK"; }
trap cleanup EXIT

echo "Source:  $SRC"
echo "Overleaf: $REMOTE"
echo ""

echo "==> Cloning the Overleaf project (you'll be asked for your Git token)"
if ! git clone "$REMOTE" "$WORK/project"; then
    echo ""
    echo "[FATAL] Clone failed. Overleaf git needs a token, not your password:"
    echo "  Overleaf -> Account Settings -> Git integration -> generate a token,"
    echo "  then use your Overleaf email as the username and the token as the"
    echo "  password."
    exit 1
fi

cd "$WORK/project" || exit 1
echo ""
echo "==> Overleaf currently holds:"
git ls-files | sed 's/^/    /' | head -30
echo ""

# Copy the paper in at project root. Excluded: build artifacts (Overleaf makes
# its own), the arXiv upload zip, the compiled PDF, and the superseded method
# draft -- none of which belong in an Overleaf source project.
RSYNC_ARGS=(-a
    --exclude '.git/'
    --exclude '*.aux' --exclude '*.bbl' --exclude '*.blg'
    --exclude '*.out' --exclude '*.synctex.gz' --exclude '*.log'
    --exclude '*.fls' --exclude '*.fdb_latexmk'
    --exclude '*.zip'
    --exclude '*.pdf'
    --exclude 'sec/old4_method.tex'
    --exclude 'sync_overleaf.sh'
)
# Figure PDFs are real content, not build output -- re-include them after the
# blanket *.pdf exclusion above drops the compiled document.
RSYNC_ARGS+=(--include 'figures/***')

if [ "$PRUNE" -eq 1 ]; then
    echo "==> --prune given: files on Overleaf but not in Arxiv-version/ will be DELETED"
    RSYNC_ARGS+=(--delete)
fi

rsync "${RSYNC_ARGS[@]}" "$SRC"/ ./

git add -A
if git diff --cached --quiet; then
    echo ""
    echo "Nothing to sync -- Overleaf already matches Arxiv-version/."
    exit 0
fi

echo ""
echo "==> Changes to push:"
git diff --cached --stat | tail -20
echo ""

git commit -q -m "Sync from Concept_LoRA/Arxiv-version

Rebuttal evidence and expanded experiment matrix folded into the paper;
fixes for the unclosed \\todo in the intro, the xcolor option clash, and
44 duplicate bib entries. Builds clean at 37 pages."

echo "==> Pushing to Overleaf"
if git push origin master 2>/dev/null || git push origin HEAD; then
    echo ""
    echo "Synced. Open https://www.overleaf.com/project/${PROJECT_ID} to recompile."
else
    echo ""
    echo "[FATAL] Push rejected. If someone edited on Overleaf since the clone,"
    echo "        rerun this script -- it clones fresh each time."
    exit 1
fi
