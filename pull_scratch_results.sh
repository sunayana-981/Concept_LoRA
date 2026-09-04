#!/usr/bin/env bash
# =============================================================================
# pull_scratch_results.sh
#
# Collects checkpoints that Turing jobs staged on a compute node's scratch
# volume (/tmp/$USER, the 14T ada-lv_scratch mount) back to this local machine.
#
# Why this exists: Turing home dirs are a 50GB NFS quota that the full
# masked-finetune sweep (~13GB of checkpoints) doesn't fit in, so jobs train
# into node scratch instead. Scratch is node-local and not visible from the
# login node, and the compute nodes don't accept this machine's SSH key --
# but the login node *can* reach them, so results are streamed as a tar
# through the existing ControlMaster connection. Nothing is written to the
# home quota along the way.
#
# Usage:
#   bash pull_scratch_results.sh            # pull from every node running our jobs
#   bash pull_scratch_results.sh node14     # pull from one specific node
#   KEEP_REMOTE=1 bash pull_scratch_results.sh   # don't delete after pulling
# =============================================================================

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CTRL="$HOME/.ssh/cm-turing"
REMOTE="sunayana.samavedam@turing.iiit.ac.in"
DEST="${REPO}/turing_scratch_results"
KEEP_REMOTE="${KEEP_REMOTE:-0}"

ssh -O check -o ControlPath="$CTRL" "$REMOTE" >/dev/null 2>&1 || {
    echo "[FATAL] SSH control socket is down. Reopen it with:"
    echo "  ssh -o ControlMaster=yes -o ControlPersist=12h -o ControlPath=$CTRL $REMOTE"
    exit 1
}

# Which nodes to check: an explicit argument, else every node currently
# running one of our jobs.
if [ $# -ge 1 ]; then
    NODES="$1"
else
    NODES=$(ssh -o ControlPath="$CTRL" "$REMOTE" \
        "squeue -u \$USER -h -t RUNNING -o '%N' | tr ',' '\n' | sort -u" 2>/dev/null)
fi

if [ -z "${NODES// /}" ]; then
    echo "No running jobs found; pass a node name explicitly to check its scratch."
    exit 0
fi

mkdir -p "$DEST"
echo "Destination: $DEST"

for node in $NODES; do
    echo ""
    echo "=== $node ==="
    listing=$(ssh -o ControlPath="$CTRL" "$REMOTE" \
        "ssh -o BatchMode=yes -o StrictHostKeyChecking=no $node 'ls /tmp/\$USER 2>/dev/null'" 2>/dev/null)
    if [ -z "${listing// /}" ]; then
        echo "  nothing staged on this node's scratch"
        continue
    fi
    echo "  staged: $(echo "$listing" | tr '\n' ' ')"

    size=$(ssh -o ControlPath="$CTRL" "$REMOTE" \
        "ssh -o BatchMode=yes -o StrictHostKeyChecking=no $node 'du -sh /tmp/\$USER 2>/dev/null | cut -f1'" 2>/dev/null)
    echo "  size: ${size:-unknown}, streaming..."

    if ssh -o ControlPath="$CTRL" "$REMOTE" \
        "ssh -o BatchMode=yes -o StrictHostKeyChecking=no $node 'tar -C /tmp/\$USER -cf - .'" \
        | tar -C "$DEST" -xf - ; then
        echo "  pulled OK -> $DEST"
        if [ "$KEEP_REMOTE" != "1" ]; then
            ssh -o ControlPath="$CTRL" "$REMOTE" \
                "ssh -o BatchMode=yes -o StrictHostKeyChecking=no $node 'rm -rf /tmp/\$USER/*'" 2>/dev/null \
                && echo "  cleared scratch on $node"
        fi
    else
        echo "  [WARN] pull from $node failed; scratch left untouched"
    fi
done

echo ""
echo "Collected so far:"
find "$DEST" -name "*.pt" -type f 2>/dev/null | wc -l | xargs echo "  checkpoints:"
du -sh "$DEST" 2>/dev/null | awk '{print "  size: " $1}'
