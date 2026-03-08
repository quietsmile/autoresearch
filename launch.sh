#!/bin/bash
set -e

TAG=${1:-$(date +%b%d | tr '[:upper:]' '[:lower:]')}  # e.g. "mar8"
NUM_GPUS=${2:-8}
BASE_DIR=$(git rev-parse --show-toplevel)
BASE_BRANCH=$(git branch --show-current)

echo "Launching $NUM_GPUS GPU workers with tag '$TAG'"
echo "Base dir: $BASE_DIR"
echo ""

for i in $(seq 0 $((NUM_GPUS - 1))); do
    BRANCH="autoresearch/${TAG}-gpu${i}"
    WORKTREE_DIR="${BASE_DIR}-gpu${i}"

    if [ $i -eq 0 ]; then
        # GPU 0 uses the original directory, just create the branch
        echo "GPU $i: using $BASE_DIR (branch: $BRANCH)"
        git checkout -b "$BRANCH" 2>/dev/null || true
    else
        # GPU 1-7 create new worktrees
        if [ ! -d "$WORKTREE_DIR" ]; then
            echo "GPU $i: creating worktree at $WORKTREE_DIR (branch: $BRANCH)"
            git worktree add "$WORKTREE_DIR" -b "$BRANCH"
        else
            echo "GPU $i: worktree already exists at $WORKTREE_DIR"
        fi
    fi

    # Launch tmux session
    SESSION="autoresearch-gpu${i}"
    WORK_DIR=$( [ $i -eq 0 ] && echo "$BASE_DIR" || echo "$WORKTREE_DIR" )
    tmux new-session -d -s "$SESSION" -c "$WORK_DIR" 2>/dev/null || true
    tmux send-keys -t "$SESSION" "export CUDA_VISIBLE_DEVICES=$i" Enter
    tmux send-keys -t "$SESSION" "export GPU_ID=$i" Enter
    tmux send-keys -t "$SESSION" "echo 'GPU $i ready. Start your Claude agent here.'" Enter
done

echo ""
echo "Done! Attach to each session with:"
for i in $(seq 0 $((NUM_GPUS - 1))); do
    echo "  tmux attach -t autoresearch-gpu${i}"
done
echo ""
echo "In each session, prompt the agent:"
echo "  'Have a look at program.md and kick off a new experiment!'"
