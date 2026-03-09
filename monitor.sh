#!/bin/bash
# monitor.sh — real-time dashboard for 8-GPU autoresearch workers
# Usage: bash monitor.sh [refresh_seconds]   (default: 10)

REFRESH=${1:-10}
BASE=$(cd "$(dirname "$0")" && pwd)

# ANSI colors
BOLD='\033[1m'
DIM='\033[2m'
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m'

gpu_dir() {
    [ "$1" -eq 0 ] && echo "$BASE" || echo "${BASE}-gpu$1"
}

# Parse best val_bpb from results.tsv (skip header, exclude crashes)
best_val() {
    local f="$1/results.tsv"
    [ -f "$f" ] || { echo ""; return; }
    tail -n +2 "$f" | awk -F'\t' '$2+0 > 0.001 {print $2}' | sort -n | head -1
}

# Count experiments (rows excluding header)
exp_count() {
    local f="$1/results.tsv"
    [ -f "$f" ] || { echo 0; return; }
    echo $(( $(wc -l < "$f") - 1 ))
}

# Is training active? (run.log modified within last 90s AND contains step pattern)
is_running() {
    local log="$1/run.log"
    [ -f "$log" ] || return 1
    local age=$(( $(date +%s) - $(stat -c %Y "$log" 2>/dev/null || echo 0) ))
    [ $age -lt 90 ] && grep -q "^step\b\|^GPU:" "$log" 2>/dev/null
}

# Get current training step line from run.log
current_step() {
    local log="$1/run.log"
    [ -f "$log" ] || return
    # step lines use \r so they may all be on one line; grab last occurrence
    tr '\r' '\n' < "$log" 2>/dev/null | grep "^step " | tail -1
}

# Get last N rows of results.tsv (excluding header)
last_results() {
    local f="$1/results.tsv"
    local n=${2:-3}
    [ -f "$f" ] || return
    tail -n +2 "$f" | tail -"$n"
}

while true; do
    clear

    echo -e "${BOLD}╔══════════════════════════════════════════════════════════════════╗${NC}"
    printf "${BOLD}║  autoresearch monitor  %-43s║${NC}\n" "$(date '+%Y-%m-%d %H:%M:%S')  refresh=${REFRESH}s"
    echo -e "${BOLD}╚══════════════════════════════════════════════════════════════════╝${NC}"
    echo ""

    GLOBAL_BEST_VAL=""
    GLOBAL_BEST_GPU=""
    GLOBAL_BEST_DESC=""

    for i in $(seq 0 7); do
        DIR=$(gpu_dir $i)

        # Skip if worktree doesn't exist
        if [ ! -d "$DIR" ]; then
            echo -e "${DIM}GPU $i: directory not found${NC}"
            echo ""
            continue
        fi

        BRANCH=$(cd "$DIR" 2>/dev/null && git branch --show-current 2>/dev/null || echo "unknown")
        N=$(exp_count "$DIR")
        BEST=$(best_val "$DIR")

        # Running status
        if is_running "$DIR"; then
            STATUS="${GREEN}● RUNNING${NC}"
            STEP_LINE=$(current_step "$DIR")
        else
            STATUS="${DIM}○ idle${NC}"
            STEP_LINE=""
        fi

        # GPU header line
        printf "${BOLD}GPU %d${NC}  ${CYAN}%-38s${NC}  exps: ${YELLOW}%-3s${NC}  best: " \
            "$i" "$BRANCH" "$N"
        if [ -n "$BEST" ]; then
            echo -ne "${GREEN}${BEST}${NC}"
        else
            echo -ne "${DIM}--${NC}"
        fi
        echo -e "  $STATUS"

        # Current step (if running)
        if [ -n "$STEP_LINE" ]; then
            echo -e "  ${DIM}${STEP_LINE}${NC}"
        fi

        # Last 3 results
        ROWS=$(last_results "$DIR" 3)
        if [ -n "$ROWS" ]; then
            while IFS=$'\t' read -r commit val mem status desc; do
                case "$status" in
                    keep)    C="${GREEN}"  ;;
                    discard) C="${YELLOW}" ;;
                    crash)   C="${RED}"    ;;
                    *)       C="${NC}"     ;;
                esac
                printf "  ${DIM}%7s${NC}  ${C}%-8s${NC}  bpb=%-10s  mem=%-7s  %s\n" \
                    "$commit" "$status" "$val" "${mem}GB" "$desc"
            done <<< "$ROWS"
        else
            echo -e "  ${DIM}(no results yet)${NC}"
        fi

        # Track global best
        if [ -n "$BEST" ]; then
            if [ -z "$GLOBAL_BEST_VAL" ] || awk "BEGIN{exit !($BEST < $GLOBAL_BEST_VAL)}"; then
                GLOBAL_BEST_VAL="$BEST"
                GLOBAL_BEST_GPU="$i"
                GLOBAL_BEST_DESC=$(tail -n +2 "$DIR/results.tsv" | \
                    awk -F'\t' '$2+0 > 0.001 {print $2"\t"$5}' | sort -n | head -1 | cut -f2)
            fi
        fi

        echo ""
    done

    # Global best bar
    echo -e "${BOLD}━━━ BEST ACROSS ALL GPUs ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    if [ -n "$GLOBAL_BEST_VAL" ]; then
        echo -e "  val_bpb: ${BOLD}${GREEN}${GLOBAL_BEST_VAL}${NC}   GPU ${GLOBAL_BEST_GPU}   ${GLOBAL_BEST_DESC}"
    else
        echo -e "  ${DIM}(no results yet)${NC}"
    fi
    echo ""
    echo -e "${DIM}Press Ctrl+C to exit${NC}"

    sleep "$REFRESH"
done
