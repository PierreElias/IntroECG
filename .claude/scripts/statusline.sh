#!/usr/bin/env bash
# Claude Code statusline — 2-line status bar

INPUT="$(cat)"

# Parse all fields via env var using | separator (paths can contain spaces)
IFS='|' read -r SESSION_ID MODEL_DISPLAY CURRENT_DIR TOTAL_COST_USD CONTEXT_PCT \
         TOTAL_DURATION_MS RATE_5H RATE_7D <<EOF
$(CLAUDE_INPUT="$INPUT" python3 - <<'PYEOF'
import json, sys, os
try:
    d = json.loads(os.environ.get('CLAUDE_INPUT', '{}'))
except Exception:
    d = {}
def g(obj, *keys):
    for k in keys:
        if not isinstance(obj, dict): return ''
        obj = obj.get(k, '')
        if obj is None or obj == '': return ''
    return str(obj)
print('|'.join([
    g(d,'session_id'),
    g(d,'model','display_name'),
    g(d,'workspace','current_dir'),
    g(d,'cost','total_cost_usd'),
    g(d,'context_window','used_percentage'),
    g(d,'cost','total_duration_ms'),
    g(d,'rate_limits','five_hour','used_percentage'),
    g(d,'rate_limits','seven_day','used_percentage'),
]))
PYEOF
)
EOF

# ── Git info with 5-second cache ──────────────────────────────────────────────
GIT_BRANCH=""; GIT_STAGED=0; GIT_MODIFIED=0

if [ -n "$SESSION_ID" ] && [ -n "$CURRENT_DIR" ]; then
    CACHE_FILE="/tmp/statusline-git-cache-${SESSION_ID}"
    NOW=$(date +%s)
    CACHE_AGE=999
    if [ -f "$CACHE_FILE" ]; then
        MTIME=$(stat -f "%m" "$CACHE_FILE" 2>/dev/null || stat -c "%Y" "$CACHE_FILE" 2>/dev/null || echo 0)
        CACHE_AGE=$(( NOW - MTIME ))
    fi
    if [ "$CACHE_AGE" -ge 5 ]; then
        if git -C "$CURRENT_DIR" rev-parse --git-dir > /dev/null 2>&1; then
            BRANCH=$(git -C "$CURRENT_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "")
            STAGED=$(git -C "$CURRENT_DIR" diff --cached --name-only 2>/dev/null | wc -l | tr -d ' ')
            MODIFIED=$(git -C "$CURRENT_DIR" diff --name-only 2>/dev/null | wc -l | tr -d ' ')
            printf "%s\t%s\t%s" "$BRANCH" "$STAGED" "$MODIFIED" > "$CACHE_FILE"
        else
            printf "\t0\t0" > "$CACHE_FILE"
        fi
    fi
    if [ -f "$CACHE_FILE" ]; then
        GIT_BRANCH=$(  awk -F'\t' '{print $1}' "$CACHE_FILE")
        GIT_STAGED=$(  awk -F'\t' '{print $2}' "$CACHE_FILE")
        GIT_MODIFIED=$(awk -F'\t' '{print $3}' "$CACHE_FILE")
    fi
fi

# ── Format fields ──────────────────────────────────────────────────────────────
DIR_BASE=""
[ -n "$CURRENT_DIR" ] && DIR_BASE=$(basename "$CURRENT_DIR")

COST_FMT='$0.00'
if [ -n "$TOTAL_COST_USD" ] && [ "$TOTAL_COST_USD" != "0" ] && [ "$TOTAL_COST_USD" != "0.0" ]; then
    COST_FMT=$(awk -v c="$TOTAL_COST_USD" 'BEGIN { printf "$%.2f", c }')
fi

DURATION_FMT="0m 0s"
if [ -n "$TOTAL_DURATION_MS" ] && [ "$TOTAL_DURATION_MS" != "0" ]; then
    DURATION_FMT=$(awk -v ms="$TOTAL_DURATION_MS" 'BEGIN { s=int(ms/1000); printf "%dm %ds", int(s/60), s%60 }')
fi

# Context bar: 10 chars, color-coded by usage
CONTEXT_BAR="░░░░░░░░░░"
if [ -n "$CONTEXT_PCT" ]; then
    PCT_INT=$(awk -v p="$CONTEXT_PCT" 'BEGIN { printf "%d", int(p) }')
    FILLED=$(( PCT_INT / 10 ))
    [ "$FILLED" -gt 10 ] && FILLED=10
    EMPTY=$(( 10 - FILLED ))
    BAR_F=""; i=0; while [ $i -lt $FILLED ]; do BAR_F="${BAR_F}█"; i=$((i+1)); done
    BAR_E=""; i=0; while [ $i -lt $EMPTY  ]; do BAR_E="${BAR_E}░"; i=$((i+1)); done
    if   [ "$PCT_INT" -ge 90 ]; then CTX_ESC="\033[31m"
    elif [ "$PCT_INT" -ge 70 ]; then CTX_ESC="\033[33m"
    else                              CTX_ESC="\033[32m"; fi
    CONTEXT_BAR="${CTX_ESC}${BAR_F}\033[0m${BAR_E} ${PCT_INT}%"
fi

# Git display string
GIT_DISPLAY=""
if [ -n "$GIT_BRANCH" ]; then
    GIT_DISPLAY="🌿 $GIT_BRANCH"
    [ "${GIT_STAGED:-0}" -gt 0 ]   && GIT_DISPLAY="${GIT_DISPLAY} +${GIT_STAGED}"
    [ "${GIT_MODIFIED:-0}" -gt 0 ] && GIT_DISPLAY="${GIT_DISPLAY} ~${GIT_MODIFIED}"
fi

# Rate limits (omitted when not present)
RATE_FMT=""
if [ -n "$RATE_5H" ] && [ "$RATE_5H" != "0" ]; then
    RATE_FMT="5h:$(awk -v r="$RATE_5H" 'BEGIN { printf "%.0f%%", r }')"
fi
if [ -n "$RATE_7D" ] && [ "$RATE_7D" != "0" ]; then
    [ -n "$RATE_FMT" ] && RATE_FMT="${RATE_FMT} "
    RATE_FMT="${RATE_FMT}7d:$(awk -v r="$RATE_7D" 'BEGIN { printf "%.0f%%", r }')"
fi

# ── Output 2 lines ─────────────────────────────────────────────────────────────
LINE1="\033[36m${MODEL_DISPLAY}\033[0m"
[ -n "$DIR_BASE"    ] && LINE1="${LINE1} 📁 ${DIR_BASE}"
[ -n "$GIT_DISPLAY" ] && LINE1="${LINE1} | ${GIT_DISPLAY}"
[ -n "$RATE_FMT"    ] && LINE1="${LINE1} | ${RATE_FMT}"

LINE2="${CONTEXT_BAR} | \033[33m${COST_FMT}\033[0m | ⏱ ${DURATION_FMT}"

printf "%b\n%b\n" "$LINE1" "$LINE2"
