#!/usr/bin/env bash
# Patch vLLM's hy_v3 reasoning + tool-call parsers for kodelow/Hy3-NVFP4-W4A16,
# whose special tokens carry a ":opensource" suffix. Applied at container runtime
# (writable layer only — the image is never modified). Idempotent. Verbatim 11-token
# set from Tony DeAngelo's proven recipe (2 reasoning + 9 tool tokens).
set -euo pipefail

# Locate vLLM robustly across images (stage-c conda /opt/env, eugr system dist-packages).
VP="$(python3 -c 'import vllm,os;print(os.path.dirname(vllm.__file__))' 2>/dev/null || true)"
[ -z "$VP" ] && VP="$(python -c 'import vllm,os;print(os.path.dirname(vllm.__file__))' 2>/dev/null || true)"
[ -z "$VP" ] && VP="$(find /usr/local/lib /opt/env /usr/lib -maxdepth 7 -type d -name vllm \( -path '*site-packages*' -o -path '*dist-packages*' \) 2>/dev/null | head -1)"

RP="$VP/reasoning/hy_v3_reasoning_parser.py"
TP="$VP/tool_parsers/hy_v3_tool_parser.py"
for f in "$RP" "$TP"; do
  [ -f "$f" ] || { echo "patch_hy3_parsers: missing $f (vLLM resolved to '$VP')" >&2; exit 1; }
done

if grep -q 'think:opensource' "$RP" 2>/dev/null; then
  echo "patch_hy3_parsers: already applied (idempotent skip) at $VP"
  exit 0
fi

sed -i 's|"<think>"|"<think:opensource>"|g; s|"</think>"|"</think:opensource>"|g' "$RP"
sed -i 's|"<tool_calls>"|"<tool_calls:opensource>"|g; s|"</tool_calls>"|"</tool_calls:opensource>"|g; s|"<tool_call>"|"<tool_call:opensource>"|g; s|"</tool_call>"|"</tool_call:opensource>"|g; s|"<tool_sep>"|"<tool_sep:opensource>"|g; s|"<arg_key>"|"<arg_key:opensource>"|g; s|"</arg_key>"|"</arg_key:opensource>"|g; s|"<arg_value>"|"<arg_value:opensource>"|g; s|"</arg_value>"|"</arg_value:opensource>"|g' "$TP"

echo "patch_hy3_parsers: applied :opensource suffix to hy_v3 reasoning + tool parsers at $VP"
