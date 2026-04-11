#!/usr/bin/env bash
# 启用本仓库虚拟环境（需在 bash/zsh 里用 source，不要直接执行）
#
#   cd /path/to/RL_project
#   source ./activate_venv.sh
#
# 之后提示符前会出现 (.venv)，python/pip 均指向 .venv。

_REPO="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
if [[ -f "$_REPO/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "$_REPO/.venv/bin/activate"
  echo "[ok] 已启用 venv: $VIRTUAL_ENV"
else
  echo "[err] 未找到 $_REPO/.venv/bin/activate" >&2
  echo "      可先执行: cd \"$_REPO\" && python3 -m venv .venv" >&2
  return 1 2>/dev/null || exit 1
fi
