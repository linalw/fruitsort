#!/usr/bin/env bash
# 用法:
#   source scripts/env.sh              # 只加载 ROS 2 Humble
#   source scripts/env.sh maniskill3   # 同时激活 conda 环境
#   source scripts/env.sh dynvla
#
# 说明: 非交互 shell 不会加载 ~/.bashrc，所以这里显式处理 ROS 与 conda。

# ---- 项目根目录 ----
__FS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export FRUITSORT_ROOT="$__FS_ROOT"
export PYTHONPATH="$__FS_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# ---- ROS 2 Humble ----
if [ -f /opt/ros/humble/setup.bash ]; then
  source /opt/ros/humble/setup.bash
fi

# 本项目 / 用户工作空间若已编译则一并加载
for __ws in "$__FS_ROOT/install/setup.bash" "$HOME/ros2_ws/install/setup.bash"; do
  if [ -f "$__ws" ]; then source "$__ws"; fi
done
unset __ws

# ---- 可选 conda 环境 ----
if [ -n "${1:-}" ]; then
  __CONDA=/home/ubuntu/linalw/App/minconda3
  eval "$("$__CONDA/bin/conda" shell.bash hook)"
  conda activate "$1" || echo "[env.sh] conda activate $1 失败" >&2
  unset __CONDA
fi

export RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}
export RCUTILS_COLORIZED_OUTPUT=${RCUTILS_COLORIZED_OUTPUT:-1}

# ---- 常用别名（交互 shell 才有意义）----
alias gpu='nvitop'
alias top='glances'
alias fr='source scripts/env.sh'
