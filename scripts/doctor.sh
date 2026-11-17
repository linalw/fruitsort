#!/usr/bin/env bash
# 环境体检：把机器、GPU、conda 环境、ROS、网络状态快照到 docs/env-report.md
# 用法: bash scripts/doctor.sh            （只打印）
#       bash scripts/doctor.sh --write    （同时写文件）
set -u
export LC_ALL=C   # 保证 free/df 等输出可解析

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ROOT=/home/ubuntu/linalw/App/minconda3
OUT="$ROOT/docs/env-report.md"
WRITE=0
[ "${1:-}" = "--write" ] && WRITE=1

TMP="$(mktemp)"
exec 3>&1

say() { echo "$*" | tee -a "$TMP" >&3; }
sec() { say ""; say "## $*"; say ""; }

say "# 环境报告（自动生成，勿手改）"
say ""
say "- 生成时间: $(date '+%Y-%m-%d %H:%M:%S %Z')"
say "- 主机: $(hostname) / 用户: $(whoami)"
say "- 系统: $(. /etc/os-release 2>/dev/null && echo "$PRETTY_NAME") 内核 $(uname -r) / $(uname -m)"

sec "GPU"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used,temperature.gpu,pstate \
    --format=csv,noheader | sed 's/^/`/; s/$/`/' | tee -a "$TMP" >&3
  say ""
  say "- 显存占用:"
  nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader | sed 's/^/  - /' | tee -a "$TMP" >&3
else
  say "nvidia-smi 不可用"
fi
say ""
say "- nvcc: $(command -v nvcc >/dev/null 2>&1 && nvcc --version | tail -1 || echo '未安装/不在 PATH')"
say "- torch 默认架构: sm_120 (Blackwell) → 需要 CUDA 12.8+ 的 wheel"

sec "CPU / 内存 / 磁盘"
say "- CPU: $(nproc) 核  $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs)"
say "- 内存: $(free -g | awk '/^Mem:/{print $2" GB total, "$7" GB available"}')"
say "- 交换: $(free -g | awk '/^Swap:/{print $2" GB"}')"
say ""
say "| 挂载点 | 大小 | 可用 | 用途 |"
say "| --- | --- | --- | --- |"
df -h / /media/ubuntu/66A27065A2703B9D 2>/dev/null | awk 'NR>1{printf "| `%s` | %s | %s | %s |\n",$6,$2,$4,($6=="/"?"系统":"大数据盘")}' | tee -a "$TMP" >&3

sec "conda 环境"
say "conda: $CONDA_ROOT"
say ""
say "| env | Python | 关键包 |"
say "| --- | --- | --- |"
PKGS="torch torchvision numpy transformers lerobot mani_skill sapien isaaclab gymnasium mujoco opencv-python h5py rerun-sdk wandb"
for dir in "$CONDA_ROOT"/envs/*/; do
  [ -d "$dir" ] || continue
  name="$(basename "$dir")"
  py="$dir/bin/python"
  [ -x "$py" ] || continue
  pyver="$("$py" -c 'import sys;print(".".join(map(str,sys.version_info[:3])))' 2>/dev/null)"
  pkgs="$("$py" - "$PKGS" <<'PY' 2>/dev/null
import sys
from importlib.metadata import version, PackageNotFoundError
out=[]
for n in sys.argv[1].split():
    try: out.append(f"{n} {version(n)}")
    except PackageNotFoundError: pass
print(", ".join(out) if out else "—")
PY
)"
  say "| \`$name\` | $pyver | $pkgs |"
done

sec "ROS"
if [ -f /opt/ros/humble/setup.bash ]; then
  # ROS 的 setup.bash 引用了未定义变量，必须在 set +u 下 source
  set +u
  # shellcheck disable=SC1091
  source /opt/ros/humble/setup.bash
  set -u
  say "- 发行版: ${ROS_DISTRO:-unknown}"
  say "- 包数量: $(ros2 pkg list 2>/dev/null | wc -l)"
  say "- 组件: $(ls /opt/ros/humble/share 2>/dev/null | grep -E '^(rviz2|nav2_bringup|gazebo|ros_gz|moveit|realsense|canopen)' | tr '\n' ' ')"
else
  say "- 未发现 /opt/ros/humble"
fi
say "- 工作空间: $(ls -d "$HOME"/ros2_ws/src/* 2>/dev/null | xargs -r -n1 basename | tr '\n' ' ')"

sec "网络（5s 超时）"
for url in "https://export.arxiv.org/api/query?search_query=all:robot&max_results=1" https://huggingface.co https://pypi.org; do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "$url" 2>/dev/null || echo ERR)
  say "- $url → \`$code\`"
done

sec "提示"
say "- 非交互 shell 先 \`source scripts/env.sh\` 才能用 ros2 / conda env。"
say "- 新建环境用 Python 3.11；安装 torch 需 cu128+（5090 是 sm_120）。"

if [ "$WRITE" = "1" ]; then
  cp "$TMP" "$OUT"
  echo "已写入 $OUT" >&3
fi
exec 3>&-
rm -f "$TMP"
