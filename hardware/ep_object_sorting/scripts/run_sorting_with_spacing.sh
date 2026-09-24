#!/usr/bin/env bash
set -eo pipefail

# Usage:
#   bash run_sorting_with_spacing.sh ROW_M COLUMN_M [HEADING_DEG] [OBJECT_LIMIT]
# Example:
#   bash run_sorting_with_spacing.sh 0.32 0.26 0 1

row_spacing="${1:-0.32}"
column_spacing="${2:-0.26}"
heading_correction="${3:-0.0}"
object_limit="${4:-1}"

positive_number='^[0-9]+([.][0-9]+)?$'
signed_number='^[-+]?[0-9]+([.][0-9]+)?$'

if [[ ! "${row_spacing}" =~ ${positive_number} ]]; then
    echo "错误：行间距必须是正数，单位为米，例如 0.32" >&2
    exit 2
fi
if [[ ! "${column_spacing}" =~ ${positive_number} ]]; then
    echo "错误：列间距必须是正数，单位为米，例如 0.26" >&2
    exit 2
fi
if [[ ! "${heading_correction}" =~ ${signed_number} ]]; then
    echo "错误：航向修正必须是数字，例如 -30" >&2
    exit 2
fi
if [[ ! "${object_limit}" =~ ^[1-6]$ ]]; then
    echo "错误：测试物体数量必须是 1 到 6" >&2
    exit 2
fi

experiment3_workspace="${EXPERIMENT3_WORKSPACE:-/home/lemon/experiment3_ws}"
arm_workspace="${ROBOMASTER_ARM_WORKSPACE:-/home/lemon/arm_ws}"

source /opt/ros/humble/setup.bash
source "${arm_workspace}/install/setup.bash"
source "${experiment3_workspace}/install/setup.bash"

echo "启动参数："
echo "  行间距       = ${row_spacing} m"
echo "  列间距       = ${column_spacing} m"
echo "  航向修正     = ${heading_correction} deg"
echo "  测试网格数量 = ${object_limit}"

exec ros2 launch ep_object_sorting full_sorting_system.launch.py \
    connection_type:=ap \
    row_spacing:="${row_spacing}" \
    column_spacing:="${column_spacing}" \
    bin_outside_margin:=0.15 \
    initial_heading_correction_deg:="${heading_correction}" \
    object_limit:="${object_limit}" \
    venv_python:="${experiment3_workspace}/.venv/bin/python3"

