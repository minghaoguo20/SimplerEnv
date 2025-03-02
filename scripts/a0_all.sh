#!/bin/bash

expoert CUDA_VISIBLE_DEVICES=3

note="run"
current_time=$(date +%Y%m%d_%H%M%S)
base_output_dir="scripts/output"

output_dir="${base_output_dir}/${note}_${current_time}"
mkdir -p "$output_dir"

# 定义脚本列表
scripts=(
    "/home/xurongtao/minghao/SimplerEnv/scripts/a0_drawer_variant_agg_.sh"
    "/home/xurongtao/minghao/SimplerEnv/scripts/a0_drawer_visual_matching_.sh"
    "/home/xurongtao/minghao/SimplerEnv/scripts/a0_move_near_variant_agg_.sh"
    "/home/xurongtao/minghao/SimplerEnv/scripts/a0_move_near_visual_matching_.sh"
    "/home/xurongtao/minghao/SimplerEnv/scripts/a0_pick_coke_can_variant_agg.sh"
    "/home/xurongtao/minghao/SimplerEnv/scripts/a0_pick_coke_can_visual_matching.sh"
    "/home/xurongtao/minghao/SimplerEnv/scripts/a0_put_in_drawer_variant_agg_.sh"
    "/home/xurongtao/minghao/SimplerEnv/scripts/a0_put_in_drawer_visual_matching_.sh"

    # "/home/xurongtao/minghao/SimplerEnv/scripts/a0_bridge_.sh"
    # "/home/xurongtao/minghao/SimplerEnv/scripts/misc/a0_drawer_variant_agg_alt_urdf_.sh"
    # "/home/xurongtao/minghao/SimplerEnv/scripts/misc/a0_move_near_variant_agg_alt_urdf_.sh"
    # "/home/xurongtao/minghao/SimplerEnv/scripts/misc/a0_pick_coke_can_variant_agg_alt_urdf_.sh"
)

# 运行脚本并保存输出到指定的输出目录中
for script in "${scripts[@]}"; do
    if [ -f "$script" ]; then
        # 提取脚本的基础文件名（不含路径和扩展名）
        base_name=$(basename "$script" .sh)
        # 构造输出文件路径：指定输出目录 + 基础文件名 + .txt扩展名
        output_file="${output_dir}/${base_name}.txt"
        output_dir_epx="${output_dir}/${base_name}"
        mkdir -p "$output_dir_epx"
        echo ""
        echo ">>> Running $script..."
        # 运行脚本，并同时将输出显示在终端和保存到指定的输出文件中
        source "$script" "$output_dir_epx" | tee "$output_file"
        echo "Output saved to $output_file"
    else
        echo "Script $script not found!"
    fi
done

echo "All scripts executed."
