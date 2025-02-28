#!/bin/bash

# 定义脚本列表
scripts=(
    "/home/xurongtao/minghao/SimplerEnv/scripts/a0_bridge_.sh"
    "/home/xurongtao/minghao/SimplerEnv/scripts/a0_drawer_variant_agg_.sh"
    "/home/xurongtao/minghao/SimplerEnv/scripts/a0_drawer_visual_matching_.sh"
    "/home/xurongtao/minghao/SimplerEnv/scripts/a0_move_near_variant_agg_.sh"
    "/home/xurongtao/minghao/SimplerEnv/scripts/a0_move_near_visual_matching_.sh"
    "/home/xurongtao/minghao/SimplerEnv/scripts/a0_pick_coke_can_variant_agg.sh"
    "/home/xurongtao/minghao/SimplerEnv/scripts/a0_pick_coke_can_visual_matching.sh"
    "/home/xurongtao/minghao/SimplerEnv/scripts/a0_put_in_drawer_variant_agg_.sh"
    "/home/xurongtao/minghao/SimplerEnv/scripts/a0_put_in_drawer_visual_matching_.sh"
    "/home/xurongtao/minghao/SimplerEnv/scripts/misc/a0_drawer_variant_agg_alt_urdf_.sh"
    "/home/xurongtao/minghao/SimplerEnv/scripts/misc/a0_move_near_variant_agg_alt_urdf_.sh"
    "/home/xurongtao/minghao/SimplerEnv/scripts/misc/a0_pick_coke_can_variant_agg_alt_urdf_.sh"
    "/home/xurongtao/minghao/SimplerEnv/scripts/misc/octo_drawer_variant_agg_alt_urdf.sh"
    "/home/xurongtao/minghao/SimplerEnv/scripts/misc/octo_move_near_variant_agg_alt_urdf.sh"
    "/home/xurongtao/minghao/SimplerEnv/scripts/misc/octo_pick_coke_can_variant_agg_alt_urdf.sh"
)

# 运行脚本并保存输出
for script in "${scripts[@]}"; do
    if [ -f "$script" ]; then
        output_file="${script%.sh}.txt"
        output_file="$(dirname "$output_file")/output_$(basename "$output_file")"
        echo ""
        echo "Running $script..."
        source "$script" | tee "$output_file"
        echo "Output saved to $output_file"
    else
        echo "Script $script not found!"
    fi
done

echo "All scripts executed."
