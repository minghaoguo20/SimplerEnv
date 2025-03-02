<!-- 

CIAI conda env: rdt.simpler

# ciai

`/home/panwen.hu/workspace/jian.zhang/EAI/AffordDiffusionTransformer`

`/home/panwen.hu/workspace/jian.zhang/EAI/AffordDiffusionTransformer/scripts/afford_inference_demo.py` 

-->

# Test In Simpler

## 1. open depth anything v2 api flask server

```bash
tmux new -s depth
cd /home/xurongtao/minghao/Depth-Anything-V2
minghaoconda
conda activate depth
CUDA_VISIBLE_DEVICES=3 python /home/xurongtao/minghao/Depth-Anything-V2/demo/depth_api.py --port=5001
```

## 2. Use RAM

### h800
```bash
conda deactivate
tmux new -s ram
cd /home/xurongtao/minghao/RAM_code
minghaoconda
conda activate ram
CUDA_VISIBLE_DEVICES=3 python run_realworld/run_server_api.py --port=5002
```

<!-- ### db
```bash
tmux new -s ram
cd /remote-home/minghao/code/aff/RAM
conda activate ram
CUDA_VISIBLE_DEVICES=3 python run_realworld/run_server_api.py
``` -->

## 3. Use RDT

<!-- No need to operate now. We use subprocess to call the RDT model.

`/home/xurongtao/jianzhang/Afford-RDT`

```bash
# cd /home/jianzhang/AffordDiffusionTransformer_deploy
cd /home/panwen.hu/workspace/jian.zhang/EAI/AffordDiffusionTransformer
CUDA_VISIBLE_DEVICES=2 python -m scripts.afford_inference_demo
``` -->

For RDT:
- Do Not use : 0, 4, 6
- Free to use : 3, 5

```bash
tmux new -s rdt.api
cd /home/xurongtao/jianzhang/Afford-RDT-deploy
defaultconda
conda activate rdt
CUDA_VISIBLE_DEVICES=0 python scripts/afford_server_minghao.py
```

## 4. Run simpler demo

For simpler:
- Do Not use : 0
- Free to use : 3, 5

```bash
ps aux | grep X
kill -9 xxxx
nohup sudo X :0 &
export DISPLAY=:0
CUDA_VISIBLE_DEVICES=3 python demo/rdt_demo.py
```

## TO DO

- [x] convert ram_result to action: `demo/rdt_demo.py` line 82
- [ ] label rotation

# Result

- ⏳ a0_drawer_variant_agg_.txt
- ⏳ a0_drawer_visual_matching_.txt
- ❌ a0_move_near_variant_agg_.txt
- ❌ a0_move_near_visual_matching_.txt
- ❌ a0_pick_coke_can_variant_agg.txt
- ❌ a0_pick_coke_can_visual_matching.txt
- ❌ a0_put_in_drawer_variant_agg_.txt
- ⏳ a0_put_in_drawer_visual_matching_.txt

# Debug

## evaluate new model

```shell
(simpler_new) xurongtao@computer4:~/minghao/SimplerEnv$ CUDA_VISIBLE_DEVICES=3 python demo/rdt_demo.py  --repeat_n=1
```

✅ new model results: `/home/xurongtao/minghao/SimplerEnv/output/rdt/20250302_211510`

## evaluate with all shell scripts

```shell
(simpler_new) xurongtao@computer4:~/minghao/SimplerEnv$ CUDA_VISIBLE_DEVICES=3 source scripts/a0_all.sh
```

⭕️ not run yet

## quaternion

```shell
(simpler_new) xurongtao@computer4:~/minghao/SimplerEnv$ CUDA_VISIBLE_DEVICES=3 python demo/rdt_demo_find_angle.py --task_names=google_robot_open_drawer --repeat_n=1
```

⏳ 找角度，遍历所有angle: `/home/xurongtao/minghao/SimplerEnv/output/rdt/20250302_223426`

