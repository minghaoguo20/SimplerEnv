<!-- 

CIAI conda env: rdt.simpler

# ciai

`/home/panwen.hu/workspace/jian.zhang/EAI/AffordDiffusionTransformer`

`/home/panwen.hu/workspace/jian.zhang/EAI/AffordDiffusionTransformer/scripts/afford_inference_demo.py` 

-->

# h800

### 1. open depth anything v2 api flask server

```bash
tmux new -s depth
cd /home/xurongtao/minghao/Depth-Anything-V2
minghaoconda
conda activate depth
CUDA_VISIBLE_DEVICES=4 python /home/xurongtao/minghao/Depth-Anything-V2/demo/depth_api.py --port=5001
```

### 2. Use RAM

```bash
conda deactivate
tmux new -s ram
cd /home/xurongtao/minghao/RAM_code
minghaoconda
conda activate ram
CUDA_VISIBLE_DEVICES=4 python run_realworld/run_server_api.py --port=5002
CUDA_VISIBLE_DEVICES=4 /mnt/data/xurongtao/minghao/conda/envs/simpler_env/bin/python run_realworld/run_server_api.py --port=5002
```

### 3. Use RDT

No need to operate now. We use subprocess to call the RDT model.

`/home/xurongtao/jianzhang/Afford-RDT`

```bash
# cd /home/jianzhang/AffordDiffusionTransformer_deploy
cd /home/panwen.hu/workspace/jian.zhang/EAI/AffordDiffusionTransformer
CUDA_VISIBLE_DEVICES=2 python -m scripts.afford_inference_demo
```

### 4. Run simpler demo

For RDT:
- Do Not use : 0, 4, 6
- Free to use : 3, 5

For simpler:
- Do Not use : 0
- Free to use : 3

```bash
ps aux | grep X
kill -9 xxxx
nohup sudo X :0 &
export DISPLAY=:0
CUDA_VISIBLE_DEVICES=0 python demo/rdt_demo.py --rdt_cuda 3
```

### TO DO

- [ ] convert ram_result to action: `demo/rdt_demo.py` line 82
