CIAI conda env: rdt.simpler

# ciai

`/home/panwen.hu/workspace/jian.zhang/EAI/AffordDiffusionTransformer`

`/home/panwen.hu/workspace/jian.zhang/EAI/AffordDiffusionTransformer/scripts/afford_inference_demo.py`

# h800

### 1. open depth anything v2 api flask server

```bash
tmux new -s depth
cd /home/xurongtao/minghao/Depth-Anything-V2
minghaoconda
conda activate depth
python /home/xurongtao/minghao/Depth-Anything-V2/demo/depth_api.py
```

### 2. Use RDT

No need to operate now. We use subprocess to call the RDT model.

`/home/xurongtao/jianzhang/Afford-RDT`

```bash
# cd /home/jianzhang/AffordDiffusionTransformer_deploy
cd /home/panwen.hu/workspace/jian.zhang/EAI/AffordDiffusionTransformer
CUDA_VISIBLE_DEVICES=2 python -m scripts.afford_inference_demo
```

### 3. Run simpler demo

Follow the instruction in `demo/rdt_demo.py`
