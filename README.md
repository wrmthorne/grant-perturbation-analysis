# grant-perturbation-analysis

Repository containing reference code and details for the paper "Evaluating LLM-Based Grant Proposal Review via Structured Perturbations" ([ArXiv](https://arxiv.org/abs/2603.08281)).

## Setup

Install and package management is administered using uv.

```bash
uv venv
source .venv/bin/activate
uv pip install -U vllm --torch-backend auto

# install the nightly build of vLLM for GLM-4.7
uv pip install -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly
uv pip install git+https://github.com/huggingface/transformers.git

# Install individual packages
uv pip install inspect-ai pandas openai anthropic pydantic asyncio scipy markdown matplotlib scikit-learn
```

## Usage

### Stage 1: Data Collection

Single evaluation:
```bash
uv run inspect eval collect_{baseline,council,sectioned}.py \
  -T include-perturbed=false \
  -T include-original=true \
  --model vllm/openai/gpt-oss-20b
```

Optionally exclude either original reviews or perturbations:
```bash
inspect eval collect_{baseline,council,sectioned}.py \
  --model vllm/openai/gpt-oss-20b
```

Batch experiments:
```bash
uv run run_batch_experiments.py \
  --epochs 5 \
  --limit 10
```

Edit `MODEL_CONFIGS` in `run_batch_experiments.py` to configure experiments.

Results saved to `./logs/` by Inspect.

```bash
OPENAI_API_KEY=inspect-ai \
OPENAI_BASE_URL=http://0.0.0.0:30000/v1 \
uv run run_batch_experiments.py \
  --epochs 1 \
  --config-path configs/gpt-oss-20b.json \
  --system baseline sectioned council \
  --max-connections 50
```

### Load and Annotate Log

Read in the log from raw model output collection and inspect perturbation samples to see if the perturbation or a direct consequence of it was mentioned negatively in the review comments.

```bash
uv run inspect eval annotate_perturbations.py \
  -T log_path=logs/<log_file>.eval
```

### Perturbation Detection Experiment

```bash
# 
INSPECT_LOG_DIR=./logs_20b uv run inspect eval perturbation_detection.py \
  --model vllm/zai-org/GLM-4.7-Flash \
  -M speculative-config.method mtp \
  -M speculative-config.num_speculative_tokens 1 \
  -M tool-call-parser glm47 \
  -M reasoning-parser glm45 \
  -M enable-auto-tool-choice \
  -M max-model-len 25k \
  --temperature 0.7 \
  --top-p 1.0 \
  --retry-on-error=3 \
  --continue-on-fail
```

## Citation

```bibtex
@misc{thorne2026evaluatingllmbasedgrantproposal,
      title={Evaluating LLM-Based Grant Proposal Review via Structured Perturbations}, 
      author={William Thorne and Joseph James and Yang Wang and Chenghua Lin and Diana Maynard},
      year={2026},
      eprint={2603.08281},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2603.08281}, 
}
```