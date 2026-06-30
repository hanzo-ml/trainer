# Hanzo fine-tuning trainer

The node entrypoint for the `hanzo-ft-{lora,qlora,full}` ClusterTrainingRuntimes
(`manifests/base/runtimes/hanzo_finetune.yaml`). One env-driven image fine-tunes
**any** HuggingFace causal LM with LoRA, QLoRA (4-bit) or full fine-tuning using
`transformers` + `peft` + `trl`.

## How it runs

A `TrainJob` references a runtime (e.g. `hanzo-ft-qlora`) and supplies:

- `initializer.model.storageUri` / `initializer.dataset.storageUri` — the base
  model and dataset (`hf://…` or `s3://…`). The runtime's model/dataset
  initializers download them to `/workspace/model` and `/workspace/dataset` before
  this entrypoint runs. A `secretRef` (the `hf-token` Secret) authenticates private
  or gated repos.
- `trainer.resourcesPerNode` — the GPU count.
- `trainer.env` — every hyperparameter (see the env contract in `finetune.py`).

The entrypoint trains, then writes a **servable** HuggingFace-format model to
`OUTPUT_DIR` (LoRA/QLoRA adapters are merged into the base, so KServe's HuggingFace
runtime can serve any method). When `OUTPUT_DIR` is an `s3://` URI the result is
uploaded to the org's S3 Space.

## Build

The image is built in CI (not locally) and pinned by the operator:

```bash
docker build . -f cmd/trainers/hanzo/Dockerfile -t ghcr.io/hanzoai/finetune-runtime:0.1.0
```

## Test

The pure config/format helpers are unit-tested (no GPU):

```bash
cd cmd/trainers/hanzo && pytest finetune_test.py
```
