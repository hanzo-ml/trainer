# Copyright 2026 The Kubeflow authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Hanzo fine-tuning trainer — the node entrypoint for the hanzo-ft-* runtimes.

It fine-tunes ANY HuggingFace causal LM with LoRA, QLoRA (4-bit) or full
fine-tuning, configured ENTIRELY from environment variables set by the Hanzo Cloud
fine-tuning broker. The base model and dataset are already on disk under
MODEL_DIR / DATASET_DIR (downloaded by the TrainJob's model/dataset initializers);
this script trains and writes a SERVABLE HuggingFace-format directory to OUTPUT_DIR
(adapters are merged into the base for lora/qlora, so one serving path — KServe's
HuggingFace runtime — works for every method). When OUTPUT_DIR is an `s3://` URI
the result is uploaded to the org's S3 Space.

Env contract (all optional except the dirs, which the runtime defaults):
  METHOD                 lora | qlora | full        (default qlora)
  MODEL_DIR              local pre-trained model dir (default /workspace/model)
  DATASET_DIR            local dataset dir           (default /workspace/dataset)
  OUTPUT_DIR             local path or s3:// URI     (default /workspace/output)
  TASK                   instruct | chat | completion
  EPOCHS, LEARNING_RATE, BATCH_SIZE, GRAD_ACCUM, MAX_SEQ_LEN
  LORA_RANK, LORA_ALPHA, LORA_DROPOUT
  QUANT_4BIT, GRADIENT_CHECKPOINTING  (true/false)
  WARMUP_RATIO, WEIGHT_DECAY
"""

import logging
import os
import shutil
import tempfile

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("hanzo-finetune")


# ── Pure config/format helpers (unit-tested in finetune_test.py) ─────────────


def parse_bool(value: str, default: bool = False) -> bool:
    """Parse a truthy env string."""
    if value is None or value == "":
        return default
    return value.strip().lower() in ("1", "true", "yes", "y", "on")


def parse_float(value: str, default: float) -> float:
    """Parse a float env string, falling back to default."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_int(value: str, default: int) -> int:
    """Parse an int env string, falling back to default."""
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def config_from_env() -> dict:
    """Build the training configuration from the environment."""
    method = (os.environ.get("METHOD") or "qlora").strip().lower()
    if method not in ("lora", "qlora", "full"):
        method = "qlora"
    return {
        "method": method,
        "model_dir": os.environ.get("MODEL_DIR", "/workspace/model"),
        "dataset_dir": os.environ.get("DATASET_DIR", "/workspace/dataset"),
        "output_dir": os.environ.get("OUTPUT_DIR", "/workspace/output"),
        "task": (os.environ.get("TASK") or "instruct").strip().lower(),
        "epochs": parse_float(os.environ.get("EPOCHS"), 3.0),
        "learning_rate": parse_float(os.environ.get("LEARNING_RATE"), 2e-4),
        "batch_size": parse_int(os.environ.get("BATCH_SIZE"), 2),
        "grad_accum": parse_int(os.environ.get("GRAD_ACCUM"), 8),
        "max_seq_len": parse_int(os.environ.get("MAX_SEQ_LEN"), 2048),
        "lora_rank": parse_int(os.environ.get("LORA_RANK"), 16),
        "lora_alpha": parse_int(os.environ.get("LORA_ALPHA"), 32),
        "lora_dropout": parse_float(os.environ.get("LORA_DROPOUT"), 0.05),
        "quant_4bit": parse_bool(os.environ.get("QUANT_4BIT"), method == "qlora"),
        "grad_checkpointing": parse_bool(os.environ.get("GRADIENT_CHECKPOINTING"), True),
        "warmup_ratio": parse_float(os.environ.get("WARMUP_RATIO"), 0.03),
        "weight_decay": parse_float(os.environ.get("WEIGHT_DECAY"), 0.01),
    }


def format_example(example: dict, task: str) -> str:
    """Render one dataset row to a single training string, covering the common
    instruction/chat/completion schemas so most HuggingFace datasets work as-is."""
    # Chat schema: {messages: [{role, content}, ...]}.
    msgs = example.get("messages")
    if isinstance(msgs, list) and msgs:
        parts = []
        for m in msgs:
            role = str(m.get("role", "user"))
            content = str(m.get("content", ""))
            parts.append(f"<|{role}|>\n{content}")
        return "\n".join(parts)
    # Instruction schema: {instruction, input?, output}.
    if "instruction" in example and ("output" in example or "response" in example):
        instruction = str(example.get("instruction", ""))
        ctx = str(example.get("input", "") or "")
        output = str(example.get("output", example.get("response", "")))
        prompt = instruction if not ctx else f"{instruction}\n\n{ctx}"
        return f"### Instruction:\n{prompt}\n\n### Response:\n{output}"
    # Prompt/completion schema.
    if "prompt" in example and "completion" in example:
        return f"{example['prompt']}{example['completion']}"
    # Plain text.
    if "text" in example:
        return str(example["text"])
    # Fallback: join all string values.
    return "\n".join(str(v) for v in example.values() if isinstance(v, str))


def is_s3(uri: str) -> bool:
    return isinstance(uri, str) and uri.startswith("s3://")


# ── Training (IO + GPU; orchestrated in main) ────────────────────────────────


def load_training_dataset(dataset_dir: str):
    """Load the initializer-downloaded dataset, trying the common on-disk layouts."""
    from datasets import load_dataset, load_from_disk

    candidates_data = os.path.join(dataset_dir, "data")
    # 1) datasets.save_to_disk layout.
    try:
        ds = load_from_disk(dataset_dir)
        return ds["train"] if hasattr(ds, "keys") and "train" in ds else ds
    except Exception:  # noqa: BLE001 — fall through to other layouts
        pass
    # 2) parquet files under data/ (the kubeflow dataset-initializer layout).
    data_dir = candidates_data if os.path.isdir(candidates_data) else dataset_dir
    for fmt in ("parquet", "json"):
        try:
            ds = load_dataset(fmt, data_dir=data_dir, split="train")
            return ds
        except Exception:  # noqa: BLE001
            continue
    # 3) a dataset directory load_dataset understands directly.
    ds = load_dataset(dataset_dir, split="train")
    return ds


def main() -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    cfg = config_from_env()
    log.info("Hanzo fine-tune starting: %s", {k: cfg[k] for k in ("method", "task", "epochs", "learning_rate", "max_seq_len")})

    local_out = cfg["output_dir"]
    upload_to = None
    if is_s3(cfg["output_dir"]):
        upload_to = cfg["output_dir"]
        local_out = os.path.join(tempfile.gettempdir(), "hanzo-ft-output")
    os.makedirs(local_out, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_dir"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = None
    if cfg["quant_4bit"]:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_dir"],
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
        device_map="auto" if quant_config is not None else None,
        trust_remote_code=True,
    )
    model.config.use_cache = False

    peft_config = None
    if cfg["method"] in ("lora", "qlora"):
        from peft import LoraConfig, prepare_model_for_kbit_training

        if cfg["quant_4bit"]:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=cfg["grad_checkpointing"]
            )
        peft_config = LoraConfig(
            r=cfg["lora_rank"],
            lora_alpha=cfg["lora_alpha"],
            lora_dropout=cfg["lora_dropout"],
            target_modules="all-linear",
            bias="none",
            task_type="CAUSAL_LM",
        )

    train_ds = load_training_dataset(cfg["dataset_dir"])
    log.info("Loaded %d training examples", len(train_ds))

    args = SFTConfig(
        output_dir=os.path.join(local_out, "checkpoints"),
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"],
        learning_rate=cfg["learning_rate"],
        warmup_ratio=cfg["warmup_ratio"],
        weight_decay=cfg["weight_decay"],
        max_seq_length=cfg["max_seq_len"],
        gradient_checkpointing=cfg["grad_checkpointing"],
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
        formatting_func=lambda ex: format_example(ex, cfg["task"]),
    )
    trainer.train()

    # Write a SERVABLE model: merge adapters into the base for lora/qlora.
    log.info("Training complete — writing servable model to %s", local_out)
    if cfg["method"] in ("lora", "qlora"):
        adapter_dir = os.path.join(local_out, "adapter")
        trainer.model.save_pretrained(adapter_dir)
        _merge_adapter(cfg["model_dir"], adapter_dir, local_out)
    else:
        trainer.save_model(local_out)
    tokenizer.save_pretrained(local_out)

    if upload_to:
        _upload_dir(local_out, upload_to)
    log.info("Done.")


def _merge_adapter(base_dir: str, adapter_dir: str, out_dir: str) -> None:
    """Reload the base in bf16, apply the trained adapter, merge, and save a
    standalone HuggingFace model dir (so any vanilla serving runtime can load it)."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(
        base_dir, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    merged = PeftModel.from_pretrained(base, adapter_dir)
    merged = merged.merge_and_unload()
    merged.save_pretrained(out_dir, safe_serialization=True)
    shutil.rmtree(adapter_dir, ignore_errors=True)


def _upload_dir(local_dir: str, s3_uri: str) -> None:
    """Upload a directory to S3 (the org's Space). Uses s3fs; endpoint/credentials
    come from the standard AWS_*/S3_ENDPOINT_URL env the pod's storage secret sets."""
    import s3fs

    endpoint = os.environ.get("S3_ENDPOINT_URL") or os.environ.get("AWS_ENDPOINT_URL")
    fs = s3fs.S3FileSystem(client_kwargs={"endpoint_url": endpoint} if endpoint else {})
    dest = s3_uri.rstrip("/")
    log.info("Uploading %s -> %s", local_dir, dest)
    fs.put(local_dir, dest, recursive=True)


if __name__ == "__main__":
    main()
