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

"""Unit tests for the pure config/format helpers of the Hanzo fine-tune trainer.
The training loop itself needs a GPU and is exercised in the cluster e2e; here we
cover the env parsing and the dataset-row formatting that decide what gets trained."""

from dataclasses import dataclass, field

import pytest

import finetune


@dataclass
class BoolCase:
    name: str
    value: str
    default: bool
    expected: bool


@pytest.mark.parametrize(
    "tc",
    [
        BoolCase("true", "true", False, True),
        BoolCase("one", "1", False, True),
        BoolCase("yes upper", "YES", False, True),
        BoolCase("false", "false", True, False),
        BoolCase("empty uses default", "", True, True),
        BoolCase("none uses default", None, False, False),
    ],
)
def test_parse_bool(tc: BoolCase):
    assert finetune.parse_bool(tc.value, tc.default) is tc.expected


def test_parse_float_and_int():
    assert finetune.parse_float("2e-4", 0.0) == pytest.approx(2e-4)
    assert finetune.parse_float("bad", 1.5) == 1.5
    assert finetune.parse_int("8", 0) == 8
    assert finetune.parse_int("8.0", 0) == 8
    assert finetune.parse_int("bad", 3) == 3


def test_is_s3():
    assert finetune.is_s3("s3://bucket/path") is True
    assert finetune.is_s3("/workspace/output") is False
    assert finetune.is_s3(None) is False


@dataclass
class FormatCase:
    name: str
    example: dict
    task: str
    expected_contains: list[str] = field(default_factory=list)


@pytest.mark.parametrize(
    "tc",
    [
        FormatCase(
            "alpaca instruction",
            {"instruction": "Sum 2+2", "input": "", "output": "4"},
            "instruct",
            ["### Instruction:", "Sum 2+2", "### Response:", "4"],
        ),
        FormatCase(
            "instruction with context",
            {"instruction": "Translate", "input": "hola", "output": "hello"},
            "instruct",
            ["Translate", "hola", "hello"],
        ),
        FormatCase(
            "chat messages",
            {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hey"}]},
            "chat",
            ["<|user|>", "hi", "<|assistant|>", "hey"],
        ),
        FormatCase(
            "prompt completion",
            {"prompt": "2+2=", "completion": "4"},
            "completion",
            ["2+2=", "4"],
        ),
        FormatCase("plain text", {"text": "raw sample"}, "completion", ["raw sample"]),
    ],
)
def test_format_example(tc: FormatCase):
    out = finetune.format_example(tc.example, tc.task)
    for needle in tc.expected_contains:
        assert needle in out


def test_config_from_env_defaults_and_overrides(monkeypatch):
    for k in list(__import__("os").environ):
        if k in (
            "METHOD", "EPOCHS", "LEARNING_RATE", "BATCH_SIZE", "QUANT_4BIT", "MAX_SEQ_LEN",
        ):
            monkeypatch.delenv(k, raising=False)
    # Defaults.
    cfg = finetune.config_from_env()
    assert cfg["method"] == "qlora"
    assert cfg["quant_4bit"] is True  # qlora ⇒ 4-bit by default
    assert cfg["max_seq_len"] == 2048
    # Overrides.
    monkeypatch.setenv("METHOD", "lora")
    monkeypatch.setenv("EPOCHS", "5")
    monkeypatch.setenv("LEARNING_RATE", "1e-4")
    monkeypatch.setenv("QUANT_4BIT", "false")
    cfg = finetune.config_from_env()
    assert cfg["method"] == "lora"
    assert cfg["epochs"] == 5.0
    assert cfg["learning_rate"] == pytest.approx(1e-4)
    assert cfg["quant_4bit"] is False
