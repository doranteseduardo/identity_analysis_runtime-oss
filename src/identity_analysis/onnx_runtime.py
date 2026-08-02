"""Bounded ONNX Runtime session construction."""

import os
from pathlib import Path

import onnxruntime as ort


def _positive_thread_count(name: str, default: int) -> int:
    value = int(os.environ.get(name, str(default)))
    if value < 1:
        raise ValueError(f"{name} must be at least 1")
    return value


def session_options() -> ort.SessionOptions:
    options = ort.SessionOptions()
    options.intra_op_num_threads = _positive_thread_count(
        "IDENTITY_ANALYSIS_ONNX_INTRA_OP_THREADS",
        1,
    )
    options.inter_op_num_threads = _positive_thread_count(
        "IDENTITY_ANALYSIS_ONNX_INTER_OP_THREADS",
        1,
    )
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return options


def create_session(model: str | Path | bytes) -> ort.InferenceSession:
    return ort.InferenceSession(
        model,
        sess_options=session_options(),
        providers=["CPUExecutionProvider"],
    )
