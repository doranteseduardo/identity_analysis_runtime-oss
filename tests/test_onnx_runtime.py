import pytest

from identity_analysis.onnx_runtime import session_options


def test_onnx_session_threads_default_to_one(monkeypatch) -> None:
    monkeypatch.delenv("IDENTITY_ANALYSIS_ONNX_INTRA_OP_THREADS", raising=False)
    monkeypatch.delenv("IDENTITY_ANALYSIS_ONNX_INTER_OP_THREADS", raising=False)

    options = session_options()

    assert options.intra_op_num_threads == 1
    assert options.inter_op_num_threads == 1


def test_onnx_session_thread_limits_are_configurable(monkeypatch) -> None:
    monkeypatch.setenv("IDENTITY_ANALYSIS_ONNX_INTRA_OP_THREADS", "2")
    monkeypatch.setenv("IDENTITY_ANALYSIS_ONNX_INTER_OP_THREADS", "3")

    options = session_options()

    assert options.intra_op_num_threads == 2
    assert options.inter_op_num_threads == 3


@pytest.mark.parametrize("value", ["0", "-1"])
def test_onnx_session_thread_limits_must_be_positive(monkeypatch, value) -> None:
    monkeypatch.setenv("IDENTITY_ANALYSIS_ONNX_INTRA_OP_THREADS", value)

    with pytest.raises(ValueError):
        session_options()
