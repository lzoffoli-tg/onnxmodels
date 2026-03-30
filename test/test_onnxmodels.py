"""onnxmodels test module"""

from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import pytest

# Import diretto dal package
from onnxmodels.onnxmodels import OnnxModel


@pytest.fixture
def dummy_onnx_session():
    dummy_session = MagicMock()
    dummy_session.get_inputs.return_value = [MagicMock()]
    dummy_session.get_inputs()[0].name = "input0"
    dummy_session.run.return_value = [
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    ]
    return dummy_session


@pytest.fixture
def patch_onnx(dummy_onnx_session):
    # Patch SUL MODULO DEL PACKAGE, non su onnxruntime globale!
    with patch(
        "onnxmodels.onnxmodels.InferenceSession", return_value=dummy_onnx_session
    ), patch("onnxmodels.onnxmodels.onnx.load", return_value=MagicMock()):
        yield


def test_predict_numpy(patch_onnx):
    model = OnnxModel("dummy.onnx", ["a", "b"], ["out1", "out2"])
    arr = np.array([[10, 20], [30, 40]], dtype=np.float32)
    out = model.predict(arr)
    assert isinstance(out, np.ndarray)
    assert out.shape == (2, 2)


def test_predict_dataframe(patch_onnx):
    model = OnnxModel("dummy.onnx", ["a", "b"], ["out1", "out2"])
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    out = model.predict(df)
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["out1", "out2"]
    assert out.shape == (2, 2)


def test_predict_dict(patch_onnx):
    model = OnnxModel("dummy.onnx", ["a", "b"], ["out1", "out2"])
    d = {"a": [1, 2], "b": [3, 4]}
    out = model.predict(d)
    assert isinstance(out, dict)
    assert set(out.keys()) == {"out1", "out2"}
    for v in out.values():
        assert isinstance(v, np.ndarray)
        assert v.shape == (2,)


def test_predict_list(patch_onnx):
    model = OnnxModel("dummy.onnx", ["a", "b"], ["out1", "out2"])
    l = [[1, 2], [3, 4]]
    out = model.predict(l)
    assert isinstance(out, np.ndarray)
    assert out.shape == (2, 2)


def test_predict_wrong_shape_numpy(patch_onnx):
    model = OnnxModel("dummy.onnx", ["a", "b"], ["out1", "out2"])
    arr = np.array([[1, 2, 3]])
    with pytest.raises(ValueError):
        model.predict(arr)


def test_predict_missing_column_dataframe(patch_onnx):
    model = OnnxModel("dummy.onnx", ["a", "b"], ["out1", "out2"])
    df = pd.DataFrame({"a": [1, 2]})
    with pytest.raises(ValueError):
        model.predict(df)


def test_predict_unsupported_type(patch_onnx):
    model = OnnxModel("dummy.onnx", ["a", "b"], ["out1", "out2"])
    with pytest.raises(TypeError):
        model.predict(42)
