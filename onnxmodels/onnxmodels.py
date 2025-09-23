"""onnxmodels module
This module provides a wrapper OnnxModel class that allows to use onnx models with python.
"""

import numpy as np
import onnx
from onnxruntime import InferenceSession
import pandas as pd

__all__ = ["OnnxModel"]


"""
onnxmodels module
This module provides a wrapper OnnxModel class that allows to use onnx models with python.
"""

import numpy as np
import onnx
from onnxruntime import InferenceSession
import pandas as pd

__all__ = ["OnnxModel"]


class OnnxModel:
    """
    ONNX model wrapper for inference with flexible input types.
    This class allows loading an ONNX model and performing inference using
    numpy.ndarray, pandas.DataFrame, or dict inputs.
    Parameters
    ----------
    model_path : str
        Path to the ONNX model file.
    input_labels : list of str
        List of input feature names.
    output_labels : list of str
        List of output feature names.
    """

    def __init__(
        self, model_path: str, input_labels: list[str], output_labels: list[str]
    ):
        self.model_path = model_path
        self._input_labels = input_labels
        self._output_labels = output_labels
        self.session = InferenceSession(model_path)
        self.model = onnx.load(model_path)

    @property
    def input_labels(self):
        """Get the input feature labels."""
        return self._input_labels

    @property
    def output_labels(self):
        """Get the output feature labels."""
        return self._output_labels

    def predict(self, data):
        """
        Run inference on the input data.
        Parameters
        ----------
        data : np.ndarray, pd.DataFrame, list, or dict
            Input data to be passed to the ONNX model.
        Returns
        -------
        result : np.ndarray, dict, or pd.DataFrame
            Model predictions in a format matching the input type.
        Raises
        ------
        ValueError
            If input data shape or columns do not match expected input labels.
        TypeError
            If input or output type is unsupported.
        """
        target_cols = len(self._input_labels)
        wrong_cols = f"Expected input tensor with shape (N, {target_cols})"
        col_list = f"DataFrame or dict must contain columns/keys: {self._input_labels}"

        # Handle numpy array
        if isinstance(data, np.ndarray):
            if data.ndim != 2 or data.shape[1] != target_cols:
                raise ValueError(wrong_cols)
            vals = data.astype(np.float32)
            source = "ndarray"

        # Handle pandas DataFrame
        elif isinstance(data, pd.DataFrame):
            if not all(label in data.columns for label in self._input_labels):
                raise ValueError(col_list)
            vals = data[self._input_labels].values.astype(np.float32)
            source = "dataframe"

        # Handle dict
        elif isinstance(data, dict):
            if not all(label in data.keys() for label in self._input_labels):
                raise ValueError(col_list)
            vals = []
            for i in self._input_labels:
                v = data[i]
                if isinstance(v, (pd.DataFrame, pd.Series)):
                    vals.append(np.asarray(v).astype(np.float32).flatten())
                elif isinstance(v, list):
                    vals.append(np.asarray(v, dtype=np.float32).flatten())
                else:
                    vals.append(np.asarray(v).astype(np.float32).flatten())
            vals = np.stack(vals, axis=1)
            source = "dict"

        # Handle list (assume list of lists or list of values)
        elif isinstance(data, list):
            arr = np.asarray(data, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(-1, target_cols)
            if arr.shape[1] != target_cols:
                raise ValueError(wrong_cols)
            vals = arr
            source = "list"

        else:
            raise TypeError("Unsupported input type")

        # Inference
        inputs = {self.session.get_inputs()[0].name: vals}
        outputs = self.session.run(None, inputs)[0]

        # Output formatting
        if source == "ndarray" or source == "list":
            return outputs
        if source == "dict":
            return {
                i: v.astype(np.float32).flatten()
                for i, v in zip(self.output_labels, outputs.T)  # type: ignore
            }
        if source == "dataframe":
            return pd.DataFrame(
                data=outputs,  # type: ignore
                index=data.index,  # type: ignore
                columns=self.output_labels,
            )

        raise TypeError("Unsupported output type")

    def __call__(self, data):
        """Call the model on input data (alias for predict)."""
        return self.predict(data)
