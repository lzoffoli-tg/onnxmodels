# onnxmodels

**onnxmodels** is a Python package that provides a simple and robust wrapper class for running inference with ONNX models. It supports flexible input types such as `numpy.ndarray`, `pandas.DataFrame`, `dict`, and `list`.

## Features

- **Easy ONNX inference** with a single class (`OnnxModel`)
- Supports multiple input types: NumPy arrays, Pandas DataFrames, dictionaries, and lists
- Output format matches the input type for seamless integration
- Robust error handling and input validation
- Easily integrable into data science and production pipelines

## Installation

Make sure you have the following dependencies installed:

```
pip install numpy pandas onnx onnxruntime
```
