# Optimized SSM vs Transformer Benchmark

## Overview

This project compares two deep learning architectures:

* A classical Transformer based on PyTorch's `TransformerEncoder`
* An optimized SSM (State Space Model) inspired by Mamba

The goal is to evaluate how both architectures behave as the sequence length increases.

The benchmark measures:

* Forward pass execution time
* GPU memory consumption
* Scalability on long sequences

---

# Motivation

Transformers are widely used in modern large language models such as GPT.

However, their self-attention mechanism has quadratic complexity with respect to the sequence length.

O(n^2)

This becomes expensive for long-context processing.

SSM-based architectures such as Mamba aim to reduce this cost using linear-time sequence processing.

O(n)

This benchmark demonstrates the scalability difference between these approaches.

---

# Features

* Transformer vs SSM comparison
* GPU execution benchmark
* VRAM usage measurement
* Multiple sequence lengths
* Automatic graph generation
* Log-scale scalability visualization

---

# Tested Sequence Lengths

```python id="2q6u4z"
[256, 512, 1024, 2048, 4096, 8192]
```

---

# Models

## Transformer

Implemented using:

```python id="v8lk39"
nn.TransformerEncoder
```

Characteristics:

* Multi-head self-attention
* Quadratic complexity
* High memory consumption on long sequences

---

## FastSSM

Custom optimized SSM-inspired model using:

* Depthwise convolutions
* Gating mechanism
* Linear-style sequence processing

Characteristics:

* Better scalability
* Lower GPU memory usage
* Faster inference on long sequences

---

# Benchmark Metrics

The benchmark measures:

## Execution Time

Median forward pass latency in milliseconds.

## GPU Memory

Maximum GPU VRAM allocated during inference.

---

# Example Results

| Sequence Length | Transformer | SSM      | Speedup |
| --------------- | ----------- | -------- | ------- |
| 256             | 16.64 ms    | 2.17 ms  | 7.66×   |
| 1024            | 67.96 ms    | 5.00 ms  | 13.60×  |
| 4096            | 690.81 ms   | 19.82 ms | 34.85×  |
| 8192            | 2583.79 ms  | 40.55 ms | 63.72×  |

---

# Key Observation

As the sequence length increases:

* Transformer latency grows rapidly
* GPU memory consumption explodes
* SSM remains significantly more stable

At 8192 tokens:

* Transformer: ~2.5 seconds
* SSM: ~40 milliseconds

This demonstrates the scalability advantage of linear architectures.

---

# Generated Visualizations

The script generates:

1. Forward pass time
2. Log-scale scalability graph
3. Speedup comparison
4. GPU memory usage

Output image:

```text id="9h36wn"
optimized_ssm_vs_transformer.png
```

---

# Installation

Install dependencies:

```bash
pip install matplotlib numpy torch
```

---

# Run

Execute the notebook or script:

```bash
python benchmark.py
```

---

# Hardware Used

Example benchmark configuration:

* GPU: Tesla T4
* CUDA enabled
* PyTorch with `torch.compile()`

---

# Notes

This project uses a simplified SSM-inspired implementation.

It is not the official Mamba implementation,
but it reproduces the main idea of linear sequence scaling.

---

# Conclusion

This benchmark highlights why SSM-based architectures such as Mamba are becoming increasingly important for:

* Long-context language models
* Audio processing
* Video understanding
* Genomics
* Efficient large-scale inference

Compared to Transformers, linear architectures provide:

* Better scalability
* Lower memory usage
* Faster inference on long sequences
