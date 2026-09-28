# LLM from Scratch: Architecture, Pretraining, and Fine-Tuning in PyTorch

A zero-framework, production-oriented implementation of an autoregressive Large Language Model (GPT-2 architecture) built natively in PyTorch. This repository covers the complete lifecycle of generative language models: tokenization, causal self-attention mechanisms, transformer architecture, pretraining on unlabeled data, supervised fine-tuning for classification, and instruction fine-tuning.

## Overview

The purpose of this repository is to construct, train, and fine-tune a Large Language Model from first principles. By eschewing high-level wrapper libraries, every mathematical and architectural layer is transparent, configurable, and mathematically verifiable.

The project is structured into three progressive development phases:
1. Architecture and Foundations: Building the data pipeline, attention mechanism, and GPT-2 transformer blocks.
2. Pretraining on Unlabeled Data: Implementing training loops, cross-entropy loss functions, perplexity tracking, and checkpointing.
3. Fine-Tuning and Alignment: Adapting the base model for downstream classification tasks and instruction-following capabilities.

## Repository Roadmap and Structure

```
.
|-- .gitignore
|-- requirements.txt
|-- README.md
|-- LICENSE
|-- gpt_modules.py
|-- the-verdict.txt
|-- 01_data_preprocessing_and_tokenization.ipynb
|-- 02_multihead_attention_mechanisms.ipynb
|-- 03_gpt2_architecture_implementation.ipynb
|-- 04_pretraining_on_unlabeled_data.ipynb
|-- 05_finetuning_for_text_classification.ipynb
`-- 06_instruction_finetuning.ipynb
```

### Module Specifications

- 01_data_preprocessing_and_tokenization.ipynb (Status: Implemented)
  Implements raw text ingestion, Byte-Pair Encoding (BPE) via tiktoken, sliding-window sequence chunking, custom PyTorch Dataset and DataLoader classes, and the integration of token and learned positional embeddings.

- 02_multihead_attention_mechanisms.ipynb (Status: Implemented)
  Formulates attention mechanics: self-attention dot products, trainable projections for Query, Key, and Value matrices, causal upper-triangular masking to preserve autoregressive constraints, and vectorized multi-head attention heads.

- 03_gpt2_architecture_implementation.ipynb (Status: Implemented)
  Assembles the core transformer: Pre-Layer Normalization, Gaussian Error Linear Units (GELU), feed-forward sub-networks with residual skip connections, model parameter scaling, and autoregressive greedy decoding.

- 04_pretraining_on_unlabeled_data.ipynb (Status: In Progress)
  Implements self-supervised pretraining on raw text: cross-entropy loss computation over vocabulary logits, training and validation tracking via perplexity, learning rate scheduling with warmup, and weight transfer from pretrained open weights.

- 05_finetuning_for_text_classification.ipynb (Status: Planned)
  Adapts the pretrained backbone for sequence classification: replacing the autoregressive language modeling head with a linear classification head, parameter freezing strategies, and fine-tuning on labeled text datasets.

- 06_instruction_finetuning.ipynb (Status: Planned)
  Implements supervised instruction alignment: prompt-response dataset formatting, selective loss masking over prompt tokens, supervised instruction tuning, and response evaluation.

- gpt_modules.py
  Shared modular repository components, including Dataset loaders (`GPTDatasetV1`, `create_dataloader_v1`) and the foundational `MultiHeadAttention` class.

- the-verdict.txt
  Sample public-domain text corpus used for data pipeline verification and tokenization benchmarking.

## Architectural Specifications

The model follows the standard GPT-2 (124M parameter) baseline configuration:

| Parameter | Value | Description |
|---|---|---|
| Vocabulary Size (vocab_size) | 50,257 | BPE tokenizer vocabulary size |
| Context Length (context_length) | 1,024 | Maximum input sequence tokens |
| Embedding Dimension (emb_dim) | 768 | Hidden state size across all layers |
| Number of Heads (n_heads) | 12 | Parallel attention heads |
| Number of Layers (n_layers) | 12 | Stacked Transformer blocks |
| Dropout Rate (drop_rate) | 0.1 | Regularization probability |
| QKV Bias (qkv_bias) | False | Bias parameters in projection layers |

## Mathematical Formulations

### Causal Multi-Head Attention

Given an input tensor X in R^(B x T x D_in):

Q = X W_q,  K = X W_k,  V = X W_v

Scaled causal dot-product attention with upper-triangular masking matrix M:

Attention(Q, K, V) = softmax((Q K^T / sqrt(d_k)) + M) V

where M_{i,j} = 0 for i >= j and M_{i,j} = -infinity for i < j.

### Transformer Block Operations

The pre-layer normalization configuration applies normalization prior to multi-head attention and feed-forward transformations:

X_1 = X + MultiHeadAttention(LayerNorm(X))
X_out = X_1 + FeedForward(LayerNorm(X_1))

where the FeedForward sub-layer is defined as:

FeedForward(Z) = Linear_2(GELU(Linear_1(Z)))

### Pretraining Objective

The model minimizes the empirical negative log-likelihood (Cross-Entropy Loss) across the sequence length T:

Loss = - (1 / T) sum_{t=1}^T log P(x_t | x_1, x_2, ..., x_{t-1})

Perplexity is tracked alongside training loss as:

Perplexity = exp(Loss)

## Getting Started

### Prerequisites

- Python 3.10 or higher
- PyTorch 2.2 or higher
- JupyterLab or VS Code Jupyter extension

### Environment Setup

1. Clone the repository:
```bash
git clone https://github.com/<your-username>/llm-from-scratch-pytorch.git
cd llm-from-scratch-pytorch
```

2. Initialize and activate an isolated virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install requirements:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Execution Order

Launch the environment with JupyterLab:
```bash
jupyter lab
```

Execute notebooks sequentially to trace the model lifecycle:
1. `01_data_preprocessing_and_tokenization.ipynb`
2. `02_multihead_attention_mechanisms.ipynb`
3. `03_gpt2_architecture_implementation.ipynb`
4. `04_pretraining_on_unlabeled_data.ipynb`
5. `05_finetuning_for_text_classification.ipynb`
6. `06_instruction_finetuning.ipynb`

## License

This project is licensed under the MIT License. See the LICENSE file for details.
