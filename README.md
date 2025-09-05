# 100 + Essential Linear Algebra Operations For Deep Learning

* Companion to my book of the same name
* Jupyter notebooks for problems 1 to 109
* AGS

---

## How LLMs Work

* AGS
* How LLMs work under the hood

## Link

* https://github.com/rcalix1/DeepLearningAlgorithms/tree/main/SecondEdition/Chapter10_Transformers/GPTs

# 🔢 Dot Product Example with PyTorch

This is a minimal example that demonstrates how to compute the **dot product** of two vectors using PyTorch.


## 🧪 Example Code

```python
import torch

# Define two 1D tensors (vectors)
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

# Compute dot product using torch.dot
dot_product = torch.dot(a, b)

print(f"Dot product: {dot_product.item()}")
```

## ✅ Output

```
Dot product: 32.0
```

## 📘 Explanation

The dot product of two vectors is calculated as:

```
1×4 + 2×5 + 3×6 = 4 + 10 + 18 = 32
```

## 🧠 Notes

- `torch.dot` requires both inputs to be 1D tensors of the same length.
- For matrix-style multiplication, use `torch.matmul` instead.


## Matrix Multiplication

![matmul](matmulImage.png)

# 🧼 Matrix Multiplication Example in PyTorch

This minimal example demonstrates how **matrix multiplication** works in PyTorch using `torch.matmul`, a foundational operation in deep learning and linear algebra.

---

## 📌 Objective

Multiply a matrix `x` of shape `[150, 4]` with a weight matrix `w` of shape `[4, 6]` to produce an output matrix `y` of shape `[150, 6]`.

This operation is commonly used in:

* Linear layers (`y = x @ W`)
* Feature transformations
* Neural network forward passes

---

## 💾 Code Overview

```python
import torch

# Input matrix: 150 samples with 4 features each
x = torch.randn(150, 4)

# Weight matrix: transforms 4D features into 6D
w = torch.randn(4, 6)

# Matrix multiplication: outputs shape [150, 6]
y = torch.matmul(x, w)
```

---

## 📀 Shapes Summary

| Variable | Shape     | Description                       |
| -------- | --------- | --------------------------------- |
| `x`      | \[150, 4] | 150 samples, each with 4 features |
| `w`      | \[4, 6]   | Transformation matrix             |
| `y`      | \[150, 6] | Output: 150 samples, now 6D       |

---

## 🧠 Why This Matters

This is the core of how neural networks compute outputs:
A layer transforms input features into a new representation by **multiplying by a learned weight matrix**.

This example is the essence of:

```python
y = x @ W + b
```

used in every `nn.Linear` layer.

---

## 🚀 Run It

To execute this code:

```bash
pip install torch
python your_script.py
```

Or run it inside a Jupyter notebook cell.

---

## ✅ Output

```python
print(y.shape)
# Output: torch.Size([150, 6])
```

This confirms that 150 input vectors were each projected into 6-dimensional space.

---


# 🧠 LLMs Under the Hood: Understanding Attention in Transformers

This repository walks through the **Attention Mechanism** at the core of Transformer models like GPT, BERT, and LLaMA — implemented from scratch in PyTorch.

---

## 📌 Objective

Implement and visualize how **Queries (Q)**, **Keys (K)**, and **Values (V)** are derived from token embeddings, how dot-product attention is computed, and how **causal masking** enables autoregressive behavior.

---

## 🔄 Workflow Overview

### 1. Input Tensor

We simulate a batch of 32 sequences, each 40 tokens long, with an embedding size of 512.

```python
x = torch.randn(32, 40, 512)
```

---

### 2. Compute Q, K, V Projections

We use learned projection weights and biases to map the input embeddings into lower-dimensional representations:

```python
Q = x @ wq + bq  # shape: (32, 40, 64)
K = x @ wk + bk
V = x @ wv + bv
```

---

### 3. Compute Attention Scores

We compute dot-product attention by multiplying Q and the transpose of K:

```python
attention_scores = Q @ K.transpose(-2, -1)  # shape: (32, 40, 40)
```

This measures how much each token should attend to others in the sequence.

---

### 4. Apply Causal Masking

To prevent the model from looking ahead (important in autoregressive models like GPT), we apply a lower-triangular mask:

```python
tril = torch.tril(torch.ones(40, 40))
attention_scores = attention_scores.masked_fill(tril == 0, float('-inf'))
```

---

### 5. Softmax Over Attention Scores

We convert scores into normalized attention weights:

```python
attention_probs = F.softmax(attention_scores, dim=-1)
```

---

### 6. Compute Weighted Output

We perform a weighted sum over the values `V` using attention weights:

```python
out = attention_probs @ V  # shape: (32, 40, 64)
```

---

### 7. Simulate Multi-Head Attention

We simulate 8 attention heads by replicating the output and concatenating across the feature dimension:

```python
out_cat = torch.cat([out] * 8, dim=-1)  # shape: (32, 40, 512)
```

---

### 8. Final Output Projection

We map the concatenated output back to the original embedding dimension:

```python
z = out_cat @ w0 + b0  # shape: (32, 40, 512)
```

---

## 🧠 Key Concepts

* **Self-Attention**: Enables each token to attend to others.
* **Causal Masking**: Ensures left-to-right generation.
* **Multi-Head Attention**: Captures diverse relationships in parallel.
* **Learned Projections**: Linear transformations to Q, K, and V spaces.

---

## 💪 How to Run

Install PyTorch:

```bash
pip install torch
```

Then run the script:

```bash
python attention_demo.py
```

---

## 📚 Recommended Reading

* [Attention is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)


---

# 🧠 Baby Transformer (GPT-style) from Scratch

This project implements a **super simple GPT-style Transformer block** in PyTorch. It includes token embeddings, learned positional embeddings, single-head self-attention, a feedforward layer, and a final projection back to the vocabulary.

---

## 📌 Objective

Build and run a **minimal decoder-only Transformer block** for demonstration and educational purposes:

* Input: token IDs
* Output: logits over vocabulary
* From-scratch attention and projection
* No `nn.Transformer`, no magic

---

## 💾 Code Overview

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# === Config ===
batch_size = 2
seq_len = 10
vocab_size = 1000
embed_dim = 64
ff_dim = 128

# === Input tokens ===
tokens = torch.randint(0, vocab_size, (batch_size, seq_len))  # [2, 10]

# === Embedding layers ===
token_embed = nn.Embedding(vocab_size, embed_dim)
pos_embed = nn.Embedding(seq_len, embed_dim)

x_token = token_embed(tokens)  # [2, 10, 64]
positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, seq_len)
x_pos = pos_embed(positions)  # [2, 10, 64]

x = x_token + x_pos  # [2, 10, 64]

# === Self-Attention ===
Wq = nn.Linear(embed_dim, embed_dim)
Wk = nn.Linear(embed_dim, embed_dim)
Wv = nn.Linear(embed_dim, embed_dim)

Q = Wq(x)  # [2, 10, 64]
K = Wk(x)
V = Wv(x)

attn_scores = Q @ K.transpose(-2, -1) / (embed_dim ** 0.5)  # [2, 10, 10]
attn_weights = F.softmax(attn_scores, dim=-1)              # [2, 10, 10]
attn_output = attn_weights @ V                             # [2, 10, 64]

# === Feedforward ===
ff1 = nn.Linear(embed_dim, ff_dim)
ff2 = nn.Linear(ff_dim, embed_dim)

ff_output = ff2(F.relu(ff1(attn_output)))  # [2, 10, 64]

# === Final projection to vocab ===
to_vocab = nn.Linear(embed_dim, vocab_size)
logits = to_vocab(ff_output)  # [2, 10, 1000]

print("Logits shape:", logits.shape)
```

---

## 👀 What This Does

| Step                  | Description                                         |
| --------------------- | --------------------------------------------------- |
| Token + Pos Embedding | Converts token IDs to dense vectors + adds position |
| Q, K, V               | Learnable linear projections                        |
| Attention Weights     | Scaled dot product + softmax                        |
| Weighted Sum          | Multiplies V by attention weights                   |
| Feedforward Layer     | Hidden → output transformation                      |
| Final Projection      | Projects to logits over vocab                       |

---

## 🧮 Why This Matters

This baby Transformer captures the **core ideas** of decoder-only models like GPT:

* Contextual token representation
* Self-attention alignment
* Feedforward transformation
* Vocabulary prediction head

Perfect for learning and experimenting.

---

## 🚀 Run It

To execute this code:

```bash
pip install torch
python baby_transformer.py
```

---

## ✅ Output

```python
Logits shape: torch.Size([2, 10, 1000])
```

Each token in each sequence now has a predicted distribution over the vocabulary.

---

## 📖 Next Steps

* Wrap it in an `nn.Module`
* Add residuals and layer norm
* Train on a toy dataset
* Extend to multi-head attention and multiple layers

---


🎓 About

This material is part of the "LLMs Under the Hood" masterclass by Ricardo Calix — a 90-minute session designed for engineers and data scientists who want to deeply understand how Transformers work.

Visit: www.rcalix.com


---

## Book

This is the repo for my new book "100 + Essential Linear Algebra Operations For Deep Learning".


<a href="https://amzn.to/3SlQGHC"><img src="llamaBook.jpeg" alt="image" width="300" height="auto"></a>

## FTC and Amazon Disclaimer: 

This post/page/article includes Amazon Affiliate links to products. This site receives income if you purchase through these links. This income helps support content such as this one. Content may also be supported by Generative AI and Recommender Advertisements. 

## Tensor Operations

* AGS
* colab
* Data: https://github.com/rcalix1/CyberSecurityAndMachineLearning/tree/main/FirstEdition/Ch10_AIassurance/AdversarialML
* Use dataset:  FruitsAdversarialML.zip
* https://github.com/rcalix1/TransferLearning/blob/main/fastai/colab/InClassFastAIcolabMalwarw.ipynb
* https://github.com/rcalix1/DeepLearningAlgorithms/tree/main/SecondEdition/Chapter10_Transformers/GPTs
* 
