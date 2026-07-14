# paperback-gpt-dpo

A from-scratch GPT-style transformer — RoPE, flash attention, mixed precision training — trained on public-domain novels and aligned with a from-scratch implementation of Direct Preference Optimization (DPO).

Built as a hands-on deep dive into modern LM training and alignment.

---

## What's in here

- **A GPT-style decoder-only transformer**, implemented from scratch in PyTorch:
  - Rotary positional embeddings (RoPE), implemented from the ground up
  - `torch.nn.functional.scaled_dot_product_attention` (flash attention) for the attention core
  - Mixed-precision training (`torch.amp`) with gradient scaling
  - Gradient clipping, weight tying between the embedding and output layers
  - Cosine LR schedule with warmup
- **Direct Preference Optimization (DPO)**, implemented from scratch:
  - A frozen reference model (`ref_model`) snapshotted before alignment
  - Preference pairs built by sampling multiple completions per prompt and scoring them by log-likelihood
  - The DPO loss (Bradley-Terry-based, per [Rafailov et al., 2023](https://arxiv.org/abs/2305.18290)), trained without any external RL library
- **A base-vs-DPO comparison harness** — generates completions from the pretrained and DPO-tuned checkpoints on the same held-out prompts, side by side.

## Dataset

Trained on a handful of public-domain novels, pulled directly from [Project Gutenberg](https://www.gutenberg.org/):

- *Pride and Prejudice* — Jane Austen
- *Alice's Adventures in Wonderland* — Lewis Carroll
- *Frankenstein* — Mary Shelley
- *Moby-Dick* — Herman Melville
- *A Tale of Two Cities* — Charles Dickens

Tokenized with the GPT-2 BPE tokenizer (`tiktoken`, `gpt2` encoding) — chosen deliberately over a character-level vocabulary, since the tokenizer's merge rules are built around properly-cased, punctuated English text like this.

## Results: base vs. DPO-tuned completions

Generations from the same pretrained checkpoint before and after DPO, on held-out prompts not used to build the preference pairs.

> _Table below — fill in after the final training run._

| Prompt | Base (pretrained) | DPO-tuned | Base score | DPO score |
|---|---|---|---|---|
| _sometimes life really is just_ | | | | |
| _i dont see the point in_ | | | | |
| _the thing about growing up is_ | | | | |
| _it is a truth universally acknowledged that_ | | | | |
| _curiouser and curiouser said_ | | | | |
| _it was the best of whales, it was the_ | | | | |
| _call me ishmael but honestly_ | | | | |

**Score** is the average negative log-likelihood the *generating* model assigns to its own continuation (lower = the model was more confident in its own output). It's a rough sanity signal, not a preference-quality metric on its own — read the completions themselves for the real comparison.

## Architecture

| | |
|---|---|
| Embedding dim | 256 |
| Layers | 8 |
| Attention heads | 8 |
| Context length | 128 |
| Positional encoding | RoPE |
| Vocab size | 50,257 (GPT-2 BPE) |
| Parameters | ~[fill in] |

## Training

**Pretraining:**
- Cosine LR schedule, warmup over 200 steps, peak LR 3e-4, min LR 3e-5
- Mixed precision (fp16) with gradient scaling, gradient clipping at norm 2.0
- Checkpointed to Google Drive to survive Colab disconnects

**DPO:**
- Preference pairs built by sampling 2 completions per prompt from the frozen reference model, scored by average log-probability, keeping (best, worst) pairs
- Trained with `beta = 0.1`, following the standard DPO loss:

```
loss = -logsigmoid(beta * [(chosen_logratio) - (rejected_logratio)])
```

where each logratio is `policy_logprob - reference_logprob` for the chosen/rejected completion.

## Running it

This project was built and run in Google Colab (GPU runtime). To reproduce:

1. Mount Google Drive and set a checkpoint directory.
2. Run the dataset-loading cell (pulls the novels directly from Project Gutenberg — no external dataset library required).
3. Run pretraining. Set a unique `experiment_name` per run — checkpoints are keyed by this name, so changing the dataset or hyperparameters without updating it will silently reload a stale checkpoint instead of retraining.
4. Run the DPO section — builds preference pairs from the pretrained model, then trains the policy against the DPO loss.
5. Run the comparison cell to generate the base-vs-DPO table above.

## What I learned building this

- Implementing RoPE and flash-attention-based causal self-attention from scratch, and running a controlled ablation against learned positional embeddings.
- Building the full DPO pipeline by hand: the Bradley-Terry preference model, why a frozen reference model is needed, and how causal masking lets you compute sequence log-probabilities in a single forward pass instead of token-by-token.
- Diagnosing and fixing overfitting from a small training corpus by watching train/val loss divergence, rather than just trusting a low training loss.
- Practical Colab/GPU debugging: RAM-safe dataset loading, CUDA OOM recovery after interrupted cells, and checkpointing strategies for a flaky free-tier runtime.
