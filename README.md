# microgpt-rs

A tiny **GPT-style model in Rust** (training + inference), inspired by Andrej Karpathy’s **microgpt**.

- Original blog post (Feb 12, 2026): https://karpathy.github.io/2026/02/12/microgpt/
- Web version: https://karpathy.ai/microgpt.html

## What this is

This repo mirrors the “algorithmic essence” of microgpt, but in Rust:

- **Dataset**: downloads Karpathy’s `names.txt` and saves it as `data.txt`
- **Tokenizer**: character-level vocab + `BOS` token (begin/end of sequence)
- **Autograd**: custom `Value` type with a computation graph + backprop
- **Model**: GPT-style forward step (token+pos embedding → attention/MLP blocks → logits)
- **Optimizer**: Adam
- **Training loop**: next-token prediction with NLL / cross-entropy
- **Inference**: autoregressive sampling with temperature

## Project layout

```text
src/
├── main.rs        # runs training then prints sampled names
├── data.rs        # downloads dataset and writes data.txt
├── tokenizer.rs   # builds vocab + tokenizes docs
├── value.rs       # autograd engine (Value + backward)
├── math.rs        # helpers (loss, reductions, etc.)
├── model.rs       # parameter storage + KV cache helpers
├── gpt.rs         # gpt_step (one-token forward pass)
├── optim.rs       # Adam optimizer
├── train.rs       # training loop
└── inference.rs   # sampling / softmax / weighted choice
````

## Run

> No CLI flags currently: `main.rs` trains first, then generates samples.

```bash
cargo run --release
```

What you’ll see:

* dataset download → `data.txt`
* training loss for `num_steps` (default: 1000)
* ~20 generated names (sampling with temperature)

Replace your **Benchmark** section with this shorter version:

## Benchmark

Measured total runtime:

* **Rust**: `25,385,296,100 ns`
* **Python**: `164,041,600,500 ns`

Rust runs **~6.5× faster**.

**Why?**

* Compiled native code vs interpreted Python
* Static typing (no dynamic object overhead)
* No GC / lower memory overhead
* Aggressive compiler optimizations

The algorithm is identical — the speedup comes purely from language/runtime efficiency.

## Notes

* Hyperparameters are currently set in `main.rs` (e.g. `block_size`, `n_layer`, `n_embed`, `n_head`, `num_steps`).
* The dataset fetch uses TLS (`native_tls`) and performs a direct HTTPS GET, then writes the response to `data.txt`.

## License

Apache-2.0
