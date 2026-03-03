use crate::value::{Value, ValueRef};
pub(crate) struct Param {
    pub(crate) v: ValueRef, // Gradient (Update with backward pass)
}

pub(crate) struct MatrixView {
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    start: usize, // index in Model.params where this matrix begins
}

impl MatrixView {
    pub(crate) fn index(&self, r: usize, c: usize) -> usize {
        self.start + r * self.cols + c
    }
}

pub(crate) struct LayerParams {
    pub(crate) attn_wq: MatrixView,
    pub(crate) attn_wk: MatrixView,
    pub(crate) attn_wv: MatrixView,
    pub(crate) attn_wo: MatrixView,
    pub(crate) mlp_fc1: MatrixView,
    pub(crate) mlp_fc2: MatrixView,
}

pub(crate) struct Model {
    pub(crate) params: Vec<Param>,
    pub(crate) wte: MatrixView,
    pub(crate) wpe: MatrixView,
    pub(crate) lm_head: MatrixView,
    pub(crate) layers: Vec<LayerParams>,
}

impl Model {
    pub(crate) fn new(
        vocab_size: usize,
        block_size: usize,
        n_layer: usize,
        n_embd: usize,
    ) -> Model {
        let mut params: Vec<Param> = Vec::new();
        let std = 0.08;

        let mut rng: u64 = 42;

        let mut alloc = |rows: usize, cols: usize| -> MatrixView {
            let start = params.len();
            for _ in 0..(rows * cols) {
                params.push(Param {
                    v: Value::leaf(gaussian(&mut rng, std)),
                });
            }
            MatrixView { rows, cols, start }
        };

        // Token and position embeddings
        let wte = alloc(vocab_size, n_embd);
        let wpe = alloc(block_size, n_embd);
        let lm_head = alloc(vocab_size, n_embd);

        // One LayerParams per layer
        let mut layers = Vec::new();
        for _ in 0..n_layer {
            layers.push(LayerParams {
                attn_wq: alloc(n_embd, n_embd),
                attn_wk: alloc(n_embd, n_embd),
                attn_wv: alloc(n_embd, n_embd),
                attn_wo: alloc(n_embd, n_embd),
                mlp_fc1: alloc(4 * n_embd, n_embd),
                mlp_fc2: alloc(n_embd, 4 * n_embd),
            });
        }

        Model {
            params,
            wte,
            wpe,
            lm_head,
            layers,
        }
    }

    pub(crate) fn row(&self, m: &MatrixView, r: usize) -> Vec<ValueRef> {
        let mut out = Vec::with_capacity(m.cols);
        for c in 0..m.cols {
            let idx = m.index(r, c);
            out.push(self.params[idx].v.clone()); // <-- use param node
        }
        out
    }
}
pub(crate) type KV = Vec<Vec<Vec<ValueRef>>>;

pub(crate) fn new_kv(n_layer: usize) -> (KV, KV) {
    let keys = (0..n_layer).map(|_| Vec::new()).collect();
    let values = (0..n_layer).map(|_| Vec::new()).collect();
    (keys, values)
}
fn gaussian(seed: &mut u64, std: f64) -> f64 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    let u1 = (*seed as f64) / (u64::MAX as f64);

    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    let u2 = (*seed as f64) / (u64::MAX as f64);

    // Box-Mullar : convert uniform random to Gaussian
    let u1 = u1.abs().max(1e-10);
    let normal = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();

    normal * std
}
