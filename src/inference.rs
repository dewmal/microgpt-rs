use std::f64;

use crate::{
    gpt::{GptConfig, gpt_step},
    model::{Model, new_kv},
};

fn xorshift64(seed: &mut u64) -> u64 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    *seed
}

fn rand_f64(seed: &mut u64) -> f64 {
    (xorshift64(seed) as f64) / (u64::MAX as f64)
}

pub(crate) fn weighted_choice(seed: &mut u64, weights: &[f64]) -> usize {
    let total: f64 = weights.iter().sum();
    debug_assert!(total > 0.0);

    let mut r = rand_f64(seed) * total;
    for (i, &w) in weights.iter().enumerate() {
        r -= w;
        if r <= 0.0 {
            return i;
        }
    }
    weights.len() - 1
}

pub(crate) fn softmax_f64(logits: &[f64]) -> Vec<f64> {
    let mut maxv = f64::NEG_INFINITY;
    for &x in logits {
        maxv = maxv.max(x);
    }
    let mut exps = Vec::with_capacity(logits.len());
    let mut sum = 0.0;
    for &x in logits {
        let e = (x - maxv).exp();
        exps.push(e);
        sum += e;
    }
    for e in &mut exps {
        *e /= sum;
    }
    exps
}

pub(crate) fn sample_name(
    model: &Model,
    cfg: &GptConfig,
    uchars: &[char],
    bos: usize,
    temprature: f64,
    seed: &mut u64,
) -> String {
    let (mut keys, mut values) = new_kv(cfg.n_layer);

    let mut token_id = bos;
    let mut out: Vec<char> = Vec::new();

    for pos_id in 0..cfg.block_size {
        let logits_v = gpt_step(model, cfg, token_id, pos_id, &mut keys, &mut values);

        //
        let mut logits: Vec<f64> = Vec::with_capacity(cfg.vocab_size);
        for l in &logits_v {
            logits.push(l.borrow().data / temprature);
        }
        let probs = softmax_f64(&logits);

        token_id = weighted_choice(seed, &probs);

        if token_id == bos {
            break;
        }
        out.push(uchars[token_id]);
    }
    out.into_iter().collect()
}
