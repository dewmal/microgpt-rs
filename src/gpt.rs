use crate::{
    math::{linear_vec, rmsnorm, softmax, vadd},
    model::{KV, Model},
    value::{Value, ValueRef},
};

pub(crate) struct GptConfig {
    pub(crate) vocab_size: usize,
    pub(crate) block_size: usize,
    pub(crate) n_layer: usize,
    pub(crate) n_head: usize,
    pub(crate) n_embed: usize,
}

pub(crate) fn gpt_step(
    model: &Model,
    cfg: &GptConfig,
    token_id: usize,
    pos_id: usize,
    keys: &mut KV,
    values: &mut KV,
) -> Vec<ValueRef> {
    let head_dim = cfg.n_embed / cfg.n_head;

    // embeddings
    let tok_emb = model.row(&model.wte, token_id);
    let pos_emb = model.row(&model.wpe, pos_id);
    let mut x = vadd(&tok_emb, &pos_emb);
    x = rmsnorm(&x);

    for li in 0..cfg.n_layer {
        // Attention
        let x_res = x.clone();
        x = rmsnorm(&x);

        let lp = &model.layers[li];
        let q = linear_vec(&x, model, &lp.attn_wq);
        let k = linear_vec(&x, model, &lp.attn_wk);
        let v = linear_vec(&x, model, &lp.attn_wv);

        keys[li].push(k);
        values[li].push(v);

        let mut x_attn: Vec<ValueRef> = Vec::with_capacity(cfg.n_embed);

        for h in 0..cfg.n_head {
            let hs = h * head_dim;

            let q_h = &q[hs..hs + head_dim];
            let k_cache = &keys[li];
            let v_cache = &values[li];

            // attn_logits[t] = dot(q_h,k_h[t])/sqrt(head_dim)
            let mut attn_logits: Vec<ValueRef> = Vec::with_capacity(k_cache.len());
            let scale = 1.0 / (head_dim as f64).sqrt();

            for t in 0..k_cache.len() {
                let k_t = &k_cache[t][hs..hs + head_dim];
                let mut dot = Value::leaf(0.0);
                for j in 0..head_dim {
                    dot = &dot + &(&q_h[j] * &k_t[j]);
                }
                attn_logits.push(&dot * scale);
            }

            let attn_w = softmax(&attn_logits);

            // head_out[j]= sum_t_attn_w[t] * v_h[t][j]
            let mut head_out: Vec<ValueRef> = Vec::with_capacity(head_dim);
            for j in 0..head_dim {
                let mut acc = Value::leaf(0.0);
                for t in 0..v_cache.len() {
                    let v_t = &v_cache[t][hs + j];
                    acc = &acc + &(&attn_w[t] * v_t);
                }
                head_out.push(acc);
            }
            x_attn.extend(head_out);
        }

        let x_proj = linear_vec(&x_attn, model, &lp.attn_wo);
        x = vadd(&x_proj, &x_res);

        let x_res = x.clone();
        x = rmsnorm(&x);

        let h = linear_vec(&x, model, &lp.mlp_fc1);
        let h = h.iter().map(|xi| Value::relu(xi)).collect::<Vec<_>>();
        let h2 = linear_vec(&h, model, &lp.mlp_fc2);
        x = vadd(&h2, &x_res);
    }

    linear_vec(&x, model, &model.lm_head)
}
