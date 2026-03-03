use crate::{
    gpt::{GptConfig, gpt_step},
    math::{nll_from_logits, vsum},
    model::{Model, new_kv},
    optim::Adam,
    tokenizer::tokenize_doc,
    value::ValueRef,
};

pub fn train(
    model: &mut Model,
    cfg: &GptConfig,
    docs: &[String],
    uchars: &[char],
    bos: usize,
    block_size: usize,
    num_steps: usize,
) {
    let mut opt = Adam::new(model.params.len(), 0.01, 0.85, 0.99, 1e8, num_steps);

    for step in 0..num_steps {
        let doc = &docs[step % docs.len()];
        let tokens = tokenize_doc(doc, uchars, bos);
        let n = block_size.min(tokens.len().saturating_sub(1));

        // fresh KV per sequence
        let (mut keys, mut values) = new_kv(cfg.n_layer);

        // forward, build graph
        let mut losses: Vec<ValueRef> = Vec::with_capacity(n);
        for pos_id in 0..n {
            let token_id = tokens[pos_id];
            let target_id = tokens[pos_id + 1];

            let logits = gpt_step(model, cfg, token_id, pos_id, &mut keys, &mut values);
            let loss_t = nll_from_logits(&logits, target_id);
            losses.push(loss_t);
        }

        //mean loss
        let total = vsum(&losses);
        let loss = &total / (n as f64);

        // backward
        loss.backward();

        // adam update
        opt.step(&mut model.params);

        println!(
            "step {:4} / {:4} | loss {:4}",
            step + 1,
            num_steps,
            loss.borrow().data
        )
    }
}
