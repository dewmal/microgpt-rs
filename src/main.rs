use crate::{inference::sample_name, train::train};
use std::time::Instant;
mod data;
mod gpt;
mod inference;
mod math;
mod model;
mod optim;
mod tokenizer;
mod train;
mod value;

fn main() {
    let start = Instant::now();
    data::load_data();
    let tokenizer_info = tokenizer::tokenizer();

    let vocab_size = tokenizer_info.vocab_size;
    let block_size = 16;
    let n_layer = 1;
    let n_embed = 16;
    let n_head = 4;
    let num_steps = 1000;

    let mut model = model::Model::new(vocab_size, block_size, n_layer, n_embed);
    println!("Num params: {}", model.params.len());
    let cfg = &gpt::GptConfig {
        vocab_size,
        block_size,
        n_layer,
        n_head,
        n_embed,
    };
    train(
        &mut model,
        cfg,
        &tokenizer_info.docs,
        &tokenizer_info.uchars,
        tokenizer_info.bos,
        block_size,
        num_steps,
    );

    let temprature = 0.5;
    println!("inference");
    let mut seed: u64 = 456454;
    for i in 0..20 {
        let name = sample_name(
            &model,
            cfg,
            &tokenizer_info.uchars,
            tokenizer_info.bos,
            temprature,
            &mut seed,
        );
        println!("Sample {:2}: {}", i + 1, name);
    }
    let duration = start.elapsed();
    println!("Duration: {} ns", duration.as_nanos());
}
