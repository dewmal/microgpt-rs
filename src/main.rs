use crate::train::train;

mod data;
mod gpt;
mod math;
mod model;
mod optim;
mod tokenizer;
mod train;
mod value;

fn main() {
    data::load_data();
    let tokenizer_info = tokenizer::tokenizer();

    let vocab_size = tokenizer_info.vocab_size;
    let block_size = 8;
    let n_layer = 1;
    let n_embed = 16;
    let n_head = 4;
    let num_steps = 1000;

    let mut model = model::Model::new(vocab_size, block_size, n_layer, n_embed);
    println!("Num params: {}", model.params.len());

    train(
        &mut model,
        &gpt::GptConfig {
            vocab_size,
            block_size,
            n_layer,
            n_head,
            n_embed,
        },
        &tokenizer_info.docs,
        &tokenizer_info.uchars,
        tokenizer_info.bos,
        block_size,
        num_steps,
    );
}
