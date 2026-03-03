use std::{
    collections::BTreeSet,
    fs,
    time::{SystemTime, UNIX_EPOCH},
};

pub(crate) struct TokenizerInfo {
    pub(crate) docs: Vec<String>,
    pub(crate) uchars: Vec<char>,
    pub(crate) bos: usize,
    pub(crate) vocab_size: usize,
}

pub(crate) fn tokenizer() -> TokenizerInfo {
    let docs = preprocess_data();

    let mut set = BTreeSet::new();
    for d in &docs {
        for ch in d.chars() {
            set.insert(ch);
        }
    }
    let uchars: Vec<char> = set.into_iter().collect();
    let bos = uchars.len();
    let vocab_size = uchars.len() + 1;
    println!("BOS:{bos},vocab size: {vocab_size}");
    TokenizerInfo {
        docs,
        uchars,
        bos,
        vocab_size,
    }
}

pub(crate) fn tokenize_doc(doc: &str, uchars: &[char], bos: usize) -> Vec<usize> {
    let mut tokens = Vec::with_capacity(doc.len() + 2);
    // Add BOS at beginning
    tokens.push(bos);

    for ch in doc.chars() {
        let tid = uchars
            .iter()
            .position(|&c| c == ch)
            .expect("character not in vocabulary");
        tokens.push(tid);
    }
    tokens.push(bos);
    tokens
}

fn preprocess_data() -> Vec<String> {
    let data_path = "data.txt";
    let contents = fs::read_to_string(data_path).expect("Cannot read data file");
    let mut docs: Vec<String> = contents
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .map(|l| l.to_string())
        .collect();
    let mut seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64;

    // Random Algoritm
    for i in (1..docs.len()).rev() {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;

        let j = seed as usize % (i + 1);
        docs.swap(i, j);
    }

    println!("num docs: {}", docs.len());

    docs
}
