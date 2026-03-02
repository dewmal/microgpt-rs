use native_tls::TlsConnector;
use std::{
    f64, fs,
    io::{BufRead, BufReader, Read, Write},
    net::TcpStream,
    time::{SystemTime, UNIX_EPOCH},
};

use std::cell::RefCell;
use std::ops::Add;
use std::rc::Rc;

fn main() {
    load_data();
    tokenizer();
}

#[derive(Debug)]
struct ValueRef(Rc<RefCell<Value>>);
impl ValueRef {
    fn borrow(&self) -> std::cell::Ref<Value> {
        self.0.borrow()
    }

    fn borrow_mut(&self) -> std::cell::RefMut<Value> {
        self.0.borrow_mut()
    }
}
#[derive(Debug)]
struct Value {
    // scalar vlaue
    data: f64,
    // dL/d (this)
    grad: f64,
    //Graph Structure
    children: Vec<ValueRef>,
    // local derivative of this node w.r.t each child
    local_grads: Vec<f64>,
}

impl Value {
    // Create a leaf node (no children)
    fn leaf(data: f64) -> ValueRef {
        ValueRef(Rc::new(RefCell::new(Value {
            data,
            grad: 0.0,
            children: vec![],
            local_grads: vec![],
        })))
    }

    // Create a node with given children and local grads
    fn node(data: f64, children: Vec<ValueRef>, local_grads: Vec<f64>) -> ValueRef {
        debug_assert_eq!(children.len(), local_grads.len());
        ValueRef(Rc::new(RefCell::new(Value {
            data,
            grad: 0.0,
            children,
            local_grads,
        })))
    }
}

impl Add for ValueRef {
    type Output = ValueRef;
    fn add(self, other: ValueRef) -> Self::Output {
        let data = self.borrow().data + other.borrow().data;

        Value::node(data, vec![self, other], vec![1.0, 1.0])
    }
}

fn tokenizer() {
    let docs = preprocess_data();
    let bos = docs.len();
    let vocab_size = docs.len() + 1;
    println!("BOS:{bos},vocab size: {vocab_size}")
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

fn load_data() {
    let url = "https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt";
    let file_name = "data.txt";
    let without_scheme = url.replace("https://", "");

    let (host, path) = match without_scheme.find("/") {
        Some(i) => (&without_scheme[..i], &without_scheme[i..]),
        None => (without_scheme.as_str(), "/"),
    };
    let tcp = TcpStream::connect(format!("{host}:443")).expect("Could not Connect to the server");

    let connector = TlsConnector::new().expect("Failed to create TLS connector");
    let mut stream = connector.connect(host, tcp).expect("TLS handshake failed");

    let request = format!("GET {path} HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\n\r\n");

    stream
        .write_all(request.as_bytes())
        .expect("Failed to send");

    let mut reader = BufReader::new(stream);

    let mut line = String::new();

    loop {
        line.clear();
        reader.read_line(&mut line).expect("Failed to read");
        if line == "\r\n" || line.is_empty() {
            break;
        }
    }
    let mut body = String::new();
    reader
        .read_to_string(&mut body)
        .expect("Failed to read body");

    save_to_file(file_name, &mut body);
}

fn save_to_file(file_name: &str, contents: &str) {
    std::fs::write(file_name, contents).expect("Could not save the file");
    println!("File saved as: {file_name}")
}
