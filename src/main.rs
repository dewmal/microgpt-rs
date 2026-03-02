use native_tls::TlsConnector;
use std::{
    cell::RefCell,
    collections::HashSet,
    f64, fs,
    io::{BufRead, BufReader, Read, Write},
    net::TcpStream,
    ops::{Add, Div, Mul, Neg, Sub},
    rc::Rc,
    time::{SystemTime, UNIX_EPOCH},
};

fn main() {
    load_data();
    tokenizer();

    let vocab_size = 27;
    let block_size = 8;
    let n_layer = 1;
    let n_embed = 16;

    let model = Model::new(vocab_size, block_size, n_layer, n_embed);
    println!("Num params: {}", model.params.len());
}

struct Param {
    data: f64, // Weight Value
    grad: f64, // Gradient (Update with backward pass)
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

struct MatrixView {
    rows: usize,
    cols: usize,
    start: usize, // index in Model.params where this matrix begins
}

impl MatrixView {
    fn index(&self, r: usize, c: usize) -> usize {
        self.start + r * self.cols + c
    }
}

struct LayerParams {
    attn_wq: MatrixView,
    attn_wk: MatrixView,
    attn_wv: MatrixView,
    attn_wo: MatrixView,
    mlp_fc1: MatrixView,
    mlp_fc2: MatrixView,
}

struct Model {
    params: Vec<Param>,
    wte: MatrixView,
    wpe: MatrixView,
    lm_head: MatrixView,
    layers: Vec<LayerParams>,
}

impl Model {
    fn new(vocab_size: usize, block_size: usize, n_layer: usize, n_embd: usize) -> Model {
        let mut params: Vec<Param> = Vec::new();
        let std = 0.08;

        let mut rng: u64 = 42;

        let mut alloc = |rows: usize, cols: usize| -> MatrixView {
            let start = params.len();
            for _ in 0..(rows * cols) {
                params.push(Param {
                    data: gaussian(&mut rng, std),
                    grad: 0.0,
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
}

#[derive(Debug, Clone)]
struct ValueRef(Rc<RefCell<Value>>);
impl ValueRef {
    fn borrow(&self) -> std::cell::Ref<'_, Value> {
        self.0.borrow()
    }

    fn borrow_mut(&self) -> std::cell::RefMut<'_, Value> {
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

    pub fn powf(x: &ValueRef, n: f64) -> ValueRef {
        let x_data = x.borrow().data;
        let data = x_data.powf(n);

        // d(x^n)/dx = n * x^(n-1)
        let local = if n == 0.0 {
            0.0 // derivate of constant 1 wrt x is 0
        } else {
            n * x_data.powf(n - 1.0)
        };

        Value::node(data, vec![ValueRef(Rc::clone(&x.0))], vec![local])
    }
}

impl PartialEq for ValueRef {
    fn eq(&self, other: &Self) -> bool {
        self.borrow().data == other.borrow().data
    }
}

impl<'a> Neg for &'a ValueRef {
    type Output = ValueRef;

    fn neg(self) -> ValueRef {
        // -x = x * (-1)
        let minus_one = Value::leaf(-1.0);
        self * &minus_one
    }
}

impl<'a, 'b> Add<&'b ValueRef> for &'a ValueRef {
    type Output = ValueRef;
    fn add(self, other: &'b ValueRef) -> Self::Output {
        let data = self.borrow().data + other.borrow().data;

        Value::node(
            data,
            vec![ValueRef(Rc::clone(&self.0)), ValueRef(Rc::clone(&other.0))],
            vec![1.0, 1.0],
        )
    }
}
impl<'a, 'b> Sub<&'b ValueRef> for &'a ValueRef {
    type Output = ValueRef;
    fn sub(self, other: &'b ValueRef) -> Self::Output {
        self + &(-other)
    }
}

impl<'a, 'b> Mul<&'b ValueRef> for &'a ValueRef {
    type Output = ValueRef;

    fn mul(self, other: &'b ValueRef) -> ValueRef {
        let x = self.borrow().data;
        let y = other.borrow().data;

        let data = x * y;
        // local grads: dz/dx, dy/dx = x
        Value::node(
            data,
            vec![ValueRef(Rc::clone(&self.0)), ValueRef(Rc::clone(&other.0))],
            vec![y, x],
        )
    }
}

impl<'a, 'b> Div<&'b ValueRef> for &'a ValueRef {
    type Output = ValueRef;

    fn div(self, other: &'b ValueRef) -> ValueRef {
        // x/y = x * y(-1)
        let inv = Value::powf(other, -1.0);
        self * &inv
    }
}

//ValueRef + f64
impl<'a> Add<f64> for &'a ValueRef {
    type Output = ValueRef;
    fn add(self, data: f64) -> ValueRef {
        self + &Value::leaf(data)
    }
}

//f64 + ValueRef
impl<'a> Add<&'a ValueRef> for f64 {
    type Output = ValueRef;
    fn add(self, data: &'a ValueRef) -> ValueRef {
        &Value::leaf(self) + data
    }
}
// Valueref - f64
impl<'a> Sub<f64> for &'a ValueRef {
    type Output = ValueRef;

    fn sub(self, other: f64) -> ValueRef {
        self - &Value::leaf(other)
    }
}
//f64 - ValueRef
impl<'a> Sub<&'a ValueRef> for f64 {
    type Output = ValueRef;
    fn sub(self, data: &'a ValueRef) -> ValueRef {
        &Value::leaf(self) - data
    }
}
// Valueref * f64
impl<'a> Mul<f64> for &'a ValueRef {
    type Output = ValueRef;

    fn mul(self, other: f64) -> ValueRef {
        self * &Value::leaf(other)
    }
}
//  f64 * Valueref
impl<'a> Mul<&'a ValueRef> for f64 {
    type Output = ValueRef;

    fn mul(self, other: &'a ValueRef) -> ValueRef {
        &Value::leaf(self) * other
    }
}
// Valueref / f64
impl<'a> Div<f64> for &'a ValueRef {
    type Output = ValueRef;

    fn div(self, other: f64) -> ValueRef {
        self / &Value::leaf(other)
    }
}
//  f64 / Valueref
impl<'a> Div<&'a ValueRef> for f64 {
    type Output = ValueRef;

    fn div(self, other: &'a ValueRef) -> ValueRef {
        &Value::leaf(self) / other
    }
}

impl ValueRef {
    fn id(&self) -> usize {
        Rc::as_ptr(&self.0) as usize
    }

    pub fn backward(&self) {
        // Topological Order
        let mut graph: Vec<ValueRef> = Vec::new();
        let mut visited: HashSet<usize> = HashSet::new();

        fn build_graph(v: &ValueRef, visited: &mut HashSet<usize>, graph: &mut Vec<ValueRef>) {
            let vid = v.id();
            if visited.contains(&vid) {
                return;
            }
            visited.insert(vid);

            //Copy children
            let children = v.borrow().children.clone();
            for child in &children {
                build_graph(child, visited, graph);
            }

            graph.push(ValueRef(Rc::clone(&v.0)));
        }

        build_graph(self, &mut visited, &mut graph);

        // Seed gradient
        self.borrow_mut().grad = 1.0;

        // Propagte grads in reverse graph order
        for v in graph.into_iter().rev() {
            // Copy
            // Drop Borrow before mutating children
            let (v_grad, children, locals) = {
                let vb = v.borrow();
                (vb.grad, vb.children.clone(), vb.local_grads.clone())
            };

            for (child, local_grad) in children.iter().into_iter().zip(locals.into_iter()) {
                child.borrow_mut().grad += local_grad * v_grad
            }
        }
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
#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn value_usage() {
        let x = &Value::leaf(2.0);
        let y = &Value::leaf(3.0);
        assert_eq!(x + y, Value::leaf(5.0));
        assert_eq!(x * y, Value::leaf(6.0));
        assert_eq!(x + 5.0, Value::leaf(7.0));
    }

    // Test subtraction
    #[test]
    fn test_subtraction() {
        let x = &Value::leaf(5.0);
        let y = &Value::leaf(3.0);
        assert_eq!(x - y, Value::leaf(2.0));
        assert_eq!(x - 1.0, Value::leaf(4.0));
    }

    // Test division
    #[test]
    fn test_division() {
        let x = &Value::leaf(6.0);
        let y = &Value::leaf(2.0);
        assert_eq!(x / y, Value::leaf(3.0));
        assert_eq!(x / 3.0, Value::leaf(2.0));
    }

    // Test negation
    #[test]
    fn test_negation() {
        let x = &Value::leaf(4.0);
        assert_eq!(-x, Value::leaf(-4.0));
    }

    // Test power
    #[test]
    fn test_power() {
        let x = &Value::leaf(3.0);
        assert_eq!(Value::powf(x, 2.0), Value::leaf(9.0)); // 3^2 = 9
        assert_eq!(Value::powf(x, 0.0), Value::leaf(1.0)); // 3^0 = 1
        assert_eq!(Value::powf(x, 1.0), Value::leaf(3.0)); // 3^1 = 3
    }

    // Test operations with zero
    #[test]
    fn test_with_zero() {
        let x = &Value::leaf(5.0);
        let zero = &Value::leaf(0.0);
        assert_eq!(x + zero, Value::leaf(5.0)); // x + 0 = x
        assert_eq!(x * zero, Value::leaf(0.0)); // x * 0 = 0
        assert_eq!(x - zero, Value::leaf(5.0)); // x - 0 = x
    }

    // Test f64 on the left side
    #[test]
    fn test_f64_left_side() {
        let x = &Value::leaf(3.0);
        assert_eq!(10.0 + x, Value::leaf(13.0)); // 10 + 3 = 13
        assert_eq!(10.0 - x, Value::leaf(7.0)); // 10 - 3 = 7
        assert_eq!(2.0 * x, Value::leaf(6.0)); // 2 * 3 = 6
        assert_eq!(9.0 / x, Value::leaf(3.0)); // 9 / 3 = 3
    }

    // Test chained operations
    #[test]
    fn test_chained_operations() {
        let x = &Value::leaf(2.0);
        let y = &Value::leaf(3.0);
        let z = &Value::leaf(4.0);

        // (2 + 3) * 4 = 20
        let result = &(x + y) * z;
        assert_eq!(result, Value::leaf(20.0));
    }

    // Test negative numbers
    #[test]
    fn test_negative_numbers() {
        let x = &Value::leaf(-2.0);
        let y = &Value::leaf(-3.0);
        assert_eq!(x + y, Value::leaf(-5.0)); // -2 + -3 = -5
        assert_eq!(x * y, Value::leaf(6.0)); // -2 * -3 = 6
    }

    #[test]
    fn test_backward_mul() {
        let a = Value::leaf(2.0);
        let b = Value::leaf(3.0);

        let c = &a * &b; // 6.0
        let l = &c + &a; // 8.0

        l.backward();

        assert_eq!(a.borrow().grad, 4.0); // 4.0 (dL/da = b + 1 = 3 + 1, via both paths)
        assert_eq!(b.borrow().grad, 2.0); // 2.0 (dL/db = a = 2)
    }
}
