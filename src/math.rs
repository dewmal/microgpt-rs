use crate::model::{MatrixView, Model};
use crate::value::{Value, ValueRef};

fn linear_vec(x: &[ValueRef], model: &Model, w: &MatrixView) -> Vec<ValueRef> {
    assert_eq!(x.len(), w.cols);
    let mut out = Vec::with_capacity(w.rows);
    for r in 0..w.rows {
        let mut acc = Value::leaf(0.0);
        for c in 0..w.cols {
            let wij = model.params[w.index(r, c)].data;
            let wv = Value::leaf(wij);
            acc = &acc + &(&wv * &x[c]);
        }
        out.push(acc);
    }
    out
}
fn vsum(x: &[ValueRef]) -> ValueRef {
    let mut acc = Value::leaf(0.0);
    for xi in x {
        acc = &acc + xi;
    }
    acc
}
fn vadd(a: &[ValueRef], b: &[ValueRef]) -> Vec<ValueRef> {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| x + y).collect()
}
fn vscale(x: &[ValueRef], s: f64) -> Vec<ValueRef> {
    x.iter().map(|xi| xi * s).collect()
}
fn rmsnorm(x: &[ValueRef]) -> Vec<ValueRef> {
    let n = x.len() as f64;
    let mut sq = Vec::with_capacity(x.len());
    for xi in x {
        sq.push(xi * xi);
    }
    let ms = &vsum(&sq) / n;
    let eps = Value::leaf(1e-5);
    let inv_sqrt = Value::powf(&(&ms + &eps), -0.5);
    x.iter().map(|xi| xi * &inv_sqrt).collect()
}
fn softmax(logits: &[ValueRef]) -> Vec<ValueRef> {
    // take max in data-space (constant)
    let mut maxv = f64::NEG_INFINITY;
    for v in logits {
        maxv = maxv.max(v.borrow().data);
    }
    let max_leaf = Value::leaf(maxv);

    let mut exps = Vec::with_capacity(logits.len());
    for l in logits {
        exps.push(Value::exp(&(l - &max_leaf)));
    }
    let denom = vsum(&exps);
    exps.into_iter().map(|e| &e / &denom).collect()
}
