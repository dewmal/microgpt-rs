use std::{
    cell::RefCell,
    collections::HashSet,
    f64,
    ops::{Add, Div, Mul, Neg, Sub},
    rc::Rc,
};

#[derive(Debug, Clone)]
pub struct ValueRef(Rc<RefCell<Value>>);
impl ValueRef {
    pub(crate) fn borrow(&self) -> std::cell::Ref<'_, Value> {
        self.0.borrow()
    }

    pub(crate) fn borrow_mut(&self) -> std::cell::RefMut<'_, Value> {
        self.0.borrow_mut()
    }
}
#[derive(Debug)]
pub struct Value {
    // scalar vlaue
    pub(crate) data: f64,
    // dL/d (this)
    pub(crate) grad: f64,
    //Graph Structure
    children: Vec<ValueRef>,
    // local derivative of this node w.r.t each child
    local_grads: Vec<f64>,
}

impl Value {
    // Create a leaf node (no children)
    pub(crate) fn leaf(data: f64) -> ValueRef {
        ValueRef(Rc::new(RefCell::new(Value {
            data,
            grad: 0.0,
            children: vec![],
            local_grads: vec![],
        })))
    }

    // Create a node with given children and local grads
    pub(crate) fn node(data: f64, children: Vec<ValueRef>, local_grads: Vec<f64>) -> ValueRef {
        debug_assert_eq!(children.len(), local_grads.len());
        ValueRef(Rc::new(RefCell::new(Value {
            data,
            grad: 0.0,
            children,
            local_grads,
        })))
    }
    pub(crate) fn powf(x: &ValueRef, n: f64) -> ValueRef {
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

    pub(crate) fn exp(x: &ValueRef) -> ValueRef {
        let x_data = x.borrow().data;
        let data = x_data.exp();
        // d(exp(x))/dx = exp(x)
        Value::node(data, vec![ValueRef(Rc::clone(&x.0))], vec![data])
    }

    pub(crate) fn relu(x: &ValueRef) -> ValueRef {
        let x_data = x.borrow().data;
        let data = if x_data > 0.0 { x_data } else { 0.0 };
        // d(relu)/dx = 1 if x>0 else 0
        let local = if x_data > 0.0 { 1.0 } else { 0.0 };
        Value::node(data, vec![ValueRef(Rc::clone(&x.0))], vec![local])
    }

    pub(crate) fn ln(x: &ValueRef) -> ValueRef {
        let xd = x.borrow().data;
        let data = xd.ln();
        Value::node(data, vec![ValueRef(Rc::clone(&x.0))], vec![1.0 / xd])
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
