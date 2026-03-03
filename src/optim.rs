use crate::model::Param;

pub(crate) struct Adam {
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    m: Vec<f64>,
    v: Vec<f64>,
    t: usize,
    num_steps: usize,
}

impl Adam {
    pub(crate) fn new(
        param_count: usize,
        lr: f64,
        beta1: f64,
        beta2: f64,
        eps: f64,
        num_steps: usize,
    ) -> Self {
        Self {
            lr,
            beta1,
            beta2,
            eps,
            m: vec![0.0; param_count],
            v: vec![0.0; param_count],
            t: 0,
            num_steps,
        }
    }

    pub(crate) fn step(&mut self, params: &mut [Param]) {
        self.t += 1;

        //LR Decay
        let frac = (self.t as f64) / (self.num_steps as f64);
        let lr_t = self.lr * (1.0 - frac).max(0.0);

        let b1 = self.beta1;
        let b2 = self.beta2;

        // bias correction denominators
        let b1t = 1.0 - b1.powi(self.t as i32);
        let b2t = 1.0 - b2.powi(self.t as i32);

        for (i, p) in params.iter_mut().enumerate() {
            let g = p.v.borrow().grad;

            // moment
            self.m[i] = b1 * self.m[i] + (1.0 - b1) * g;
            self.v[i] = b2 * self.v[i] + (1.0 - b2) * g * g;

            // bias correct
            let m_hat = self.m[i] / b1t;
            let v_hat = self.v[i] / b2t;
            {
                let mut pb = p.v.borrow_mut();
                pb.data -= lr_t * m_hat / (v_hat.sqrt() + self.eps);
                pb.grad = 0.0;
            }
        }
    }
}
