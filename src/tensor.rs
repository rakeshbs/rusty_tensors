use ndarray::{Array2, IxDyn};
use rand::Rng;
use std::cell::RefCell;
use std::rc::Rc;

pub type TensorRef = Rc<Tensor>;
type Float = f32;
pub type ArrayT = Array2<f32>;
pub type ArrayTRef = Rc<RefCell<ArrayT>>;

pub struct Tensor {
    data: ArrayTRef,
    grad: ArrayTRef,
    requires_grad: bool,
    backward_fn: Option<Box<dyn Fn(ArrayTRef)>>,
}

impl Tensor {
    pub fn new(
        _data: ArrayT,
        requires_grad: bool,
        backward_fn: impl Fn(ArrayTRef) + 'static,
    ) -> TensorRef {
        let grad = Rc::new(RefCell::new(ArrayT::zeros(_data.dim())));
        let data = Rc::new(RefCell::new(_data));
        Rc::new(Tensor {
            data,
            grad,
            requires_grad,
            backward_fn: Some(Box::new(backward_fn)),
        })
    }

    pub fn backward(&self) {
        let grad = Rc::new(RefCell::new(ArrayT::ones(self.data.borrow().dim())));
        self.backward_fn.as_ref().unwrap()(grad);
    }

    pub fn output(&self) -> ArrayT {
        self.data.borrow().clone()
    }

    pub fn step(&self, lr: Float) {
        if self.requires_grad {
            let mut data = self.data.borrow_mut();
            let grad = self.grad.borrow();
            *data = &*data - lr * &*grad;
        }
    }

    pub fn zero_grad(&self) {
        let mut grad = self.grad.borrow_mut();
        *grad = ArrayT::zeros(grad.dim());
    }
}

fn create_random_array2(rows: usize, cols: usize) -> ArrayT {
    let mut rng = rand::thread_rng();
    ArrayT::from_shape_fn((rows, cols), |_| rng.gen::<f32>())
}

pub fn tensor_rand(rows: usize, cols: usize) -> TensorRef {
    Tensor::new(create_random_array2(rows, cols), true, |_: ArrayTRef| {})
}

pub fn tensor(data: ArrayT, requires_grad: bool) -> TensorRef {
    let backward_fn = |_: ArrayTRef| {};
    Tensor::new(data, requires_grad, backward_fn)
}

pub fn add(left: &TensorRef, right: &TensorRef) -> TensorRef {
    let func = |left: &TensorRef, right: &TensorRef| {
        let (left, right) = (left.clone(), right.clone());
        move |grad: ArrayTRef| {
            {
                let grad = &*grad.borrow();
                let mut l = left.grad.borrow_mut();
                let mut r = right.grad.borrow_mut();
                *l = &*l + &*grad;
                *r = &*r + &*grad;
            }
            left.backward_fn.as_ref().unwrap()(left.grad.clone());
            right.backward_fn.as_ref().unwrap()(right.grad.clone());
        }
    };
    let data = &*left.data.borrow() + &*right.data.borrow();
    Tensor::new(data, true, func(left, right))
}

pub fn mul(left: &TensorRef, right: &TensorRef) -> TensorRef {
    let func = |left: &TensorRef, right: &TensorRef| {
        let (left, right) = (left.clone(), right.clone());
        move |grad: ArrayTRef| {
            {
                let grad = &*grad.borrow();
                let mut l = left.grad.borrow_mut();
                let mut r = right.grad.borrow_mut();
                let d_l = left.data.borrow();
                let d_r = right.data.borrow();
                *l = &*l + d_r.dot(&grad.t());
                *r = &*r + d_l.dot(grad);
            }
            left.backward_fn.as_ref().unwrap()(left.grad.clone());
            right.backward_fn.as_ref().unwrap()(right.grad.clone());
        }
    };
    let a = &*left.data.borrow();
    let b = &*right.data.borrow();
    let data = a.t().dot(b);
    Tensor::new(data, true, func(left, right))
}

pub fn pow(left: &TensorRef, right: Float) -> TensorRef {
    let func = |left: &TensorRef, right: Float| {
        let left = left.clone();
        move |grad: ArrayTRef| {
            {
                let grad = &*grad.borrow();
                let mut l = left.grad.borrow_mut();
                *l = &*l + &(&*left.data.borrow() * &*grad * right);
            }
            left.backward_fn.as_ref().unwrap()(left.grad.clone());
        }
    };
    let data = left.data.borrow().mapv(|x| x.powf(right));
    Tensor::new(data, true, func(left, right))
}

pub fn sub(left: &TensorRef, right: &TensorRef) -> TensorRef {
    let func = |left: &TensorRef, right: &TensorRef| {
        let (left, right) = (left.clone(), right.clone());
        move |grad: ArrayTRef| {
            {
                let grad = &*grad.borrow();
                let mut l = left.grad.borrow_mut();
                let mut r = right.grad.borrow_mut();
                *l = &*l + &*grad;
                *r = &*r - &*grad;
            }
            left.backward_fn.as_ref().unwrap()(left.grad.clone());
            right.backward_fn.as_ref().unwrap()(right.grad.clone());
        }
    };
    let data = &*left.data.borrow() - &*right.data.borrow();
    Tensor::new(data, true, func(left, right))
}

pub fn neg(left: &TensorRef) -> TensorRef {
    let func = |left: &TensorRef| {
        let left = left.clone();
        move |grad: ArrayTRef| {
            {
                let grad = &*grad.borrow();
                let mut l = left.grad.borrow_mut();
                *l = &*l - &*grad;
            }
            left.backward_fn.as_ref().unwrap()(left.grad.clone());
        }
    };
    let data = -&*left.data.borrow();
    Tensor::new(data, true, func(left))
}

pub fn sum(left: &TensorRef) -> TensorRef {
    let func = |left: &TensorRef| {
        let left = left.clone();
        move |grad: ArrayTRef| {
            {
                let grad = &*grad.borrow();
                let mut l = left.grad.borrow_mut();
                *l = &*l + &*grad;
            }
            left.backward_fn.as_ref().unwrap()(left.grad.clone());
        }
    };
    let data = left.data.borrow();
    let summed = data.sum_axis(ndarray::Axis(0));
    let len = summed.len();
    let reshaped = summed.into_shape((len, 1)).unwrap();
    Tensor::new(reshaped, true, func(left))
}

pub fn relu(left: &TensorRef) -> TensorRef {
    let func = |left: &TensorRef| {
        let left = left.clone();
        move |grad: ArrayTRef| {
            {
                let grad = &*grad.borrow();
                let mut l = left.grad.borrow_mut();
                *l = &*l + &(&*grad * left.data.borrow().mapv(|x| if x > 0. { 1. } else { 0. }));
            }
            left.backward_fn.as_ref().unwrap()(left.grad.clone());
        }
    };
    let data = left.data.borrow().mapv(|x| if x > 0. { x } else { 0. });
    Tensor::new(data, true, func(left))
}

pub fn leaky_relu(left: &TensorRef) -> TensorRef {
    let func = |left: &TensorRef| {
        let left = left.clone();
        move |grad: ArrayTRef| {
            {
                let grad = &*grad.borrow();
                let mut l = left.grad.borrow_mut();
                *l = &*l + &(&*grad * left.data.borrow().mapv(|x| if x > 0. { 1. } else { 0.01 }));
            }
            left.backward_fn.as_ref().unwrap()(left.grad.clone());
        }
    };
    let data = left
        .data
        .borrow()
        .mapv(|x| if x > 0. { x } else { 0.01 * x });
    Tensor::new(data, true, func(left))
}
