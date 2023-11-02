use ndarray::{ArrayD, IxDyn};
use std::cell::RefCell;
use std::rc::Rc;

type TensorRef = Rc<Tensor>;
type Float = f32;
type ArrayT = ArrayD<Float>;
type ArrayTRef = Rc<RefCell<ArrayD<Float>>>;

pub struct Tensor {
    data: ArrayTRef,
    grad: ArrayTRef,
    requires_grad: bool,
    backward_fn: Option<Box<dyn Fn(ArrayTRef)>>,
}

impl Tensor {
    pub fn new(
        _data: ArrayD<f32>,
        requires_grad: bool,
        backward_fn: impl Fn(ArrayTRef) + 'static,
    ) -> TensorRef {
        let data = Rc::new(RefCell::new(_data));
        let grad = Rc::new(RefCell::new(ArrayD::zeros(data.borrow().shape())));
        Rc::new(Tensor {
            data,
            grad,
            requires_grad,
            backward_fn: Some(Box::new(backward_fn)),
        })
    }

    pub fn backward(&self) {
        self.backward_fn.as_ref().unwrap()(self.grad.clone());
    }

    pub fn output(&self) -> ArrayT {
        self.data.borrow().clone()
    }
}

pub fn tensor(data: &[Float], requires_grad: bool) -> TensorRef {
    let backward_fn = |_: ArrayTRef| {};
    let data = ArrayD::from_shape_vec(IxDyn(&[data.len()]), data.to_vec()).unwrap();
    Tensor::new(data, requires_grad, backward_fn)
}

pub fn add(left: &TensorRef, right: &TensorRef) -> TensorRef {
    let func = |left: &TensorRef, right: &TensorRef| {
        let (left, right) = (left.clone(), right.clone());
        move |grad: ArrayTRef| {
            let grad = &*grad.borrow();
            let mut l = &*left.grad.borrow_mut();
            let mut r = &*right.grad.borrow_mut();
            l = &(l + grad);
            r = &(r + grad);
        }
    };
    let data = &*left.data.borrow() + &*right.data.borrow();
    Tensor::new(data, true, func(left, right))
}

pub fn mul(left: &TensorRef, right: &TensorRef) -> TensorRef {
    let func = |left: &TensorRef, right: &TensorRef| {
        let (left, right) = (left.clone(), right.clone());
        move |grad: ArrayTRef| {
            let grad = &*grad.borrow();
            let mut l = &*left.grad.borrow_mut();
            let mut r = &*right.grad.borrow_mut();
            l = &(l + &(grad * &*right.data.borrow()));
            r = &(r + &(grad * &*left.data.borrow()));
        }
    };
    let data = &*left.data.borrow() * &*right.data.borrow();
    Tensor::new(data, true, func(left, right))
}

//
// // Overload Add for Tensor
// impl Add for Tensor {
//     type Output = Self;
//
//     fn add(self, other: Self) -> Self {
//         let result_data = &self.data + &other.data;
//         let mut result_tensor = Tensor::new(result_data);
//
//         // Backward function for add
//         let backward_fn = move |grad: &Tensor| {
//             self.grad.as_ref().unwrap().data += &grad.data;
//             other.grad.as_ref().unwrap().data += &grad.data;
//         };
//
//         result_tensor.set_backward_fn(backward_fn);
//         result_tensor
//     }
// }
//
// // Overload Mul for Tensor
// impl Mul for Tensor {
//     type Output = Self;
//
//     fn mul(self, other: Self) -> Self {
//         let result_data = &self.data * &other.data;
//         let mut result_tensor = Tensor::new(result_data);
//
//         // Backward function for multiply
//         let backward_fn = move |grad: &Tensor| {
//             self.grad.as_ref().unwrap().data += &(&other.data * &grad.data);
//             other.grad.as_ref().unwrap().data += &(&self.data * &grad.data);
//         };
//
//         result_tensor.set_backward_fn(backward_fn);
//         result_tensor
//     }
// }
//
// fn main() {
//     let data_x = ArrayD::from_elem(IxDyn(&[2, 2]), 2.0);
//     let data_y = ArrayD::from_elem(IxDyn(&[2, 2]), 3.0);
//
//     let mut tensor_x = Tensor::new(data_x);
//     let tensor_y = Tensor::new(data_y);
//
//     let tensor_z = tensor_x + tensor_y;
//     tensor_z.backward();
// }
