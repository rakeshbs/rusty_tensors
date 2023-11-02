use crate::tensor::*;

pub trait Layer {
    fn forward(&self, input: TensorRef) -> TensorRef;
    fn parameters(&self) -> Vec<TensorRef>;
}

pub struct Linear {
    weight: TensorRef,
    bias: TensorRef,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        let weight = tensor_rand(in_features, out_features);
        let bias = tensor_rand(out_features, 1);
        Linear { weight, bias }
    }
}

impl Layer for Linear {
    fn forward(&self, input: TensorRef) -> TensorRef {
        add(&mul(&self.weight, &input), &self.bias)
    }

    fn parameters(&self) -> Vec<TensorRef> {
        vec![self.weight.clone(), self.bias.clone()]
    }
}
