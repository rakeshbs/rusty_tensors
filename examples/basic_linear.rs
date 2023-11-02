extern crate rusty_tensors;
use ndarray::array;
use rusty_tensors::nn::*;
use rusty_tensors::tensor::*;

struct Network {
    linear1: Linear,
    linear2: Linear,
    linear3: Linear,
}

impl Network {
    pub fn new() -> Self {
        let linear1 = Linear::new(3, 100);
        let linear2 = Linear::new(100, 30);
        let linear3 = Linear::new(30, 2);
        Network {
            linear1,
            linear2,
            linear3,
        }
    }

    pub fn forward(&self, input: TensorRef) -> TensorRef {
        let mut x = self.linear1.forward(input);
        x = relu(&x);
        x = self.linear2.forward(x);
        x = relu(&x);
        self.linear3.forward(x)
    }

    pub fn parameters(&self) -> Vec<TensorRef> {
        let mut params = self.linear1.parameters();
        params.append(&mut self.linear2.parameters());
        params.append(&mut self.linear3.parameters());
        params
    }
}
fn main() {
    let net = Network::new();
    let num_epochs = 100;
    let lr = 0.1;
    let input = tensor(array![[1.0], [2.0], [3.0]], false);
    let target = tensor(array![[3.0], [5.0]], false);

    for i in 0..num_epochs {
        println!("Epoch {}", i);
        print!("");
        let output = net.forward(input.clone());
        let loss = &pow(&sub(&target, &output), 2.);
        println!("loss: {:?}\n", (loss.output())[(0, 0)]);
        loss.backward();
        for param in net.parameters() {
            param.step(lr);
            param.zero_grad();
        }
    }
    let output = net.forward(input.clone());
    println!("output: {:?}", output.output());
}
