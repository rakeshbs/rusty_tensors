extern crate rusty_tensors;
use ndarray::array;
use rusty_tensors::tensor::*;

fn main() {
    let a = tensor(array![[1.0], [2.0], [3.0]], true);
    let b = tensor(array![[1.0], [2.0], [3.0]], true);

    let num_epochs = 20;
    let lr = 0.1;

    for i in 0..num_epochs {
        println!("Epoch {}", i);
        print!("");
        let c = add(&pow(&a, 2.), &pow(&b, 2.));
        let loss = sum(&c);
        println!("loss: {:?}\n", (loss.output())[(0, 0)]);
        loss.backward();
        a.step(lr);
        b.step(lr);
        a.zero_grad();
        b.zero_grad();
    }
    println!("a: {:?}\n", a.output());
    println!("b: {:?}\n", b.output());
}
