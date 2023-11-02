extern crate rusty_tensors;
use ndarray::array;
use rusty_tensors::tensor::*;

fn main() {
    let a = tensor(array![[1.0], [2.0], [3.0]], true);
    let b = tensor(array![[1.0], [2.0], [3.0]], true);

    let num_epochs = 1;

    for i in 0..num_epochs {
        println!("Epoch {}", i);
        print!("");
        let c = add(&pow(&a, 2.), &pow(&b, 2.));
        let d = sum(&c);

        println!("d: {:?}\n", d.output());
        d.backward();
        a.step(0.01);
        b.step(0.01);
        a.zero_grad();
        b.zero_grad();
        println!("a: {:?}\n", a.output());
        println!("b: {:?}\n", b.output());
    }
}
