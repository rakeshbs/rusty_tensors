extern crate rusty_tensors;
use rusty_tensors::tensor::*;

fn main() {
    let a = tensor(&[1.0, 2.0, 3.0], true);
    let b = tensor(&[4.0, 5.0, 6.0], true);

    let num_epochs = 10;

    for i in 0..num_epochs {
        let c = add(&a, &b);
        let d = mul(&a, &b);
        let e = add(&c, &d);

        println!("e: {:?}", e.output());
        e.backward();
        a.step(0.1);
        b.step(0.1);
        a.zero_grad();
        b.zero_grad();
    }

    println!("a: {:?}", a.output());
    println!("b: {:?}", b.output());
}
