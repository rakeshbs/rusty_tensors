extern crate rusty_tensors;
use rusty_tensors::tensor::*;

fn main() {
    let a = tensor(&[1.0, 2.0, 3.0], true);
    let b = tensor(&[4.0, 5.0, 6.0], true);

    let c = add(&a, &b);
    let d = mul(&a, &b);
    let e = add(&c, &d);

    println!("e: {}", d.output());
}
