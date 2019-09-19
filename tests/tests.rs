use generations::{Clearable, Generations};

// This struct provides no new, clone, or default; in principle it should
// be impossible for Generations to make more instances of it, which means
// that its use in our API demonstrates that Generations is always reusing
// the same 2 instances
#[derive(Eq, PartialEq, Debug)]
struct UnconstructableWrapper<T>(T);

impl<T: Clearable> Clearable for UnconstructableWrapper<T> {
    fn clear(&mut self) {
        self.0.clear()
    }
}

#[test]
fn basic_test() {
    // For these tests, the model that we're using is a vector, where in
    // each generation, the new value is the sum of the old value and its
    // neighbors. For instance:
    //
    // [1  0  1  1  -1 -1]
    // [1  2  2  1  -1 -2]
    // [3  5  5  2  -2 -3]
    // [8  13 12 5  -3 -5]
    // [21 33 30 14 -3 -8]
    //
    // For simplicity of implementation of bounds checking, we use a vector
    // bounded by zeroes to make this work
    let gen = Generations::new(
        UnconstructableWrapper(vec![0, 1, 0, 1, 1, -1, -1, 0]),
        UnconstructableWrapper(vec![]),
    );

    let mut gen = gen.with_rule(move |current_gen, next_gen| {
        next_gen.0.push(0);

        for window in current_gen.0.windows(3) {
            next_gen.0.push(window[0] + window[1] + window[2]);
        }
        next_gen.0.push(0);
    });

    assert_eq!(&gen.current().0, &[0, 1, 0, 1, 1, -1, -1, 0]);
    gen.step();
    assert_eq!(&gen.current().0, &[0, 1, 2, 2, 1, -1, -2, 0]);
    gen.step();
    assert_eq!(&gen.current().0, &[0, 3, 5, 5, 2, -2, -3, 0]);
    gen.step();
    assert_eq!(&gen.current().0, &[0, 8, 13, 12, 5, -3, -5, 0]);
    gen.step();
    assert_eq!(&gen.current().0, &[0, 21, 33, 30, 14, -3, -8, 0]);

    // TODO: find a way to test that no reallocations are happening
}
