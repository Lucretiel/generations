use generations::{Generations};

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
        vec![0, 1, 0, 1, 1, -1, -1, 0],
        vec![],
    );

    let mut gen = gen.with_transition(|current_gen, mut next_gen| {
        next_gen.push(0);

        for window in current_gen.windows(3) {
            next_gen.push(window[0] + window[1] + window[2]);
        }
        next_gen.push(0);
        next_gen
    });

    assert_eq!(gen.current(), &[0, 1, 0, 1, 1, -1, -1, 0]);
    gen.step();
    assert_eq!(gen.current(), &[0, 1, 2, 2, 1, -1, -2, 0]);
    gen.step();
    assert_eq!(gen.current(), &[0, 3, 5, 5, 2, -2, -3, 0]);
    gen.step();
    assert_eq!(gen.current(), &[0, 8, 13, 12, 5, -3, -5, 0]);
    gen.step();
    assert_eq!(gen.current(), &[0, 21, 33, 30, 14, -3, -8, 0]);

    // TODO: find a way to test that no reallocations are happening
}
