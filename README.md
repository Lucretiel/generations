# generations

Generations is a utility library for running generation-based simulations, such as [Conway's Game of Life](https://www.wikiwand.com/en/Conway%27s_Game_of_Life), other [cellular automata](https://www.wikiwand.com/en/Elementary_cellular_automaton), and [genetic algorithms](https://www.wikiwand.com/en/Genetic_algorithm). Its purpose is to make it unnecessary to allocate new model structures with each generation; instead, two generations instances are stored in a [`Generations`] struct, one of which is always considered the visible "current" generation.

# Example

In this example, our system is an array of integers. With each subsequent generation, the values in the array are the sum of the original value and its two neighbors. For simplicity, we ignore the first and last element.

For example:

```text
[0  1  0  1  1  -1 -1  0]
[0  1  2  2  1  -1 -2  0]
[0  3  5  5  2  -2 -3  0]
[0  8  13 12 5  -3 -5  0]
[0  21 33 30 14 -3 -8  0]
```

```rust
use generations::Generations;

// Our model type is a vector. Any type that implements the `Clearable`
// trait can be used as a model; this includes all std data structures.
let initial_state = vec![0, 1, 0, 1, 1, -1, -1, 0];

// A Generations instance stores our model, plus some scratch space to
// use as the next generation each step. When stepped, the new generation is
// written to this scratch space, which then becomes the current generation
// while the previous generation becomes the new scratch space.
let gens = Generations::new_defaulted(initial_state);

// A Simulation combines a Generations instance with a rule to apply with
// each step of the simulation
let mut sim = gens.with_rule(move |current_gen, next_gen| {
    if let Some(&first) = current_gen.first() {
        next_gen.push(first);
    }

    for window in current_gen.windows(3) {
        next_gen.push(window[0] + window[1] + window[2]);
    }

    if current_gen.len() > 1 {
        next_gen.push(*current_gen.last().unwrap())
    }
});

assert_eq!(sim.current(), &[0, 1, 0, 1, 1, -1, -1, 0]);
sim.step();
assert_eq!(sim.current(), &[0, 1, 2, 2, 1, -1, -2, 0]);
sim.step();
assert_eq!(sim.current(), &[0, 3, 5, 5, 2, -2, -3, 0]);
sim.step();
assert_eq!(sim.current(), &[0, 8, 13, 12, 5, -3, -5, 0]);
sim.step();
assert_eq!(sim.current(), &[0, 21, 33, 30, 14, -3, -8, 0]);
```
