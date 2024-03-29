/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 *
 * Copyright 2019 Nathan West
 */

#![cfg_attr(not(test), no_std)]

//! Generations is a utility library for running generation-based simulations,
//! such as [Conway's Game of Life](https://www.wikiwand.com/en/Conway%27s_Game_of_Life),
//! other [cellular automata](https://www.wikiwand.com/en/Elementary_cellular_automaton),
//! and [genetic algorithms](https://www.wikiwand.com/en/Genetic_algorithm). Its
//! purpose is to make it unnecessary to allocate new model structures with
//! each generation; instead, two generations instances are stored in a
//! [`Generations`] struct, one of which is always considered the visible "current"
//! generation.
//!
//! # Example
//!
//! In this example, our system is an array of integers. With each subsequent
//! generation, the values in the array are the sum of the original value and
//! its two neighbors. For simplicity, we ignore the first and last element.
//!
//! For example:
//!
//! ```text
//! [0  1  0  1  1  -1 -1  0]
//! [0  1  2  2  1  -1 -2  0]
//! [0  3  5  5  2  -2 -3  0]
//! [0  8  13 12 5  -3 -5  0]
//! [0  21 33 30 14 -3 -8  0]
//! ```
//! ```
//! use generations::Generations;
//!
//! // Our model type is a vector.
//! let initial_state = vec![0, 1, 0, 1, 1, -1, -1, 0];
//!
//! // A Generations instance stores our model, plus some scratch space to
//! // use as the next generation each step. When stepped, the new generation is
//! // written to this scratch space, which then becomes the current generation
//! // while the previous generation becomes the new scratch space.
//! let gens = Generations::new_defaulted(initial_state);
//!
//! // A Simulation combines a Generations instance with a rule to apply with
//! // each step of the simulation
//! let mut sim = gens.with_rule(move |current_gen, next_gen| {
//!     // Make sure to reset the `next_gen` to a blank state before continuing
//!     next_gen.clear();
//!
//!     if let Some(&first) = current_gen.first() {
//!         next_gen.push(first);
//!     }
//!
//!     for window in current_gen.windows(3) {
//!         next_gen.push(window[0] + window[1] + window[2]);
//!     }
//!
//!     if current_gen.len() > 1 {
//!         next_gen.push(*current_gen.last().unwrap())
//!     }
//! });
//!
//! assert_eq!(sim.current(), &[0, 1, 0, 1, 1, -1, -1, 0]);
//! sim.step();
//! assert_eq!(sim.current(), &[0, 1, 2, 2, 1, -1, -2, 0]);
//! sim.step();
//! assert_eq!(sim.current(), &[0, 3, 5, 5, 2, -2, -3, 0]);
//! sim.step();
//! assert_eq!(sim.current(), &[0, 8, 13, 12, 5, -3, -5, 0]);
//! sim.step();
//! assert_eq!(sim.current(), &[0, 21, 33, 30, 14, -3, -8, 0]);
//! ```

use core::fmt;
use core::mem;

/// This struct manages transitions between generations. It stores two models,
/// one of which is considered "current" and the other "scratch". The [`step`][Generations::step]
/// method advances the simulation by calling a function with a reference to
/// the current generation and a mutable reference to the scratch generation;
/// the function uses the current generation to write the next generation out
/// to the scratch generation, after which it becomes the new current generation
/// (and the previous generation becomes the new scratch generation).
#[derive(Debug, Clone)]
pub struct Generations<Model> {
    current: Model,
    scratch: Model,
}

impl<Model> Generations<Model> {
    /// Create a new `Generations` instance with a seed model, which will
    /// become the initial current generation, and a scratch generation.
    ///
    /// # Example
    ///
    /// Create a simulation using a vector for the seed generation and a
    /// pre-allocated vector for the scratch generation
    ///
    /// ```
    /// use generations::Generations;
    ///
    /// let mut gen = Generations::new(
    ///     vec![1, 2, 3, 4, 5],
    ///     Vec::with_capacity(5)
    /// );
    ///
    /// gen.step(|current_gen, next_gen| {
    ///     assert_eq!(current_gen, &[1, 2, 3, 4, 5]);
    ///     assert_eq!(next_gen, &[]);
    ///     assert_eq!(next_gen.capacity(), 5);
    /// });
    /// ```
    #[inline]
    #[must_use]
    pub fn new(seed: Model, scratch: Model) -> Self {
        Generations {
            current: seed,
            scratch,
        }
    }

    /// Get a reference to the current generation. This is the result of the
    /// most recent step or reset, or the seed generation if no steps have been
    /// run.
    ///
    /// # Example
    ///
    /// ```
    /// use generations::Generations;
    ///
    /// let gen = Generations::new_defaulted(vec![1, 2, 3, 4]);
    /// assert_eq!(gen.current(), &[1, 2, 3, 4]);
    /// ```
    #[inline]
    #[must_use]
    pub fn current(&self) -> &Model {
        &self.current
    }

    /// Advance the simulation 1 step using a stepping function. The stepping
    /// function takes a reference to the current generation and a mutable
    /// reference to the new generation. The stepping function should advance
    /// the simulation by reading the current genration and writing the next
    /// generation. After the stepping function writes the new generation,
    /// it is marked as current.
    ///
    /// This function returns a reference to the *previously current*
    /// generation, so that it can be compared if desired with the current
    /// generation.
    ///
    /// # Example
    ///
    /// ```
    /// // Simple example that rotates a vector 1 step
    /// use generations::Generations;
    ///
    /// let mut gen = Generations::new_cloned(vec![1, 2, 3, 4]);
    ///
    /// let prev = gen.step(|current_gen, next_gen| {
    ///     next_gen.clear();
    ///     next_gen.extend(current_gen.iter().skip(1));
    ///     next_gen.extend(current_gen.first());
    /// });
    ///
    /// assert_eq!(prev, &[1, 2, 3, 4]);
    /// assert_eq!(gen.current(), &[2, 3, 4, 1]);
    /// ```
    #[inline]
    pub fn step(&mut self, stepper: impl FnOnce(&Model, &mut Model)) -> &Model {
        stepper(&self.current, &mut self.scratch);
        mem::swap(&mut self.current, &mut self.scratch);
        &self.scratch
    }

    /// Replace the current generation with a new seed generation using a
    /// function. Has no effect on the existing scratch generation.
    ///
    /// ```
    /// use generations::Generations;
    ///
    /// let mut gen = Generations::new_defaulted(vec![1, 2, 3, 4]);
    /// gen.reset_with(|seed_gen| {
    ///     seed_gen.clear();
    ///     seed_gen.extend(&[5, 5, 5, 5]);
    /// });
    /// assert_eq!(gen.current(), &[5, 5, 5, 5]);
    /// ```
    #[inline]
    pub fn reset_with(&mut self, seeder: impl FnOnce(&mut Model)) {
        seeder(&mut self.current)
    }

    /// Replace the current generation with a new seed generation. Has no
    /// effect on the existing scratch generation.
    ///
    /// See also [`reset_with`][Generations::reset_with] for a reset
    /// method that reuses the existing storage of the current generation.
    ///
    /// ```
    /// use generations::Generations;
    ///
    /// let mut gen = Generations::new_defaulted(vec![1, 2, 3, 4]);
    /// gen.reset(vec![4, 3, 2, 1]);
    /// assert_eq!(gen.current(), &[4, 3, 2, 1]);
    /// ```
    #[inline]
    pub fn reset(&mut self, seed: Model) {
        self.current = seed;
    }

    /// Combine a `Generations` struct with a repeatable stepping function, to
    /// create a simulation that can be stepped with the same logic each
    /// generation. See [`step`][Generations::step] for an explaination of
    /// the stepping function.
    #[inline]
    #[must_use]
    pub fn with_rule<F: FnMut(&Model, &mut Model)>(self, stepper: F) -> Simulation<Model, F> {
        Simulation {
            generations: self,
            stepper,
        }
    }
}

impl<Model: Clone> Generations<Model> {
    /// Create a new `Generations` instance with a seed model. Clone the seed
    /// model to create an initial scratch model.
    #[inline]
    #[must_use]
    pub fn new_cloned(seed_generation: Model) -> Self {
        let scratch = seed_generation.clone();
        Self::new(seed_generation, scratch)
    }

    /// Replace the current generation with a clone of a new seed generation.
    /// Has no effect on the current scratch generation.
    #[inline]
    pub fn reset_from(&mut self, seed: &Model) {
        self.current.clone_from(seed)
    }
}

impl<Model: Default> Generations<Model> {
    /// Create a new `Generations` instance with a seed model. The Model type's
    /// default value is used as the initial scratch model.
    #[inline]
    #[must_use]
    pub fn new_defaulted(seed_generation: Model) -> Self {
        Self::new(seed_generation, Model::default())
    }
}

/// A [`Simulation`] is a [`Generations`] instance combined with a stepping
/// function. It allows you to repeatedly step through generations, using
/// the same logic with each step.
///
/// It provides a [`step`][Simulation::step] method, which advances the
/// simulaton with the underlying stepper.
///
/// It is constructed with the [`Generations::with_rule`] method.
#[derive(Clone)]
pub struct Simulation<Model, Step> {
    generations: Generations<Model>,
    stepper: Step,
}

impl<Model, Step> Simulation<Model, Step> {
    /// Return a reference to the current generation, which is the result of
    /// the most recent step, or the seed generation if no steps have been
    /// performed.
    #[inline]
    #[must_use]
    pub fn current(&self) -> &Model {
        self.generations.current()
    }

    /// Replace the current generation with a new seed generation using a
    /// function. Has no effect on the existing scratch generation. The current
    /// generation is cleared before the seed function is called.
    #[inline]
    pub fn reset_with(&mut self, seeder: impl FnOnce(&mut Model)) {
        self.generations.reset_with(seeder)
    }

    /// Replace the current generation with a new seed generation. Has no
    /// effect on the existing scratch generation.
    #[inline]
    pub fn reset(&mut self, seed: Model) {
        self.generations.reset(seed)
    }

    /// Discard the stepping function and return the underlying `Generations`
    /// instance
    #[inline]
    pub fn unwrap(self) -> Generations<Model> {
        self.generations
    }
}

impl<Model: Clone, Step> Simulation<Model, Step> {
    /// Replace the current generation with a clone of a new seed generation.
    /// Has no effect on the current scratch generation.
    #[inline]
    pub fn reset_from(&mut self, seed: &Model) {
        self.generations.reset_from(seed)
    }
}

impl<Model, Step: FnMut(&Model, &mut Model)> Simulation<Model, Step> {
    /// Step the simulation using the stored stepping function. Returns a
    /// reference to the *previously current* generation.
    #[inline]
    pub fn step(&mut self) -> &Model {
        self.generations.step(&mut self.stepper)
    }
}

// TODO: default impl this, add a specialization for Step: Debug
impl<Model: fmt::Debug, Step> fmt::Debug for Simulation<Model, Step> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Simulation")
            .field("generations", &self.generations)
            .field("stepper", &"<closure>")
            .finish()
    }
}

impl<Model, Step> AsRef<Generations<Model>> for Simulation<Model, Step> {
    fn as_ref(&self) -> &Generations<Model> {
        &self.generations
    }
}
// TODO: create an iterator. Probably impossible until we get GATs.
// Conceivably we could just clone the model at every step of the iteration,
// but that would... defeat the entire purpose of the library? unless we wrapped
// it all up in an Rc or something.

#[cfg(test)]
mod tests {
    use crate::Generations;

    #[test]
    fn basic_test() {
        // This test checks essentially all of the functionality of the library.
        // individual methods not covered here are covered by doctests.

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
        let gen = Generations::new(vec![0, 1, 0, 1, 1, -1, -1, 0], vec![]);

        let mut gen = gen.with_rule(|current_gen, next_gen| {
            next_gen.clear();
            next_gen.push(0);

            for window in current_gen.windows(3) {
                next_gen.push(window[0] + window[1] + window[2]);
            }
            next_gen.push(0);
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

    #[test]
    fn debug_print_test() {
        // Simulations should be debug-printable when created with a closure
        let thing = Generations::new_defaulted(vec![0]).with_rule(|_c, _n| {});
        format!("{:?}", &thing);
    }
}
