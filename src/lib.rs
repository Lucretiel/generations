//TODO: no_std

//! Generations is a utility library for running generation-based simulations,
//! such as Game of Life, other cellular automata, and genetic algorithms. Its
//! purpose is to make it unnecessary to allocate new model structures with
//! each generation; instead, two generations instances are stored in a
//! [`Generations`] struct, one of which is always considered the visible "current"
//! generation

use std::collections;
use std::ffi;
use std::mem;

// TODO: make this derivable
/// [`Clearable`] is a trait for data structures that can be cleared, especially
/// without deallocating storage. Models used in [`Generations`] must be
/// [`Clearable`], and the output generation will be cleared before each step.
///
/// It is implemented for all std data structures and many std types; feel free
/// to add pull requests for any std types that you feel should be clearable
pub trait Clearable {
    fn clear(&mut self);
}

macro_rules! clearable {
    ($type:ident $(:: $type_tail:ident)* $(< $($param:ident $(: $bound:ident $(+ $tail:ident)* $(+)?)?),+ $(,)? >)?) => {
        impl $(< $($param $(: $bound $(+ $tail)*)? ,)+ >)? $crate::Clearable for $type $(:: $type_tail)* $(< $($param,)+ >)? {
            #[inline]
            fn clear(&mut self) {
                $type $(:: $type_tail)* ::clear(self)
            }
        }
    };
    (($($name:ident),* $(,)?)) => {
        impl < $($name : Clearable),* > Clearable for ($($name,)*) {
            #[allow(non_snake_case)]
            fn clear(&mut self) {
                let ($(ref mut $name,)*) = *self;
                $(Clearable::clear($name);)*
            }
        }
    }
}

clearable! {Vec<T>}
clearable! {String}
clearable! {collections::BTreeMap<K: Ord, V>}
clearable! {collections::BTreeSet<T: Ord>}
clearable! {collections::BinaryHeap<T>}
clearable! {collections::HashMap<K, V, S>}
clearable! {collections::HashSet<T, S>}
clearable! {collections::LinkedList<T>}
clearable! {collections::VecDeque<T>}
clearable! {ffi::OsString}
clearable! {()}
clearable! {(A)}
clearable! {(A, B)}
clearable! {(A, B, C)}
clearable! {(A, B, C, D)}
clearable! {(A, B, C, D, E)}
clearable! {(A, B, C, D, E, F)}
clearable! {(A, B, C, D, E, F, G)}
clearable! {(A, B, C, D, E, F, G, H)}
clearable! {(A, B, C, D, E, F, G, H, I)}
clearable! {(A, B, C, D, E, F, G, H, I, J)}
clearable! {(A, B, C, D, E, F, G, H, I, J, K)}
clearable! {(A, B, C, D, E, F, G, H, I, J, K, L)}
clearable! {(A, B, C, D, E, F, G, H, I, J, K, L, M)}
clearable! {(A, B, C, D, E, F, G, H, I, J, K, L, M, N)}
clearable! {(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O)}
clearable! {(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P)}

impl<T: Clearable> Clearable for Box<T> {
    fn clear(&mut self) {
        T::clear(self)
    }
}

impl<T: Clearable> Clearable for Option<T> {
    fn clear(&mut self) {
        if let Some(value) = self.as_mut() {
            value.clear();
        }
    }
}

/// For (defaultable) models, this helper function converts a stepper that
/// takes a scratch generation by value (and returns it) to a version that
/// takes a scratch generation by mutable reference and mutates it in place.
/// In order to ensure panic safety, the value is replaced with a default if
/// the stepper panics, to prevent double-frees and other unwinding issues.
#[inline]
fn transitioner<T: Default>(mut f: impl FnMut(&T, T) -> T) -> impl FnMut(&T, &mut T) {
    // TODO: replace this with a version that uses ptr::read and ptr::write,
    // and only write the default to scratch_gen_ref in the event of a panic.
    // That version is much more complex, and I *think* LLVM can optimize this
    // to the same thing (especially when there are no panicking code paths)
    #[inline] move |current_gen_ref, scratch_gen_ref| {
        let scratch_gen = mem::replace(scratch_gen_ref, T::default());
        let mut next_gen = f(current_gen_ref, scratch_gen);
        mem::swap(scratch_gen_ref, &mut next_gen);
    }
}

/// This struct manages transitions between generations. It stores two models,
/// one of which is considered "current" and the other "scratch". The [`step`][Generations::step]
/// method advances the simulation by calling a function with a reference to
/// the current generation and a mutable reference to the scratch generation;
/// the function uses the current generation to write the next generation out
/// to the scratch generation, after which it becomes the new current generation
/// (and the previous generation becomes the new scratch generation).
#[derive(Debug, Clone)]
pub struct Generations<Model: Clearable> {
    current: Model,
    scratch: Model,
}

impl<Model: Clearable> Generations<Model> {
    /// Create a new `Generations` instance with a seed model, which will
    /// become the initial current generation, and a scratch generation.
    #[inline]
    pub fn new(seed: Model, scratch: Model) -> Self {
        Generations {
            current: seed,
            scratch,
        }
    }

    /// Get a reference to the current generation. This is the result of the
    /// most recent step or reset, or the seed generation if no steps have
    #[inline]
    pub fn current(&self) -> &Model {
        &self.current
    }

    /// Advance the simulation 1 step using a stepping function. The stepping
    /// function takes a reference to the current generation and a mutable
    /// reference to the new generation, which is cleared before the stepping
    /// function is called. It is expected to advance the simulation by reading
    /// the current genration and writing the next generation. After the
    /// stepping function writes the new generation, it is marked as current.
    ///
    /// This function returns a reference to the *previously current*
    /// generation, so that it can be compared if desired with the current
    /// generation.
    #[inline]
    pub fn step(&mut self, stepper: impl FnOnce(&Model, &mut Model)) -> &Model {
        self.scratch.clear();
        stepper(&self.current, &mut self.scratch);
        mem::swap(&mut self.current, &mut self.scratch);
        &self.scratch
    }

    /// Replace the current generation with a new seed generation using a
    /// function. Has no effect on the existing scratch generation. The
    /// current generation is cleared before the seed function is called.
    #[inline]
    pub fn reset_with(&mut self, seeder: impl FnOnce(&mut Model)) {
        self.current.clear();
        seeder(&mut self.current)
    }

    /// Replace the current generation with a new seed generation. Has no
    /// effect on the existing scratch generation.
    #[inline]
    pub fn reset(&mut self, seed: Model) {
        self.current = seed;
    }

    /// Combine a `Generations` struct with a repeatable stepping function, to
    /// create a simulation that can be stepped with the same logic each
    /// generation. See [`step`][Generations::step] for an explaination of
    /// the stepping function.
    #[inline]
    pub fn with_rule<F: FnMut(&Model, &mut Model)>(self, stepper: F) -> Simulation<Model, F> {
        Simulation {
            generations: self,
            stepper,
        }
    }
}

impl<Model: Clone + Clearable> Generations<Model> {
    /// Create a new `Generations` instance with a seed model. Clone the seed
    /// model to create an initial scratch model.
    #[inline]
    pub fn new_cloned(seed_generation: Model) -> Self {
        let scratch = seed_generation.clone();
        Self::new(seed_generation, scratch)
    }
}

impl<Model: Default + Clearable> Generations<Model> {
    /// Create a new `Generations` instance with a seed model. The Model type's
    /// default value is used as the initial scratch model.
    #[inline]
    pub fn new_defaulted(seed_generation: Model) -> Self {
        Self::new(seed_generation, Model::default())
    }

    #[inline]
    pub fn step_transition(&mut self, f: impl FnMut(&Model, Model) -> Model) -> &Model {
        self.step(transitioner(f))
    }

    #[inline]
    pub fn with_transition(self, f: impl FnMut(&Model, Model) -> Model) -> Simulation<Model, impl FnMut(&Model, &mut Model)> {
        self.with_rule(transitioner(f))
    }
}

#[derive(Debug, Clone)]
pub struct Simulation<Model: Clearable, Step: FnMut(&Model, &mut Model)> {
    generations: Generations<Model>,
    stepper: Step,
}

impl<Model: Clearable, Step: FnMut(&Model, &mut Model)> Simulation<Model, Step> {
    /// Return a reference to the current generation, which is the result of
    /// the most recent step, or the seed generation if no steps have been
    /// performed.
    #[inline]
    pub fn current(&self) -> &Model {
        self.generations.current()
    }

    /// Step the simulation using the stored stepping function.
    #[inline]
    pub fn step(&mut self) -> &Model {
        self.generations.step(&mut self.stepper)
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
// TODO: create an iterator. Probably impossible until we get GATs.
