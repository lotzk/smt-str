use itertools::Itertools;

use crate::alphabet::AlphabetPartitionMap;

use super::{Automaton, State, StateId, Transition, TransitionType};
pub type DFA = Automaton<DState>;

/// A state in a deterministic finite automaton.
/// The transitions are partitioned by the characters they are associated with.
/// Each character range is associated with the state it transitions to.
/// Adding a transition that overlaps with an existing transition is not allowed.
#[derive(Debug, Clone, Default)]
pub struct DState {
    /// A partitioning of the characters. Each range is associated with the state it transitions to.
    trans: AlphabetPartitionMap<StateId>,
}

impl State for DState {
    fn transitions(&self) -> Vec<Transition> {
        self.trans
            .iter()
            .map(|(r, s)| Transition::range(*r, *s))
            .collect_vec()
    }

    /// Adds a transition to this state. The transition must be a range transition and the range must not overlap with any existing transitions.
    /// Panics if
    ///
    /// - the transition is an epsilon transition
    /// - the range of the transition is not disjoint from the ranges of the other transitions
    ///
    /// In both cases the automaton is not deterministic.
    fn add_transition(&mut self, transition: Transition) {
        match transition.type_ {
            TransitionType::Range(r) => match self.trans.insert(r, transition.get_dest()) {
                Ok(_) => (),
                Err(e) => panic!(
                    "cannot add transition {transition}, overlaps with '{}'.\n{:?}",
                    e, self.trans
                ),
            },
            _ => panic!("Cannot add epsilon transitions to deterministic automaton"),
        }
    }

    fn remove_transition(&mut self, transition: &Transition) {
        if let TransitionType::Range(r) = transition.type_ {
            self.trans.remove(r);
        }
    }

    fn offset_transitions(&self, offset: usize) -> Self {
        let mut cloned = self.trans.clone();
        cloned.iter_mut().for_each(|(_, s)| {
            *s += offset;
        });
        Self { trans: cloned }
    }
}
