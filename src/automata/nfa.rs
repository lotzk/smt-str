use std::collections::{HashMap, VecDeque};

use crate::alphabet::Alphabet;

use super::{det, Automaton, AutomatonError, DState, State, Transition, DFA};

pub type NFA = Automaton<NState>;

/// A state in a nondeterministic finite automaton.
/// Contains a collection of transitions to other states.
/// In nondeterministic automata, a state can have multiple transitions with the same input leading to different states.
#[derive(Debug, Clone, Eq, Hash, PartialEq, Default)]
pub struct NState {
    transitions: Vec<Transition>,
}

impl NState {
    /// Create a new state with the given transitions
    pub fn with_transitions(transitions: Vec<Transition>) -> Self {
        Self { transitions }
    }
}

impl State for NState {
    fn transitions(&self) -> Vec<Transition> {
        self.transitions.clone()
    }

    fn remove_transition(&mut self, transition: &Transition) {
        self.transitions.retain(|t| t != transition);
    }

    fn add_transition(&mut self, transition: Transition) {
        self.transitions.push(transition);
    }

    /// Adds a constant value to all destination states of all transitions and returns the new state.
    fn offset_transitions(&self, offset: usize) -> Self {
        let mut new_transitions = vec![];
        for mut t in self.transitions.iter().cloned() {
            t.destination += offset;
            new_transitions.push(t);
        }
        Self {
            transitions: new_transitions,
        }
    }
}

impl NFA {
    /// Determinizes this automaton. The result is a deterministic finite automaton that accepts the same language as this automaton.
    pub fn determinize(&self) -> Result<DFA, AutomatonError> {
        det::determinize(self)
    }

    /// Removes epsilon transitions from this automaton.
    /// Returns a new automaton that accepts the same language as this automaton but has no epsilon transitions.
    pub fn remove_epsilons(&self) -> Result<NFA, AutomatonError> {
        let mut queue = VecDeque::new();
        let mut aut = Self::new();

        // Maps state ids from the old automaton to the new automaton
        let mut old_to_new = HashMap::new();

        // Initialize the queue and the automaton with the initial state
        if let Some(initial_state) = self.initial {
            let new_initial_state = aut.new_state();
            aut.initial = Some(new_initial_state);
            queue.push_back(initial_state);
            old_to_new.insert(initial_state, new_initial_state);
        }

        // Invariant: The old_to_new map contains a mapping from the given state id to the new state id
        while let Some(state_id) = queue.pop_front() {
            let new_state_id = *old_to_new.get(&state_id).unwrap();

            // compute the epsilon closure of the current state
            let epsi_closure = self.epsilon_closure(state_id)?;

            for closure_state_id in &epsi_closure {
                let closure_state = self.get_state(*closure_state_id)?;
                // If any state in the closure is a final state, mark the new state as final
                if self.finals.contains(closure_state_id) {
                    aut.finals.insert(new_state_id);
                }
                for transition in &closure_state.transitions {
                    if !transition.is_epsilon() {
                        // Process all non-epsilon transitions by adding a transition from the new state to the destination state
                        let dest = transition.get_dest();
                        // If the destination state is not yet in the old_to_new map, create it and add it to the queue
                        let new_dest_id = match old_to_new.get(&dest) {
                            Some(id) => *id,
                            None => {
                                let new_state = aut.new_state();
                                old_to_new.insert(dest, new_state);
                                queue.push_back(dest);
                                new_state
                            }
                        };
                        let new_transition = Transition {
                            type_: transition.type_,
                            destination: new_dest_id,
                        };
                        aut.get_state_mut(new_state_id)?
                            .add_transition(new_transition)
                    }
                }
            }
        }

        Ok(aut)
    }

    /// Merges the given automaton into this automaton.
    /// The states of the given automaton are offset by the number of states in this automaton such that they do not overlap.
    /// The final states of the other automaton are added to the final states of this automaton.
    /// The initial state of the other automaton is ignored. That means, trimming the resulting automaton directly after merging would undo the merge.
    /// Returns the offset of the states of the other automaton.
    pub fn merge(&mut self, other: Self) -> usize {
        let offset = self.states.len();
        for state in other.states.iter() {
            self.add_state(state.offset_transitions(offset));
        }
        for final_state in other.finals.iter() {
            self.finals.insert(*final_state + offset);
        }
        offset
    }

    pub fn compress_edges(&mut self) {
        for state in self.states.iter_mut() {
            // Sort transitions by destination state
            let mut dest_map = HashMap::new();
            for transition in &state.transitions {
                let dest = transition.get_dest();
                let entry = dest_map.entry(dest).or_insert_with(Vec::new);
                entry.push(transition.clone());
            }
            // Compress edges with same destination in an alphabet of ranges
            let mut new_transitions = Vec::new();
            let mut alph = Alphabet::empty();
            for (dest, t) in dest_map.into_iter() {
                for transition in t {
                    match transition.get_type() {
                        super::TransitionType::Range(char_range) => alph.insert(*char_range),
                        super::TransitionType::NotRange(_) => {
                            // cannot compress
                            new_transitions.push(transition.clone())
                        }
                        super::TransitionType::Epsilon => new_transitions.push(transition.clone()),
                    }
                }
                for range in alph.iter_ranges() {
                    new_transitions.push(Transition::range(range, dest));
                }
            }
            state.transitions = new_transitions;
        }
    }
}

impl From<DState> for NState {
    /// Every state in a deterministic automaton is also a state in a nondeterministic automaton
    fn from(value: DState) -> Self {
        NState::with_transitions(value.transitions())
    }
}

impl From<DFA> for NFA {
    fn from(value: DFA) -> Self {
        let mut aut = Automaton::new();
        aut.initial = value.initial;
        aut.finals = value.finals;
        for state in value.states {
            aut.add_state(NState::with_transitions(state.transitions()));
        }
        aut
    }
}

#[cfg(test)]
mod tests {
    use crate::{alphabet::CharRange, automata::TransitionType};

    use super::*;
    #[test]
    fn test_remove_epsilon_basic() {
        // s0 --ε--> s1 --ε--> *s2*
        let mut a: NFA = Automaton::new();
        let s0 = a.new_state();
        let s1 = a.new_state();
        let s2 = a.new_state();

        a.initial = Some(s0);
        a.finals.insert(s2);

        a.get_state_mut(s0)
            .unwrap()
            .add_transition(Transition::epsilon(s1));
        a.get_state_mut(s1)
            .unwrap()
            .add_transition(Transition::epsilon(s2));

        let result = a.remove_epsilons().unwrap();

        assert_eq!(result.states.len(), 1);
        assert!(result.initial.is_some());
        let initial = result.initial.unwrap();
        assert!(result.finals.contains(&initial));
        assert!(result.get_state(initial).unwrap().transitions.is_empty())
    }

    #[test]
    fn test_remove_epsilon_no_epsilon() {
        // s0 --a--> s1
        let mut a: NFA = Automaton::new();
        let s0 = a.new_state();
        let s1 = a.new_state();

        a.initial = Some(s0);
        a.finals.insert(s1);

        a.get_state_mut(s0)
            .unwrap()
            .add_transition(Transition::char('a', s1));

        let result = a.remove_epsilons().unwrap();

        assert_eq!(result.states.len(), 2);
        assert!(result.initial.is_some());
        assert_eq!(result.finals.len(), 1);

        let initial_state = result.initial.unwrap();
        let final_state = *result.finals.iter().next().unwrap();

        assert_eq!(initial_state, s0);
        assert_eq!(final_state, s1);

        let initial_transitions = &result.get_state(initial_state).unwrap().transitions;
        assert_eq!(initial_transitions.len(), 1);
        assert_eq!(
            *initial_transitions[0].get_type(),
            TransitionType::Range(CharRange::singleton('a'))
        );
        assert_eq!(initial_transitions[0].get_dest(), final_state);
    }

    #[test]
    fn test_remove_epsilon_complex() {
        // s0 --a--> s1 --ε--> s2 --b--> *s3*
        let mut a: NFA = Automaton::new();
        let s0 = a.new_state();
        let s1 = a.new_state();
        let s2 = a.new_state();
        let s3 = a.new_state();

        a.initial = Some(s0);
        a.finals.insert(s3);

        a.get_state_mut(s0)
            .unwrap()
            .add_transition(Transition::char('a', s1));
        a.get_state_mut(s1)
            .unwrap()
            .add_transition(Transition::epsilon(s2));
        a.get_state_mut(s2)
            .unwrap()
            .add_transition(Transition::char('b', s3));

        let result = a.remove_epsilons().unwrap();

        assert_eq!(result.states.len(), 3);
        assert!(result.initial.is_some());
        assert_eq!(result.finals.len(), 1);

        let initial_state = result.initial.unwrap();

        let initial_transitions = &result.get_state(initial_state).unwrap().transitions;
        assert_eq!(initial_transitions.len(), 1);
        assert_eq!(
            *initial_transitions[0].get_type(),
            TransitionType::Range(CharRange::singleton('a'))
        );
        let middle_state = initial_transitions[0].get_dest();

        let middle_transitions = &result.get_state(middle_state).unwrap().transitions;
        assert_eq!(middle_transitions.len(), 1);
        assert_eq!(
            *middle_transitions[0].get_type(),
            TransitionType::Range(CharRange::singleton('b'))
        );
        let final_state = middle_transitions[0].get_dest();
        assert!(result.finals.contains(&final_state));
    }

    #[test]
    fn test_remove_epsilon_multiple_epsilons() {
        // s0 --ε--> s1 --ε--> s2 --ε--> *s3*
        let mut a: NFA = Automaton::new();
        let s0 = a.new_state();
        let s1 = a.new_state();
        let s2 = a.new_state();
        let s3 = a.new_state();

        a.initial = Some(s0);
        a.finals.insert(s3);

        a.get_state_mut(s0)
            .unwrap()
            .add_transition(Transition::epsilon(s1));
        a.get_state_mut(s1)
            .unwrap()
            .add_transition(Transition::epsilon(s2));
        a.get_state_mut(s2)
            .unwrap()
            .add_transition(Transition::epsilon(s3));

        let result = a.remove_epsilons().unwrap();

        assert_eq!(result.states.len(), 1);
        assert!(result.initial.is_some());
        let initial = result.initial.unwrap();
        assert!(result.finals.contains(&initial));
        assert!(result.get_state(initial).unwrap().transitions.is_empty())
    }

    #[test]
    fn test_remove_epsilon_with_direct_transition() {
        // s0 --(a|ε)--> s1
        let mut a: NFA = Automaton::new();
        let s0 = a.new_state();
        let s1 = a.new_state();

        a.initial = Some(s0);
        a.finals.insert(s1);

        a.get_state_mut(s0)
            .unwrap()
            .add_transition(Transition::char('a', s1));
        a.get_state_mut(s0)
            .unwrap()
            .add_transition(Transition::epsilon(s1));

        let result = a.remove_epsilons().unwrap();

        assert_eq!(result.states.len(), 2);
        assert!(result.initial.is_some());
        assert_eq!(result.finals.len(), 2);

        let initial_state = result.initial.unwrap();

        let initial_transitions = &result.get_state(initial_state).unwrap().transitions;
        assert_eq!(initial_transitions.len(), 1);
        assert_eq!(
            *initial_transitions[0].get_type(),
            TransitionType::Range(CharRange::singleton('a'))
        );
        let dest = initial_transitions[0].get_dest();
        assert!(result.finals.contains(&dest));
    }
}
