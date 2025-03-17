use std::{
    collections::{BTreeSet, HashMap, VecDeque},
    fmt::Display,
};

use crate::alphabet::AlphabetPartitionMap;

use super::{Automaton, AutomatonError, State, StateId, Transition, TransitionType, DFA, NFA};

#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct StateSet(BTreeSet<StateId>);
impl StateSet {
    fn new() -> Self {
        Self(BTreeSet::new())
    }

    fn insert(&mut self, state: StateId) {
        self.0.insert(state);
    }

    fn iter(&self) -> impl Iterator<Item = &StateId> {
        self.0.iter()
    }

    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl From<StateId> for StateSet {
    fn from(state: StateId) -> Self {
        let mut set = StateSet::new();
        set.insert(state);
        set
    }
}
impl Display for StateSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut first = true;
        write!(f, "{{")?;
        for state in self.iter() {
            if first {
                first = false;
            } else {
                write!(f, ", ")?;
            }
            write!(f, "{}", state)?;
        }
        write!(f, "}}")
    }
}

/// Creates the deterministic automaton corresponding to this automaton, i.e., a deterministic automaton that accepts the same language as this automaton.
///
/// # Errors
///
/// This function will return an error if the given automaton contains an epsilon transition.
pub fn determinize(nfa: &NFA) -> Result<DFA, AutomatonError> {
    let mut dfa: DFA = Automaton::new();
    // Maps a set of NFA states to a DFA state
    let mut state_map: HashMap<StateSet, StateId> = HashMap::new();
    let mut queue: VecDeque<StateSet> = VecDeque::new();

    // Closure of the initial NFA state
    if let Some(initial_state) = nfa.initial {
        let mut initial_closure = StateSet::new();
        initial_closure.insert(initial_state);
        // Initial DFA state
        let dfa_initial_state = dfa.new_state();
        dfa.initial = Some(dfa_initial_state);
        if nfa.finals.contains(&initial_state) {
            dfa.finals.insert(dfa_initial_state);
        }
        state_map.insert(initial_closure.clone(), dfa_initial_state);
        queue.push_back(initial_closure);
    }

    while let Some(nfa_states) = queue.pop_front() {
        let &dfa_state_id = state_map.get(&nfa_states).unwrap();

        // Collect transitions for each symbol by partitioning the transitions of the NFA states from the current DFA state
        // based on the char-ranges of the transitions
        // After that, `trans_partitioning` contains a key for each range of the alphabet that is covered by a transition and no two ranges overlap.
        let mut trans_partitioning = AlphabetPartitionMap::default();
        for &nfa_state_id in nfa_states.iter() {
            let nfa_state = nfa.get_state(nfa_state_id)?;
            for transition in nfa_state.transitions() {
                let rnges = match transition.get_type() {
                    TransitionType::Range(rn) => vec![*rn],
                    TransitionType::NotRange(rn) => rn.complement(),
                    _ => {
                        return Err(AutomatonError::RequiresEpsilonFree(
                            "Determinization".to_owned(),
                        ))
                    }
                };
                let dest = transition.get_dest();
                for rn in rnges {
                    // Refine the partitioning.
                    // If this range overlaps with a range that is already in the partitioning, the partitioning is refined by adding the destination state to the set of the intersecting range.
                    trans_partitioning =
                        trans_partitioning.refine_single(rn, dest.into(), |ldest: &StateSet, _| {
                            // both ranges overlap, so add the destination state to the set of the intersecting range
                            let mut states: StateSet = ldest.clone();
                            states.insert(dest);
                            states
                        });
                }
            }
        }

        for (range, nfa_states) in trans_partitioning.iter() {
            //in trans_partition {
            // Get the DFA state corresponding to the destination NFA states
            if !nfa_states.is_empty() {
                let dfa_dest_state_id = if let Some(&state_id) = state_map.get(nfa_states) {
                    state_id
                } else {
                    let new_dfa_state = dfa.new_state();
                    if nfa_states.iter().any(|&s| nfa.finals.contains(&s)) {
                        dfa.finals.insert(new_dfa_state);
                    }
                    state_map.insert(nfa_states.clone(), new_dfa_state);
                    queue.push_back(nfa_states.clone());
                    new_dfa_state
                };
                // Add a transition from the current DFA state to the destination DFA state for all ranges in the alphabet
                dfa.get_state_mut(dfa_state_id)?
                    .add_transition(Transition::range(*range, dfa_dest_state_id));
            }
        }
    }

    Ok(dfa)
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_determinize_deterministic() {
        // s0 --a--> s1 --b--> s2
        let mut nfa: NFA = Automaton::new();
        let s0 = nfa.new_state();
        let s1 = nfa.new_state();
        let s2 = nfa.new_state();

        nfa.initial = Some(s0);
        nfa.finals.insert(s2);

        nfa.get_state_mut(s0)
            .unwrap()
            .add_transition(Transition::char('a', s1));
        nfa.get_state_mut(s1)
            .unwrap()
            .add_transition(Transition::char('b', s2));

        let dfa = determinize(&nfa).unwrap();

        assert_eq!(dfa.states.len(), 3);
        assert!(dfa.initial.is_some());
        assert_eq!(dfa.finals.len(), 1);

        let initial_state = dfa.initial.unwrap();
        let final_state = *dfa.finals.iter().next().unwrap();

        assert!(dfa.get_state(initial_state).is_ok());
        assert!(dfa.get_state(final_state).is_ok());

        let initial_transitions = dfa.get_state(initial_state).unwrap().transitions();
        assert_eq!(initial_transitions.len(), 1);
        assert_eq!(
            *initial_transitions[0].get_type(),
            TransitionType::char('a')
        );

        let middle_state_id = initial_transitions[0].get_dest();
        let middle_transitions = dfa.get_state(middle_state_id).unwrap().transitions();
        assert_eq!(middle_transitions.len(), 1);
        assert_eq!(*middle_transitions[0].get_type(), TransitionType::char('b'));
        assert_eq!(middle_transitions[0].get_dest(), final_state);
    }
}
