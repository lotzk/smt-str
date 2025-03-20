//! Determinization of non-deterministic finite automata.
//! An automatons is deterministic if for each state and each character in the alphabet there is at most one transition.

use std::{
    collections::{HashMap, VecDeque},
    fmt::Display,
};

use bit_set::BitSet;
use indexmap::IndexMap;

use crate::alphabet::AlphabetPartitionMap;

use super::{StateId, TransitionType, NFA};

/// A set of states. Each set of states corresponds to a single state in the determinized automaton.
/// The set is implemented as a BTreeSet of state IDs contained in the set.
#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct StateSet(BitSet<StateId>);
impl StateSet {
    /// Creates a new empty state set.
    fn new() -> Self {
        Self(BitSet::default())
    }

    /// Inserts a state into the set.
    fn insert(&mut self, state: StateId) {
        self.0.insert(state);
    }

    /// Returns an iterator over the state IDs in the set.
    fn iter(&self) -> impl Iterator<Item = StateId> + '_ {
        self.0.iter()
    }

    /// Returns true if the set is empty.
    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Extends the set with the elements of another set.
    fn extend(&mut self, other: &StateSet) {
        self.0.union_with(&other.0);
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

/// Determinizes an NFA.
/// The result is an automaton recognizing the same language as the input NFA.
/// Returns an NFA struct that describes the deterministic finite automaton.
///
/// The function uses the subset construction algorithm to determinize the NFA.
/// As a result, the number of states in the resulting automaton can be exponential in the number of states of the input NFA.
pub fn determinize(nfa: &NFA) -> NFA {
    let mut det = NFA::new();
    // Maps a set of NFA states to a DFA state
    let mut state_map: IndexMap<StateSet, StateId> = IndexMap::new();
    // The queue of states to process
    let mut queue: VecDeque<StateSet> = VecDeque::new();

    let mut epsilon_cache: HashMap<StateId, StateSet> = HashMap::new(); // Cache epsilon closures

    // Compute epsilon closures for all states
    for q in nfa.states() {
        let mut closure = StateSet::new();
        // Safe to unwrap because q is guaranteed to be a valid state ID
        for p in nfa.epsilon_closure(q).unwrap() {
            closure.insert(p);
        }
        epsilon_cache.insert(q, closure);
    }

    // Compute the closure of the initial NFA state
    if let Some(q0) = nfa.initial() {
        let initial_set = epsilon_cache.get(&q0).unwrap().clone();
        let q0_dfa = det.new_state();
        det.set_initial(q0_dfa).unwrap();
        if initial_set.iter().any(|q| nfa.is_final(q)) {
            det.add_final(q0_dfa).unwrap();
        }
        state_map.insert(initial_set.clone(), q0_dfa);
        queue.push_back(initial_set);
    }

    // Process the states in the queue
    while let Some(nfa_states) = queue.pop_front() {
        let dfa_state = *state_map.get(&nfa_states).unwrap();

        let trans_partitioning = partition_transitions(&nfa_states, nfa);

        // Now each partition in `trans_partitioning` corresponds to a transition in the DFA.
        for (range, nfa_states) in trans_partitioning
            .into_iter()
            .filter(|(r, s)| !r.is_empty() && !s.is_empty())
        {
            // Compute the epsilon closure of the destination set using the precomputed cache
            let mut nfa_states_closure = StateSet::new();
            for q in nfa_states.iter() {
                if let Some(closure) = epsilon_cache.get(&q) {
                    nfa_states_closure.extend(closure);
                }
            }
            let dest = *state_map
                .entry(nfa_states_closure.clone())
                .or_insert_with(|| {
                    let new_state = det.new_state();
                    if nfa_states_closure.iter().any(|q| nfa.is_final(q)) {
                        det.add_final(new_state).unwrap();
                    }
                    queue.push_back(nfa_states_closure.clone());
                    new_state
                });
            det.add_transition(dfa_state, dest, TransitionType::Range(range))
                .unwrap();
        }
    }
    det.trim()
}

/// Collect transitions for each symbol by partitioning the transitions of the NFA states from the state set on the char-ranges of the transitions
/// Returns a partitioning map that contains a key for each range of the alphabet that is covered by a transition and no two ranges overlap.
fn partition_transitions(states: &StateSet, nfa: &NFA) -> AlphabetPartitionMap<StateSet> {
    let mut trans_partitioning: AlphabetPartitionMap<StateSet> = AlphabetPartitionMap::default();
    for q in states.iter() {
        for transition in nfa.transitions_from(q).unwrap() {
            let ranges = match transition.get_type() {
                TransitionType::Range(r) => vec![*r],
                TransitionType::NotRange(nr) => nr.complement(),
                TransitionType::Epsilon => continue,
            };
            let p = transition.get_dest();
            for range in ranges {
                trans_partitioning =
                    trans_partitioning.refine_single(range, p.into(), |ldest: &StateSet, _| {
                        let mut new_set = ldest.clone();
                        new_set.insert(p);
                        new_set
                    });
            }
        }
    }
    trans_partitioning
}

#[cfg(test)]
mod tests {
    use crate::alphabet::CharRange;

    use super::*;

    #[test]
    fn test_determinize_empty_nfa() {
        let nfa = NFA::new(); // Empty NFA with no states
        let dfa = determinize(&nfa);

        assert!(dfa.states().next().is_none());
    }

    #[test]
    fn test_determinize_single_state_nfa() {
        let mut nfa = NFA::new();
        let q0 = nfa.new_state();
        nfa.set_initial(q0).unwrap();
        nfa.add_final(q0).unwrap(); // The initial state is also final

        let dfa = determinize(&nfa);
        assert!(dfa.is_det());

        assert!(dfa.accepts(&"".into())); // DFA should accept the empty string
        assert!(!dfa.accepts(&"a".into()));
    }

    #[test]
    fn test_determinize_nfa_with_epsilon() {
        let mut nfa = NFA::new();
        let q0 = nfa.new_state();
        let q1 = nfa.new_state();

        nfa.set_initial(q0).unwrap();
        nfa.add_final(q1).unwrap();

        nfa.add_transition(q0, q1, TransitionType::Epsilon).unwrap();

        let dfa = determinize(&nfa);
        assert!(dfa.is_det());

        assert!(dfa.accepts(&"".into()));
        assert!(!dfa.accepts(&"a".into()));
    }

    #[test]
    fn test_determinize_basic_nfa() {
        let mut nfa = NFA::new();
        let q0 = nfa.new_state();
        let q1 = nfa.new_state();
        let q2 = nfa.new_state();

        nfa.set_initial(q0).unwrap();
        nfa.add_final(q2).unwrap();

        nfa.add_transition(q0, q1, TransitionType::Range(CharRange::singleton('a')))
            .unwrap();
        nfa.add_transition(q1, q2, TransitionType::Range(CharRange::singleton('b')))
            .unwrap();

        let dfa = determinize(&nfa);
        assert!(dfa.is_det());

        assert!(dfa.accepts(&"ab".into()));
        assert!(!dfa.accepts(&"a".into()));
        assert!(!dfa.accepts(&"b".into()));
        assert!(!dfa.accepts(&"ba".into()));
    }

    #[test]
    fn test_determinize_nfa_with_multiple_paths() {
        let mut nfa = NFA::new();
        let q0 = nfa.new_state();
        let q1 = nfa.new_state();
        let q2 = nfa.new_state();
        let q3 = nfa.new_state();

        nfa.set_initial(q0).unwrap();
        nfa.add_final(q3).unwrap();

        nfa.add_transition(q0, q1, TransitionType::Range(CharRange::new('a', 'a')))
            .unwrap();
        nfa.add_transition(q0, q2, TransitionType::Range(CharRange::new('a', 'a')))
            .unwrap();
        nfa.add_transition(q1, q3, TransitionType::Range(CharRange::new('b', 'b')))
            .unwrap();
        nfa.add_transition(q2, q3, TransitionType::Range(CharRange::new('c', 'c')))
            .unwrap();

        let dfa = determinize(&nfa);
        assert!(dfa.is_det());

        assert!(dfa.accepts(&"ab".into()));
        assert!(dfa.accepts(&"ac".into()));
        assert!(!dfa.accepts(&"a".into()));
        assert!(!dfa.accepts(&"bc".into()));
    }

    #[test]
    fn test_determinize_nfa_with_overlapping_ranges() {
        let mut nfa = NFA::new();
        let q0 = nfa.new_state();
        let q1 = nfa.new_state();
        let q2 = nfa.new_state();

        // -> q0 --[a-c]--> q1*
        //       --[b-d]--> q2*

        nfa.set_initial(q0).unwrap();
        nfa.add_final(q1).unwrap();
        nfa.add_final(q2).unwrap();

        nfa.add_transition(q0, q1, TransitionType::Range(CharRange::new('a', 'c')))
            .unwrap();
        nfa.add_transition(q0, q2, TransitionType::Range(CharRange::new('b', 'd')))
            .unwrap();

        let dfa = determinize(&nfa);
        assert_eq!(dfa.num_states(), 4);
        assert!(dfa.is_det());

        assert!(dfa.accepts(&"a".into()));
        assert!(dfa.accepts(&"b".into()));
        assert!(dfa.accepts(&"c".into()));
        assert!(dfa.accepts(&"d".into()));
        assert!(!dfa.accepts(&"e".into()));
    }
}
