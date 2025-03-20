//! Complementation of an automaton.
//! The complement of an automaton is an automaton that recognizes precisely those strings that are not recognized by the original automaton.

use crate::alphabet::{Alphabet, CharRange};

use super::{det::determinize, TransitionType, NFA};

/// Compute the complement of an NFA.
/// The resulting NFA accepts all string that are not accepted by the input NFA and rejects all strings that are accepted by the input NFA.
/// This is achieved by swapping the final and non-final states of the input automaton.
/// This requires that the automaton is deterministic. If the input automaton is not deterministic, it will be determinized first.
/// If the the input automaton is already deterministic, the algorithm takes O(n) time, where n is the number of states in the automaton.
/// If the input automaton is not deterministic, the algorithm takes another O(2^n) time for determinization, where n is the number of states in the automaton.
pub fn complement(nfa: &NFA) -> NFA {
    let mut det = if !nfa.is_det() {
        determinize(nfa)
    } else {
        nfa.clone()
    };

    if det.is_empty() {
        return NFA::universal();
    }

    // Additionally, we need to add a new sink that we transition to when the dfa reaches a dead end.
    let sink = det.new_state();
    det.add_transition(sink, sink, TransitionType::Range(CharRange::all()))
        .unwrap(); // Cycle on all characters

    // For each state, compute the alphabet of character for which no move is possible.
    // Then add a transition to the sink for all of these characters.
    for state in det.states() {
        let mut alph = Alphabet::default();
        for t in det.transitions_from(state).unwrap() {
            match t.get_type() {
                &TransitionType::Range(r) => alph.insert(r),
                &TransitionType::NotRange(r) => {
                    r.complement().into_iter().for_each(|r| alph.insert(r))
                }
                TransitionType::Epsilon => unreachable!("DFA cannot have epsilon transitions"),
            }
        }
        // Complement the alphabet to get the alphabet of characters for which no move is possible.
        let comp = alph.complement();
        for r in comp.iter_ranges() {
            det.add_transition(state, sink, TransitionType::Range(r))
                .unwrap();
        }
    }

    det.finals = det.states().filter(|s| !det.finals.contains(s)).collect();
    det.trim()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complement_basic() {
        let mut nfa = NFA::new();
        let q0 = nfa.new_state();
        let q1 = nfa.new_state();

        nfa.set_initial(q0).unwrap();
        nfa.add_final(q1).unwrap();

        nfa.add_transition(q0, q1, TransitionType::Range(CharRange::new('a', 'a')))
            .unwrap();

        let complement_dfa = complement(&nfa);

        assert!(!complement_dfa.accepts(&"a".into())); // Previously accepted, now rejected
        assert!(complement_dfa.accepts(&"b".into())); // Previously rejected, now accepted
        assert!(complement_dfa.accepts(&"z".into())); // Anything except "a" should be accepted
    }

    #[test]
    fn test_complement_handles_determinization() {
        let mut nfa = NFA::new();
        let q0 = nfa.new_state();
        let q1 = nfa.new_state();
        let q2 = nfa.new_state();

        nfa.set_initial(q0).unwrap();
        nfa.add_final(q2).unwrap();

        // Non-deterministic transitions
        nfa.add_transition(q0, q1, TransitionType::Range(CharRange::new('a', 'a')))
            .unwrap();
        nfa.add_transition(q0, q2, TransitionType::Range(CharRange::new('a', 'a')))
            .unwrap();

        let complement_dfa = complement(&nfa);

        assert!(!complement_dfa.accepts(&"a".into())); // "a" was accepted in the original
        assert!(complement_dfa.accepts(&"b".into())); // Previously rejected
    }

    #[test]
    fn test_complement_universal_automaton() {
        let nfa = NFA::universal();

        let complement_dfa = complement(&nfa).trim();
        assert_eq!(complement_dfa.num_states(), 0); // Complement of a universal automaton should be empty
    }

    #[test]
    fn test_complement_empty_automaton() {
        let nfa = NFA::new(); // Empty automaton

        let complement_dfa = complement(&nfa);

        assert!(complement_dfa.num_states() == 1); // Should only have the sink state
        assert!(complement_dfa.accepts(&"x".into())); // Everything should be accepted
    }

    #[test]
    fn test_complement_handles_sink_correctly() {
        let mut nfa = NFA::new();
        let q0 = nfa.new_state();
        let q1 = nfa.new_state();

        nfa.set_initial(q0).unwrap();
        nfa.add_final(q1).unwrap();

        nfa.add_transition(q0, q1, TransitionType::Range(CharRange::new('x', 'x')))
            .unwrap();

        let complement_dfa = complement(&nfa);

        assert!(!complement_dfa.accepts(&"x".into())); // "x" should now be rejected
        assert!(complement_dfa.accepts(&"y".into())); // Anything except "x" should be accepted
    }

    #[test]
    fn test_complement_multiple_transitions() {
        let mut nfa = NFA::new();
        let q0 = nfa.new_state();
        let q1 = nfa.new_state();
        let q2 = nfa.new_state();

        nfa.set_initial(q0).unwrap();
        nfa.add_final(q2).unwrap();

        nfa.add_transition(q0, q1, TransitionType::Range(CharRange::new('a', 'c')))
            .unwrap();
        nfa.add_transition(q1, q2, TransitionType::Range(CharRange::new('d', 'f')))
            .unwrap();

        let complement_dfa = complement(&nfa);

        assert!(!complement_dfa.accepts(&"ad".into())); // Previously accepted
        assert!(complement_dfa.accepts(&"az".into())); // Previously rejected
        assert!(complement_dfa.accepts(&"bq".into())); // Previously rejected
    }
}
