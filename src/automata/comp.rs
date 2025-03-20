//! Complementation of an automaton.
//! The complement of an automaton is an automaton that recognizes precisely those strings that are not recognized by the original automaton.

use super::{det::determinize, NFA};

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
    det.finals = det.states().filter(|s| !det.finals.contains(s)).collect();
    det
}
