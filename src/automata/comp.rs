use std::collections::HashSet;

use crate::alphabet::Alphabet;

use super::{Automaton, AutomatonError, State, StateId, Transition, TransitionType, DFA};

/// Calculates the complement of the given automaton.
/// The complement of an automaton is the automaton that accepts the language that is not accepted by the given automaton.
pub fn complement(a: &DFA) -> Result<DFA, AutomatonError> {
    // If the automaton accepts no words, the complement is the automaton that accepts all words.
    if a.is_empty() {
        let mut comp: DFA = Automaton::new();
        let s0 = comp.new_state();
        comp.initial = Some(s0);
        comp.finals.insert(s0);
        comp.get_state_mut(s0)
            .unwrap()
            .add_transition(Transition::any(s0));
        Ok(comp)
    } else {
        do_complement(a)
    }
}

/// Actual implementation of the complement function.
/// Inverts the final states and adds a sink state that is transitioned to when the automaton reaches a dead end.
fn do_complement(a: &DFA) -> Result<DFA, AutomatonError> {
    let mut complemented = a.clone();
    let all_states: HashSet<StateId> = (0..complemented.states.len()).collect();
    let inverted_finals = all_states
        .difference(&complemented.finals)
        .cloned()
        .collect();
    complemented.finals = inverted_finals;
    // Additionally, we need to add a new sink that we transition to when the dfa reaches a dead end.
    let sink = complemented.new_state();
    complemented.finals.insert(sink);
    complemented
        .get_state_mut(sink)?
        .add_transition(Transition::any(sink));
    for &state in &all_states {
        let mut alph = Alphabet::default();
        for t in complemented.get_state(state)?.transitions().iter() {
            match t.get_type() {
                &TransitionType::Range(r) => alph.insert(r),
                &TransitionType::NotRange(r) => {
                    r.complement().into_iter().for_each(|r| alph.insert(r))
                }
                TransitionType::Epsilon => unreachable!("DFA cannot have epsilon transitions"),
            }
        }
        let comp = alph.complement();
        for r in comp.iter_ranges() {
            complemented
                .get_state_mut(state)?
                .add_transition(Transition::range(r, sink));
        }
    }

    Ok(complemented)
}

#[cfg(test)]
mod tests {
    use quickcheck_macros::quickcheck;

    use crate::SmtString;

    use super::*;

    #[test]
    fn test_complement_basic() {
        // s0 --a--> s1 --b--> s2
        let mut a: DFA = Automaton::new();
        let s0 = a.new_state();
        let s1 = a.new_state();
        let s2 = a.new_state();

        a.initial = Some(s0);
        a.finals.insert(s2);

        a.get_state_mut(s0)
            .unwrap()
            .add_transition(Transition::char('a', s1));
        a.get_state_mut(s1)
            .unwrap()
            .add_transition(Transition::char('b', s2));

        let complement_dfa = complement(&a).unwrap();

        assert_eq!(complement_dfa.states.len(), 4);
        assert!(complement_dfa.initial.is_some());

        assert!(complement_dfa.accepts(&"a".into()).unwrap());
        assert!(complement_dfa.accepts(&"b".into()).unwrap());
        assert!(!complement_dfa.accepts(&"ab".into()).unwrap());
        assert!(complement_dfa.accepts(&"c".into()).unwrap());
        assert!(complement_dfa.accepts(&"abc".into()).unwrap());
    }

    #[test]
    fn test_complement_with_all_final_states() {
        // s0* --a--> s1*
        let mut a: DFA = Automaton::new();
        let s0 = a.new_state();
        let s1 = a.new_state();

        a.initial = Some(s0);
        a.finals.insert(s0);
        a.finals.insert(s1);

        a.get_state_mut(s0)
            .unwrap()
            .add_transition(Transition::char('a', s1));

        let complement_dfa = complement(&a).unwrap();

        assert_eq!(complement_dfa.states.len(), 3);
        assert!(complement_dfa.initial.is_some());
        assert_eq!(complement_dfa.finals.len(), 1);
    }

    #[quickcheck]
    fn test_complement_with_no_final_states(w: SmtString) {
        let mut a: DFA = Automaton::new();
        let s0 = a.new_state();
        let s1 = a.new_state();

        a.initial = Some(s0);

        a.get_state_mut(s0)
            .unwrap()
            .add_transition(Transition::char('a', s1));

        let complement_dfa = complement(&a).unwrap();

        assert_eq!(complement_dfa.states.len(), 1);
        assert!(complement_dfa.initial.is_some());
        assert_eq!(complement_dfa.finals.len(), 1);
        assert!(complement_dfa.accepts(&w).unwrap());
    }

    #[quickcheck]
    fn test_complement_with_no_initial_state(w: SmtString) {
        let mut a: DFA = Automaton::new();
        let s0 = a.new_state();
        let s1 = a.new_state();

        a.finals.insert(s1);

        a.get_state_mut(s0)
            .unwrap()
            .add_transition(Transition::char('a', s1));

        let complement_dfa = complement(&a).unwrap();

        assert_eq!(complement_dfa.states.len(), 1);
        assert!(complement_dfa.initial.is_some());
        assert_eq!(complement_dfa.finals.len(), 1);
        assert!(complement_dfa.accepts(&w).unwrap());
    }

    #[test]
    fn test_complement_complex() {
        // s0 --a--> s1 --b--> *s2* --c--> s3 --b--> s2
        // ab(cb)*
        let mut a: DFA = Automaton::new();
        let s0 = a.new_state();
        let s1 = a.new_state();
        let s2 = a.new_state();
        let s3 = a.new_state();

        a.initial = Some(s0);
        a.finals.insert(s2);

        a.get_state_mut(s0)
            .unwrap()
            .add_transition(Transition::char('a', s1));
        a.get_state_mut(s1)
            .unwrap()
            .add_transition(Transition::char('b', s2));
        a.get_state_mut(s2)
            .unwrap()
            .add_transition(Transition::char('c', s3));
        a.get_state_mut(s3)
            .unwrap()
            .add_transition(Transition::char('b', s2));

        let complement_dfa = complement(&a).unwrap();

        assert!(complement_dfa.accepts(&"a".into()).unwrap());
        assert!(!complement_dfa.accepts(&"ab".into()).unwrap());
        assert!(complement_dfa.accepts(&"abc".into()).unwrap());
        assert!(!complement_dfa.accepts(&"abcb".into()).unwrap());
        assert!(complement_dfa.accepts(&"abcbc".into()).unwrap());
        assert!(complement_dfa.accepts(&"de".into()).unwrap());
    }
}
