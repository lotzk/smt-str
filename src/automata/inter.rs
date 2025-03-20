//! Computation of the intersection of two NFAs.
//! The intersection of two NFAs is an NFA that accepts the intersection of the languages accepted by the two NFAs.
//! The intersection is computed by constructing a product automaton.
//! The product automaton has a state for each pair of states from the two input automata.
//! The transitions are constructed by taking the cartesian product of the transitions of the two input automata.

use std::collections::VecDeque;

use indexmap::IndexMap;

use super::{StateId, Transition, TransitionType, NFA};

/// Computes the intersection of two NFAs.
/// The intersection of two NFAs is an NFA that accepts the intersection of the languages accepted by the two NFAs.
/// The intersection is computed by constructing a product automaton, which has at most |N| * |M| states, where N and M are the number of states of the input automata.
pub fn intersect(n: &NFA, m: &NFA) -> NFA {
    let mut result = NFA::new();
    // Maps pairs of states from the input automata to the corresponding state in the result automaton.
    let mut state_map = IndexMap::new();
    // Queue of pairs of states from the input automata that need to be processed.
    let mut queue = VecDeque::new();

    // Add the initial state of the product automaton.
    if let Some((n_start, m_start)) = n.initial().zip(m.initial()) {
        let start = result.new_state();
        result.set_initial(start).unwrap(); // this cannot be an error, as the automaton is empty
        state_map.insert((n_start, m_start), start);
        queue.push_back((n_start, m_start));
    }

    // Process the queue.
    while let Some((n_state, m_state)) = queue.pop_front() {
        // Add the transitions from the product state to the product states that correspond to the transitions of the input automata.
        let mapped_state = *state_map.get(&(n_state, m_state)).unwrap();
        // If both states are final, the product state is final.
        if n.is_final(n_state) && m.is_final(m_state) {
            result.add_final(mapped_state).unwrap();
        }
        for t1 in n.transitions_from(n_state).unwrap() {
            for t2 in m.transitions_from(m_state).unwrap() {
                let intersections = intersect_transitions(n_state, t1, m_state, t2);
                for (label, (p1, p2)) in intersections {
                    let new_dest = state_map.entry((p1, p2)).or_insert_with(|| {
                        let new_state = result.new_state();
                        queue.push_back((p1, p2));
                        new_state
                    });
                    result
                        .add_transition(mapped_state, *new_dest, label)
                        .unwrap();
                }
            }
        }
    }
    result.trim()
}

/// Intersects the two transitions
///
/// - q1: the source state of the first transition
/// - t1: the first transition leaving the source state q1
/// - q2: the source state of the second transition
/// - t2: the second transition leaving the source state q2
///
/// Return pairs of the type of the intersection and the destination states, given as pairs of state ids in the input automata.
fn intersect_transitions(
    q1: StateId,
    t1: &Transition,
    q2: StateId,
    t2: &Transition,
) -> Vec<(TransitionType, (StateId, StateId))> {
    let type1 = t1.label;
    let type2 = t2.label;
    let p1 = t1.destination;
    let p2 = t2.destination;

    let mut res = Vec::new();

    match (type1, type2) {
        (TransitionType::Range(r1), TransitionType::Range(r2)) => {
            let inter = r1.intersect(&r2);
            if !inter.is_empty() {
                res.push((TransitionType::Range(inter), (p1, p2)));
            }
        }
        (TransitionType::Range(r2), TransitionType::NotRange(r))
        | (TransitionType::NotRange(r), TransitionType::Range(r2)) => {
            let comp = r.complement();
            for c in comp {
                let inter = c.intersect(&r2);
                if !inter.is_empty() {
                    res.push((TransitionType::Range(inter), (p1, p2)));
                }
            }
        }
        (TransitionType::NotRange(r1), TransitionType::NotRange(r2)) => {
            for r1 in r1.complement() {
                for r2 in r2.complement() {
                    let inter = r1.intersect(&r2);
                    if !inter.is_empty() {
                        res.push((TransitionType::Range(inter), (p1, p2)));
                    }
                }
            }
        }
        (TransitionType::Epsilon, _) => {
            // We stay in the same state in the second automaton and move to p1 when reading an epsilon.
            res.push((TransitionType::Epsilon, (p1, q2)));
        }
        (_, TransitionType::Epsilon) => {
            // We stay in the same state in the first automaton and move to p2 when reading an epsilon.
            res.push((TransitionType::Epsilon, (q1, p2)));
        }
    };
    res
}

#[cfg(test)]
mod tests {

    use quickcheck_macros::quickcheck;

    use crate::alphabet::CharRange;

    use super::*;

    #[test]
    fn test_intersect_transitions_disjoint() {
        let q1 = 0;
        let q2 = 1;
        let p1 = 2;
        let p2 = 3;
        let t1 = Transition::new(TransitionType::Range(CharRange::new('a', 'c')), p1);
        let t2 = Transition::new(TransitionType::Range(CharRange::new('d', 'e')), p2);

        let res = intersect_transitions(q1, &t1, q2, &t2);
        assert_eq!(res.len(), 0);
    }

    #[test]
    fn test_intersect_transitions_overlap() {
        let q1 = 0;
        let q2 = 1;
        let p1 = 2;
        let p2 = 3;
        let t1 = Transition::new(TransitionType::Range(CharRange::new('a', 'g')), p1);
        let t2 = Transition::new(TransitionType::Range(CharRange::new('d', 'k')), p2);

        let res = intersect_transitions(q1, &t1, q2, &t2);
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].0, TransitionType::Range(CharRange::new('d', 'g')));
    }

    #[test]
    fn test_intersect_transitions_equal() {
        let q1 = 0;
        let q2 = 1;
        let p1 = 2;
        let p2 = 3;
        let t1 = Transition::new(TransitionType::Range(CharRange::new('a', 'g')), p1);
        let t2 = Transition::new(TransitionType::Range(CharRange::new('a', 'g')), p2);

        let res = intersect_transitions(q1, &t1, q2, &t2);
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].0, TransitionType::Range(CharRange::new('a', 'g')));
    }

    #[quickcheck]
    fn test_intersect_transitions_ranges(r1: CharRange, r2: CharRange) {
        let q1 = 0;
        let q2 = 1;
        let p1 = 2;
        let p2 = 3;
        let t1 = Transition::new(TransitionType::Range(r1), p1);
        let t2 = Transition::new(TransitionType::Range(r2), p2);

        let res = intersect_transitions(q1, &t1, q2, &t2);
        if r1.intersect(&r2).is_empty() {
            assert_eq!(res.len(), 0);
        } else {
            assert_eq!(res.len(), 1);
            assert_eq!(res[0].0, TransitionType::Range(r1.intersect(&r2)));
            assert_eq!(res[0].1, (p1, p2));
        }
    }

    #[test]
    fn test_intersect_transitions_epsilon() {
        let q1 = 0;
        let q2 = 1;
        let p1 = 2;
        let p2 = 3;
        let t1 = Transition::new(TransitionType::Epsilon, p1);
        let t2 = Transition::new(TransitionType::Range(CharRange::new('a', 'z')), p2);

        let res = intersect_transitions(q1, &t1, q2, &t2);
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].0, TransitionType::Epsilon);
        assert_eq!(res[0].1, (p1, q2));
    }

    #[test]
    fn test_intersect_empty() {
        let a1: NFA = NFA::new();
        let a2: NFA = NFA::new();
        let result = intersect(&a1, &a2);
        assert_eq!(result.states.len(), 0);
        assert!(result.initial.is_none());
        assert_eq!(result.finals.len(), 0);
    }

    #[test]
    fn test_intersection_simple() {
        let mut nfa1 = NFA::new();
        let q0 = nfa1.new_state();
        let q1 = nfa1.new_state();
        nfa1.set_initial(q0).unwrap();
        nfa1.add_final(q1).unwrap();
        nfa1.add_transition(q0, q1, TransitionType::Range(CharRange::new('a', 'b')))
            .unwrap();

        let mut nfa2 = NFA::new();
        let p0 = nfa2.new_state();
        let p1 = nfa2.new_state();
        nfa2.set_initial(p0).unwrap();
        nfa2.add_final(p1).unwrap();
        nfa2.add_transition(p0, p1, TransitionType::Range(CharRange::new('b', 'c')))
            .unwrap();

        let result = intersect(&nfa1, &nfa2);

        assert!(result.accepts(&"b".into()));
        assert!(!result.accepts(&"a".into()));
        assert!(!result.accepts(&"c".into()));
    }

    #[test]
    fn test_intersection_epsilon() {
        let mut nfa1 = NFA::new();
        let q0 = nfa1.new_state();
        let q1 = nfa1.new_state();
        nfa1.set_initial(q0).unwrap();
        nfa1.add_final(q1).unwrap();
        nfa1.add_transition(q0, q1, TransitionType::Epsilon)
            .unwrap();

        let mut nfa2 = NFA::new();
        let p0 = nfa2.new_state();
        let p1 = nfa2.new_state();
        nfa2.set_initial(p0).unwrap();
        nfa2.add_final(p1).unwrap();
        nfa2.add_transition(p0, p1, TransitionType::Range(CharRange::new('a', 'z')))
            .unwrap();

        let result = intersect(&nfa1, &nfa2);

        assert!(!result.accepts(&"a".into()));
        assert!(!result.accepts(&"".into()));
    }

    #[test]
    fn test_intersection_negated_range() {
        let mut nfa1 = NFA::new();
        let q0 = nfa1.new_state();
        let q1 = nfa1.new_state();
        nfa1.set_initial(q0).unwrap();
        nfa1.add_final(q1).unwrap();
        nfa1.add_transition(q0, q1, TransitionType::NotRange(CharRange::new('a', 'z')))
            .unwrap();

        let mut nfa2 = NFA::new();
        let p0 = nfa2.new_state();
        let p1 = nfa2.new_state();
        nfa2.set_initial(p0).unwrap();
        nfa2.add_final(p1).unwrap();
        nfa2.add_transition(p0, p1, TransitionType::Range(CharRange::new('0', '9')))
            .unwrap();

        let result = intersect(&nfa1, &nfa2);

        assert!(result.accepts(&"0".into()));
        assert!(result.accepts(&"9".into()));
        assert!(!result.accepts(&"a".into()));
        assert!(!result.accepts(&"z".into()));
    }

    #[test]
    fn test_intersection_disjoint() {
        let mut nfa1 = NFA::new();
        let q0 = nfa1.new_state();
        let q1 = nfa1.new_state();
        nfa1.set_initial(q0).unwrap();
        nfa1.add_final(q1).unwrap();
        nfa1.add_transition(q0, q1, TransitionType::Range(CharRange::new('a', 'b')))
            .unwrap();

        let mut nfa2 = NFA::new();
        let p0 = nfa2.new_state();
        let p1 = nfa2.new_state();
        nfa2.set_initial(p0).unwrap();
        nfa2.add_final(p1).unwrap();
        nfa2.add_transition(p0, p1, TransitionType::Range(CharRange::new('c', 'd')))
            .unwrap();

        let result = intersect(&nfa1, &nfa2);

        assert!(!result.accepts(&"a".into()));
        assert!(!result.accepts(&"b".into()));
        assert!(!result.accepts(&"c".into()));
        assert!(!result.accepts(&"d".into()));
    }

    #[test]
    fn test_intersection_multiple_steps() {
        let mut nfa1 = NFA::new();
        let q0 = nfa1.new_state();
        let q1 = nfa1.new_state();
        let q2 = nfa1.new_state();
        nfa1.set_initial(q0).unwrap();
        nfa1.add_final(q2).unwrap();
        nfa1.add_transition(q0, q1, TransitionType::Range(CharRange::new('a', 'a')))
            .unwrap();
        nfa1.add_transition(q1, q2, TransitionType::Range(CharRange::new('b', 'b')))
            .unwrap();

        let mut nfa2 = NFA::new();
        let p0 = nfa2.new_state();
        let p1 = nfa2.new_state();
        let p2 = nfa2.new_state();
        nfa2.set_initial(p0).unwrap();
        nfa2.add_final(p2).unwrap();
        nfa2.add_transition(p0, p1, TransitionType::Range(CharRange::new('a', 'a')))
            .unwrap();
        nfa2.add_transition(p1, p2, TransitionType::Range(CharRange::new('b', 'b')))
            .unwrap();

        let result = intersect(&nfa1, &nfa2);

        assert!(result.accepts(&"ab".into()));
        assert!(!result.accepts(&"a".into()));
        assert!(!result.accepts(&"b".into()));
        assert!(!result.accepts(&"ba".into()));
    }
}
