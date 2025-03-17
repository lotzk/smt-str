use std::collections::{HashMap, HashSet, VecDeque};

use super::{Automaton, AutomatonError, State, StateId, Transition, TransitionType};

/// Calculates the intersection of this automaton with another automaton and returns the result.
/// Use the product construction to calculate the intersection of two automatons.
/// Both automatons must be epsilon-free. Returns an error if any of the automatons contains epsilon transitions.
pub fn intersect<S: State>(
    a1: &Automaton<S>,
    a2: &Automaton<S>,
) -> Result<Automaton<S>, AutomatonError> {
    // The product automaton
    let mut aut = Automaton::new();

    // The states of the product automaton.
    // Each key is a pair of states from the two automatons, and the value is the corresponding state in the product automaton.
    let mut state_pairs: HashMap<(StateId, StateId), StateId> = HashMap::new();

    // Do a breadth-first search on the product of the two automata, starting from the initial states of both automata
    let mut queue = VecDeque::new();

    if let (Some(q01), Some(q02)) = (a1.initial, a2.initial) {
        let initial = aut.add_state(S::default());
        state_pairs.insert((q01, q02), initial);
        aut.initial = Some(initial);
        queue.push_back((q01, q02));
    }

    // Invariant: The new state must have been created before, when the pair of states was added to the queue
    while let Some((s1id, s2id)) = queue.pop_front() {
        let s1 = a1.get_state(s1id)?;
        let s2 = a2.get_state(s2id)?;

        // Fetch the new state represented by the pair of states in the argument automata. This entry must exist, as it was created when the pair of states was added to the queue, so it is safe to unwrap here.
        let &new_state = state_pairs.get(&(s1id, s2id)).unwrap();

        // If both states are final states, the new state is a final state of the product automaton
        if a1.finals.contains(&s1id) && a2.finals.contains(&s2id) {
            aut.finals.insert(new_state);
        }
        // Find all transitions from the product state
        for t1 in s1.transitions().iter() {
            for t2 in s2.transitions().iter() {
                let ranges_left = match t1.get_type() {
                    TransitionType::Range(r) => vec![*r],
                    TransitionType::NotRange(r) => r.complement(),
                    _ => {
                        return Err(AutomatonError::RequiresEpsilonFree(
                            "Intersection".to_string(),
                        ))
                    }
                };
                let ranges_right = match t2.get_type() {
                    TransitionType::Range(r) => vec![*r],
                    TransitionType::NotRange(r) => r.complement(),
                    _ => {
                        return Err(AutomatonError::RequiresEpsilonFree(
                            "Intersection".to_string(),
                        ))
                    }
                };
                let mut intersections =
                    HashSet::with_capacity(ranges_left.len() * ranges_right.len());
                for r1 in ranges_left.iter() {
                    for r2 in ranges_right.iter() {
                        let inter = r1.intersect(r2);
                        if !inter.is_empty() {
                            intersections.insert(inter);
                        }
                    }
                }
                for inter in intersections {
                    // Add the intersection to the product automaton
                    let dest_pair = (t1.get_dest(), t2.get_dest());
                    let new_dest = match state_pairs.get(&dest_pair) {
                        Some(id) => *id,
                        None => {
                            // Create a new state for the pair of states
                            let new_id = aut.add_state(S::default());
                            state_pairs.insert(dest_pair, new_id);
                            // Enqueue the new pair of states to be processed
                            queue.push_back(dest_pair);
                            new_id
                        }
                    };
                    // Add the transition to the product automaton
                    aut.get_state_mut(new_state)?
                        .add_transition(Transition::range(inter, new_dest));
                }
            }
        }
    }

    Ok(aut)
}

#[cfg(test)]
mod tests {

    use crate::{alphabet::CharRange, automata::NFA};

    use super::*;

    #[test]
    fn test_basic_intersection() {
        let mut a1: NFA = Automaton::new();
        let s0_a1 = a1.new_state();
        let s1_a1 = a1.new_state();
        a1.initial = Some(s0_a1);
        a1.finals.insert(s1_a1);
        a1.get_state_mut(s0_a1)
            .unwrap()
            .add_transition(Transition::range_from('a', 'b', s1_a1));

        let mut a2: NFA = Automaton::new();
        let s0_a2 = a2.new_state();
        let s1_a2 = a2.new_state();
        a2.initial = Some(s0_a2);
        a2.finals.insert(s1_a2);
        a2.get_state_mut(s0_a2)
            .unwrap()
            .add_transition(Transition::range_from('b', 'c', s1_a2));

        let result = intersect(&a1, &a2).unwrap();

        assert_eq!(result.states.len(), 2);
        assert!(result.initial.is_some());
        assert_eq!(result.finals.len(), 1);

        let initial_state = result.initial.unwrap();
        let final_state = *result.finals.iter().next().unwrap();

        let initial_transitions = &result.get_state(initial_state).unwrap().transitions();
        assert_eq!(initial_transitions.len(), 1);
        assert_eq!(
            *initial_transitions[0].get_type(),
            TransitionType::char('b',)
        );
        assert_eq!(initial_transitions[0].get_dest(), final_state);
    }

    #[test]
    fn test_no_intersection() {
        let mut a1: NFA = Automaton::new();
        let s0_a1 = a1.new_state();
        let s1_a1 = a1.new_state();
        a1.initial = Some(s0_a1);
        a1.finals.insert(s1_a1);
        a1.get_state_mut(s0_a1)
            .unwrap()
            .add_transition(Transition::char('a', s1_a1));

        let mut a2: NFA = Automaton::new();
        let s0_a2 = a2.new_state();
        let s1_a2 = a2.new_state();
        a2.initial = Some(s0_a2);
        a2.finals.insert(s1_a2);
        a2.get_state_mut(s0_a2)
            .unwrap()
            .add_transition(Transition::char('b', s1_a2));

        let result = intersect(&a1, &a2).unwrap();

        assert_eq!(result.finals.len(), 0);
    }

    #[test]
    fn test_intersection_with_epsilon_transition() {
        let mut a1: NFA = Automaton::new();
        let s0_a1 = a1.new_state();
        let s1_a1 = a1.new_state();
        a1.initial = Some(s0_a1);
        a1.finals.insert(s1_a1);
        a1.get_state_mut(s0_a1).unwrap().add_transition(Transition {
            type_: TransitionType::Epsilon,
            destination: s1_a1,
        });

        let mut a2: NFA = Automaton::new();
        let s0_a2 = a2.new_state();
        let s1_a2 = a2.new_state();
        a2.initial = Some(s0_a2);
        a2.finals.insert(s1_a2);
        a2.get_state_mut(s0_a2)
            .unwrap()
            .add_transition(Transition::char('a', s1_a2));

        let result = intersect(&a1, &a2);

        assert!(result.is_err());
        if let Err(AutomatonError::RequiresEpsilonFree(message)) = result {
            assert_eq!(message, "Intersection");
        } else {
            panic!("Expected RequiresEpsilonFree error");
        }
    }

    #[test]
    fn test_range_intersection() {
        let mut a1: NFA = Automaton::new();
        let s0_a1 = a1.new_state();
        let s1_a1 = a1.new_state();
        a1.initial = Some(s0_a1);
        a1.finals.insert(s1_a1);
        a1.get_state_mut(s0_a1)
            .unwrap()
            .add_transition(Transition::range_from('a', 'd', s1_a1));

        let mut a2: NFA = Automaton::new();
        let s0_a2 = a2.new_state();
        let s1_a2 = a2.new_state();
        a2.initial = Some(s0_a2);
        a2.finals.insert(s1_a2);
        a2.get_state_mut(s0_a2)
            .unwrap()
            .add_transition(Transition::range_from('c', 'f', s1_a2));

        let result = intersect(&a1, &a2).unwrap();

        assert_eq!(result.states.len(), 2);
        assert!(result.initial.is_some());
        assert_eq!(result.finals.len(), 1);

        let initial_state = result.initial.unwrap();
        let final_state = *result.finals.iter().next().unwrap();

        let initial_transitions = &result.get_state(initial_state).unwrap().transitions();
        assert_eq!(initial_transitions.len(), 1);
        assert_eq!(
            *initial_transitions[0].get_type(),
            TransitionType::Range(CharRange::new('c', 'd'))
        );
        assert_eq!(initial_transitions[0].get_dest(), final_state);
    }
}
