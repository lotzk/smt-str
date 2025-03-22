use bit_set::BitSet;

use rand::{rng, seq::IteratorRandom};

use crate::{
    automata::{TransitionType, NFA},
    re::{deriv::DerivativeBuilder, ReBuilder, Regex},
    SmtString,
};

/// Tries to sample a word that is accepted by the regex.
/// The function aborts if no word is found after `max_depth` steps.
pub fn sample_regex(regex: &Regex, builder: &mut ReBuilder, max_depth: usize) -> Option<SmtString> {
    let mut w = SmtString::empty();
    let mut deriver = DerivativeBuilder::default();

    let mut i = 0;
    let mut re = regex.clone();

    if re.nullable() {
        return Some(w);
    }

    while !re.nullable() && i < max_depth {
        let next = re
            .first()
            .iter()
            .choose(&mut rng())
            .and_then(|c| c.choose())?;
        w.push(next);
        re = deriver.deriv(&re, next, builder);
        i += 1;
    }

    if re.nullable() {
        Some(w)
    } else {
        None
    }
}

/// Tries to sample a word that is accepted or not accepted by the NFA.
/// Randomly picks transitions to follow until a final state is reached.
/// Once a final state is reached, the function returns the word that was sampled.
/// The function aborts and returns `None` if no word is found after `max_depth` transitions.
pub fn sample_nfa(nfa: &NFA, max: usize) -> Option<SmtString> {
    let mut w = SmtString::empty();

    let mut states = BitSet::new();
    if let Some(q0) = nfa.initial() {
        states = BitSet::from_iter(nfa.epsilon_closure(q0).unwrap());
    }
    let mut i = 0;

    while i <= max {
        i += 1;
        // Check if the current state set contains a final state
        if states.iter().any(|q| nfa.is_final(q)) {
            return Some(w);
        }

        // Collect all transitions from the current state set
        let mut transitions = Vec::new();
        for q in states.iter() {
            transitions.extend(nfa.transitions_from(q).unwrap());
        }
        // Pick a random transition
        let transition = transitions.iter().choose(&mut rng())?;
        // Pick a random character from the transition
        let c = match transition.get_type() {
            TransitionType::Range(r) => r.choose(),
            TransitionType::NotRange(nr) => {
                let r = nr.complement();
                r.into_iter()
                    .filter(|r| !r.is_empty())
                    .choose(&mut rng())
                    .and_then(|r| r.choose())
            }
            TransitionType::Epsilon => None,
        };
        match c {
            Some(c) => {
                w.push(c);
                // set the next state set to the epsilon closure of the destination state
                states = BitSet::from_iter(nfa.epsilon_closure(transition.get_dest()).unwrap());
            }
            None => continue,
        }
    }

    None
}

#[cfg(test)]
mod tests {

    use quickcheck_macros::quickcheck;
    use smallvec::smallvec;

    use crate::alphabet::CharRange;

    use super::*;

    #[test]
    fn sample_const() {
        let mut builder = ReBuilder::default();
        let regex = builder.to_re("foo".into());

        assert_eq!(sample_regex(&regex, &mut builder, 3), Some("foo".into()));
        assert_eq!(sample_regex(&regex, &mut builder, 10), Some("foo".into()));
        assert_eq!(sample_regex(&regex, &mut builder, 2), None);
    }

    #[test]
    fn sample_with_optional_characters() {
        let mut builder = ReBuilder::default();

        // fo(o|bar)
        let o = builder.to_re("o".into());
        let fo = builder.to_re("fo".into());
        let bar = builder.to_re("bar".into());
        let o_or_bar = builder.union(smallvec![o, bar]);
        let regex = builder.concat(smallvec![fo, o_or_bar]);

        // Test matching "foo"
        assert!(sample_regex(&regex, &mut builder, 5).is_some());
    }

    #[quickcheck]
    fn sample_with_character_range(range: CharRange) {
        let mut builder = ReBuilder::default();
        let regex = builder.range(range);

        assert!(sample_regex(&regex, &mut builder, 1).is_some());
        // Test matching word within the class
        assert!(sample_regex(&regex, &mut builder, 3).is_some());
    }

    #[quickcheck]
    fn sample_character_range(range: CharRange) {
        let mut builder = ReBuilder::default();
        let regex = builder.range(range);

        assert!(sample_regex(&regex, &mut builder, 1).is_some());
        // Test matching word within the class
        assert!(sample_regex(&regex, &mut builder, 3).is_some());
    }

    #[quickcheck]
    fn sample_character_range_pow(range: CharRange, n: u32) {
        let n = n % 100;
        let mut builder = ReBuilder::default();
        let regex = builder.range(range);
        let regex = builder.pow(regex, n as u32);

        for i in 0..n {
            assert!(sample_regex(&regex, &mut builder, i as usize).is_none());
        }
        assert!(sample_regex(&regex, &mut builder, n as usize).is_some());
    }

    #[quickcheck]
    fn sample_alternatives(rs: Vec<CharRange>) {
        let n = rs.len();
        let mut builder = ReBuilder::default();
        let rs = rs.into_iter().map(|r| builder.range(r)).collect();
        let regex = builder.union(rs);

        if n > 0 {
            assert!(sample_regex(&regex, &mut builder, 1).is_some());
        } else {
            assert!(sample_regex(&regex, &mut builder, 10).is_none());
        }
    }

    #[test]
    fn sampling_alternatives_bug() {
        let rs = vec![
            //CharRange::new(76887, 179877),
            //CharRange::new(142686, 186533),
            //CharRange::new(51684, 146039),
            CharRange::new(2u32, 5u32),
            CharRange::new(3u32, 6u32),
            CharRange::new(1u32, 4u32),
        ];

        //  CharRange  CharRange { start: SmtChar(51684), end: SmtChar(146039) }])]
        let n = rs.len();
        let mut builder = ReBuilder::default();
        let rs = rs.into_iter().map(|r| builder.range(r)).collect();
        let regex = builder.union(rs);

        if n > 0 {
            assert!(sample_regex(&regex, &mut builder, 1).is_some());
        } else {
            assert!(sample_regex(&regex, &mut builder, 10).is_none());
        }
    }

    #[quickcheck]
    fn sample_opt(r: CharRange) {
        let mut builder = ReBuilder::default();
        let r = builder.range(r);
        let regex = builder.opt(r);

        assert!(sample_regex(&regex, &mut builder, 0).is_some());
        assert!(sample_regex(&regex, &mut builder, 1).is_some());
    }

    #[test]
    fn sample_empty_string() {
        let mut builder = ReBuilder::default();
        let regex = builder.epsilon();

        assert!(sample_regex(&regex, &mut builder, 0).is_some());
    }

    #[test]
    fn sample_empty_regex() {
        let mut builder = ReBuilder::default();
        let regex = builder.none();

        assert!(sample_regex(&regex, &mut builder, 0).is_none());
        assert!(sample_regex(&regex, &mut builder, 20).is_none());
    }

    #[test]
    fn sample_all() {
        let mut builder = ReBuilder::default();
        let regex = builder.all();

        assert!(sample_regex(&regex, &mut builder, 0).is_some());
        assert!(sample_regex(&regex, &mut builder, 20).is_some());
    }

    #[test]
    fn sample_any() {
        let mut builder = ReBuilder::default();
        let regex = builder.any_char();

        assert!(sample_regex(&regex, &mut builder, 0).is_none());
        assert!(sample_regex(&regex, &mut builder, 20).is_some());
    }

    #[test]
    fn test_sample_nfa_accepts_word() {
        let mut nfa = NFA::new();
        let q0 = nfa.new_state();
        let q1 = nfa.new_state();

        nfa.set_initial(q0).unwrap();
        nfa.add_final(q1).unwrap();

        nfa.add_transition(q0, q1, TransitionType::Range(CharRange::new('a', 'a')))
            .unwrap();

        let sample = sample_nfa(&nfa, 10);
        assert_eq!(sample, Some(SmtString::from("a")));
    }

    #[test]
    fn test_sample_nfa_rejects_unreachable_final_state() {
        let mut nfa = NFA::new();
        let q0 = nfa.new_state();
        let q1 = nfa.new_state(); // Final state, but not reachable

        nfa.set_initial(q0).unwrap();
        nfa.add_final(q1).unwrap();

        let sample = sample_nfa(&nfa, 10);
        assert_eq!(sample, None);
    }

    #[test]
    fn test_sample_nfa_handles_epsilon_transitions() {
        let mut nfa = NFA::new();
        let q0 = nfa.new_state();
        let q1 = nfa.new_state();
        let q2 = nfa.new_state();

        nfa.set_initial(q0).unwrap();
        nfa.add_final(q2).unwrap();

        nfa.add_transition(q0, q1, TransitionType::Epsilon).unwrap();
        nfa.add_transition(q1, q2, TransitionType::Range(CharRange::new('b', 'b')))
            .unwrap();

        let sample = sample_nfa(&nfa, 10);
        assert_eq!(sample, Some(SmtString::from("b")));
    }

    #[test]
    fn test_sample_nfa_stops_at_max_depth() {
        let mut nfa = NFA::new();
        let q0 = nfa.new_state();
        let q1 = nfa.new_state();
        let q2 = nfa.new_state();

        nfa.set_initial(q0).unwrap();
        nfa.add_final(q2).unwrap();

        // Large range transition that makes random sampling harder
        nfa.add_transition(q0, q1, TransitionType::Range(CharRange::new('a', 'z')))
            .unwrap();
        nfa.add_transition(q1, q2, TransitionType::Range(CharRange::new('a', 'z')))
            .unwrap();

        let sample = sample_nfa(&nfa, 1); // Very low max depth
        assert_eq!(sample, None); // Should not reach q2 in one step
    }

    #[test]
    fn test_sample_nfa_handles_not_range_transitions() {
        let mut nfa = NFA::new();
        let q0 = nfa.new_state();
        let q1 = nfa.new_state();

        nfa.set_initial(q0).unwrap();
        nfa.add_final(q1).unwrap();

        nfa.add_transition(q0, q1, TransitionType::NotRange(CharRange::new('x', 'z')))
            .unwrap();

        let sample = sample_nfa(&nfa, 10);
        assert!(sample.is_some()); // Should produce a valid word
        if let Some(word) = sample {
            assert!(
                !word.contains_char('x') && !word.contains_char('y') && !word.contains_char('z')
            );
        }
    }

    #[test]
    fn test_sample_nfa_multiple_paths() {
        let mut nfa = NFA::new();
        let q0 = nfa.new_state();
        let q1 = nfa.new_state();
        let q2 = nfa.new_state();
        let q3 = nfa.new_state();

        nfa.set_initial(q0).unwrap();
        nfa.add_final(q3).unwrap();

        nfa.add_transition(q0, q1, TransitionType::Range(CharRange::new('a', 'a')))
            .unwrap();
        nfa.add_transition(q1, q3, TransitionType::Range(CharRange::new('b', 'b')))
            .unwrap();
        nfa.add_transition(q0, q2, TransitionType::Range(CharRange::new('x', 'x')))
            .unwrap();
        nfa.add_transition(q2, q3, TransitionType::Range(CharRange::new('y', 'y')))
            .unwrap();

        let sample = sample_nfa(&nfa, 10);
        assert!(sample == Some(SmtString::from("ab")) || sample == Some(SmtString::from("xy")));
    }

    #[test]
    fn test_sample_nfa_leaves_loops() {
        let mut nfa = NFA::new();
        let q0 = nfa.new_state();
        let q1 = nfa.new_state();

        nfa.set_initial(q0).unwrap();
        nfa.add_final(q1).unwrap();

        nfa.add_transition(q0, q0, TransitionType::Range(CharRange::singleton('a')))
            .unwrap();
        nfa.add_transition(q0, q1, TransitionType::Range(CharRange::singleton('b')))
            .unwrap();

        match sample_nfa(&nfa, 100) {
            Some(w) => {
                let l = w.len();
                let mut expected = SmtString::from("a").repeat(l - 1);
                expected.push('b');
                assert_eq!(w, expected);
            }
            None => unreachable!("Sample should not return None"),
        }
    }
}
