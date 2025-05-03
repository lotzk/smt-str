use bit_set::BitSet;

use rand::{rng, seq::IteratorRandom};

use crate::{
    automata::{TransitionType, NFA},
    re::{deriv::DerivativeBuilder, ReBuilder, ReOp, Regex},
    SmtString,
};

/// The result of sampling from a regex or automaton.
#[derive(Debug, PartialEq, Eq, Clone)]
pub enum SampleResult {
    /// Founda word in the language
    Sampled(SmtString),
    /// The language tried to sample from is empty
    Empty,
    /// Maximum depth was reached without finding a word.
    MaxDepth,
}

impl SampleResult {
    /// Unwraps the sampled string.
    /// Panics if the sampling was not successfull.
    pub fn unwrap(self) -> SmtString {
        match self {
            SampleResult::Sampled(s) => s,
            _ => panic!("called `unwrap` on empty value"),
        }
    }

    /// Return true if sampling was successfull and this result carries a value.
    /// Othwerwise returns false.
    pub fn success(&self) -> bool {
        matches!(self, SampleResult::Sampled(_))
    }
}

/// Tries to sample a word that is accepted by the regex.
/// The function aborts if no word is found after `max_depth` steps.
/// If `comp` is set to `true`, the function will return a word that is not accepted by the regex.
/// In other words, the function will sample a word from the complement of the regex's language.
pub fn sample_regex(
    regex: &Regex,
    builder: &mut ReBuilder,
    max_depth: usize,
    comp: bool,
) -> SampleResult {
    fn fast_sample(re: &Regex, d: usize, max: usize) -> SampleResult {
        if d > max {
            return SampleResult::MaxDepth;
        }
        match re.op() {
            ReOp::Literal(w) => SampleResult::Sampled(w.clone()),
            ReOp::Range(r) => {
                if let Some(r) = r.choose().map(|c| c.into()) {
                    SampleResult::Sampled(r)
                } else {
                    SampleResult::Empty
                }
            }
            ReOp::None => SampleResult::Empty,
            ReOp::Any | ReOp::All => SampleResult::Sampled(SmtString::from("a")),
            ReOp::Concat(rs) => {
                let mut res = SmtString::empty();
                for r in rs {
                    match fast_sample(r, d + 1, max) {
                        SampleResult::Sampled(s) => res.append(&s),
                        SampleResult::Empty => return SampleResult::Empty,
                        SampleResult::MaxDepth => return SampleResult::MaxDepth,
                    }
                }
                SampleResult::Sampled(res)
            }
            ReOp::Comp(comped) => match comped.op() {
                ReOp::Literal(s) => {
                    if s.is_empty() {
                        SampleResult::Sampled("a".into())
                    } else {
                        SampleResult::Sampled(SmtString::empty())
                    }
                }
                ReOp::Range(range) => {
                    for c in range.complement() {
                        if let Some(c) = c.choose() {
                            return SampleResult::Sampled(c.into());
                        }
                    }
                    SampleResult::Empty
                }
                ReOp::None => SampleResult::Sampled("a".into()),
                ReOp::Any => SampleResult::Sampled("aa".into()),
                ReOp::All => SampleResult::Empty,
                ReOp::Comp(r) => fast_sample(r, d + 1, max), // Double complement
                _ => SampleResult::MaxDepth,
            },
            ReOp::Union(rs) => {
                let mut max_reached = false;
                for r in rs {
                    match fast_sample(r, d + 1, max) {
                        SampleResult::Sampled(s) => return SampleResult::Sampled(s),
                        SampleResult::Empty => (),
                        SampleResult::MaxDepth => max_reached = true,
                    }
                }
                if max_reached {
                    SampleResult::MaxDepth
                } else {
                    SampleResult::Empty
                }
            }
            ReOp::Star(_) | ReOp::Opt(_) => SampleResult::Sampled(SmtString::empty()),
            ReOp::Plus(r) => fast_sample(r, d + 1, max),
            ReOp::Pow(r, e) => match fast_sample(r, d + 1, max) {
                SampleResult::Sampled(s) => SampleResult::Sampled(s.repeat(*e as usize)),
                SampleResult::Empty => SampleResult::Empty,
                SampleResult::MaxDepth => SampleResult::MaxDepth,
            },
            ReOp::Loop(r, l, u) if l <= u => match fast_sample(r, d + 1, max) {
                SampleResult::Sampled(s) => SampleResult::Sampled(s.repeat(*l as usize)),
                SampleResult::Empty => SampleResult::Empty,
                SampleResult::MaxDepth => SampleResult::MaxDepth,
            },
            ReOp::Loop(_, _, _) => SampleResult::Empty,
            _ => SampleResult::MaxDepth,
        }
    }

    if !comp {
        match fast_sample(regex, 0, max_depth) {
            SampleResult::Sampled(s) => return SampleResult::Sampled(s),
            SampleResult::Empty => return SampleResult::Empty,
            SampleResult::MaxDepth => (),
        }
    }

    let mut w = SmtString::empty();
    let mut deriver = DerivativeBuilder::default();

    let mut i = 0;
    let mut re = regex.clone();

    let done = |re: &Regex| {
        if comp {
            !re.nullable()
        } else {
            re.nullable()
        }
    };

    if done(&re) {
        return SampleResult::Sampled(w);
    }

    while !done(&re) && i < max_depth {
        let next = if let Some(c) = re
            .first()
            .iter()
            .choose(&mut rng())
            .and_then(|c| c.choose())
        {
            c
        } else {
            return SampleResult::Empty;
        };
        w.push(next);
        re = deriver.deriv(&re, next, builder);
        i += 1;
    }

    if done(&re) {
        SampleResult::Sampled(w)
    } else {
        SampleResult::MaxDepth
    }
}

/// Tries to sample a word that is accepted or not accepted by the NFA.
/// Randomly picks transitions to follow until a final state is reached.
/// Once a final state is reached, the function returns the word that was sampled.
/// The function aborts if no word is found after `max_depth` transitions.
/// If `comp` is set to `true`, the function will return a word that is not accepted by the NFA.
/// In other words, the function will sample a word from the complement of the NFA's language.
///
/// The NFA should be trim. Othwerwise the function returns `SampleResult::Empty` even though
/// it is not. That happens if it runs into a state from which is cannot make progress anymore.
/// Such states do not occur in trim automata.
pub fn sample_nfa(nfa: &NFA, max: usize, comp: bool) -> SampleResult {
    let mut w = SmtString::empty();

    println!("Sampling from: {}", nfa.dot());

    let mut states = BitSet::new();
    if let Some(q0) = nfa.initial() {
        states = BitSet::from_iter(nfa.epsilon_closure(q0).unwrap());
    }
    let mut i = 0;

    let done = |s: &BitSet| {
        if comp {
            !s.iter().any(|q| nfa.is_final(q))
        } else {
            s.iter().any(|q| nfa.is_final(q))
        }
    };

    while i <= max {
        i += 1;
        // Check if the current state set contains a final state
        if done(&states) {
            return SampleResult::Sampled(w);
        }

        // Collect all transitions from the current state set
        let mut transitions = Vec::new();
        for q in states.iter() {
            transitions.extend(nfa.transitions_from(q).unwrap());
        }
        // Pick a random transition
        let transition = match transitions.iter().choose(&mut rng()) {
            Some(t) => t,
            None => return SampleResult::Empty,
        };
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
                states = BitSet::from_iter(
                    states
                        .iter()
                        .flat_map(|s| nfa.consume(s, c))
                        .flatten()
                        .flat_map(|q| nfa.epsilon_closure(q))
                        .flatten(),
                );
            }
            None => continue,
        }
    }

    SampleResult::MaxDepth
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

        assert_eq!(
            sample_regex(&regex, &mut builder, 3, false).unwrap(),
            "foo".into()
        );
        assert_eq!(
            sample_regex(&regex, &mut builder, 10, false).unwrap(),
            "foo".into()
        );
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
        assert!(sample_regex(&regex, &mut builder, 5, false).success());
    }

    #[quickcheck]
    fn sample_with_character_range(range: CharRange) {
        let mut builder = ReBuilder::default();
        let regex = builder.range(range);

        assert!(sample_regex(&regex, &mut builder, 1, false).success());
        // Test matching word within the class
        assert!(sample_regex(&regex, &mut builder, 3, false).success());
    }

    #[quickcheck]
    fn sample_character_range(range: CharRange) {
        let mut builder = ReBuilder::default();
        let regex = builder.range(range);

        assert!(sample_regex(&regex, &mut builder, 1, false).success());
        // Test matching word within the class
        assert!(sample_regex(&regex, &mut builder, 3, false).success());
    }

    #[quickcheck]
    fn sample_character_range_pow(range: CharRange, n: u32) {
        let n = n % 100;
        let mut builder = ReBuilder::default();
        let regex = builder.range(range);
        let regex = builder.pow(regex, n);

        assert!(sample_regex(&regex, &mut builder, n as usize, false).success());
    }

    #[quickcheck]
    fn sample_alternatives(rs: Vec<CharRange>) {
        let n = rs.len();
        let mut builder = ReBuilder::default();
        let rs = rs.into_iter().map(|r| builder.range(r)).collect();
        let regex = builder.union(rs);

        if n > 0 {
            assert!(sample_regex(&regex, &mut builder, 1, false).success());
        } else {
            assert!(!sample_regex(&regex, &mut builder, 10, false).success());
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
            assert!(sample_regex(&regex, &mut builder, 1, false).success());
        } else {
            assert!(!sample_regex(&regex, &mut builder, 10, false).success());
        }
    }

    #[quickcheck]
    fn sample_opt(r: CharRange) {
        let mut builder = ReBuilder::default();
        let r = builder.range(r);
        let regex = builder.opt(r);

        assert!(sample_regex(&regex, &mut builder, 0, false).success());
        assert!(sample_regex(&regex, &mut builder, 1, false).success());
    }

    #[test]
    fn sample_empty_string() {
        let mut builder = ReBuilder::default();
        let regex = builder.epsilon();

        assert!(sample_regex(&regex, &mut builder, 0, false).success());
    }

    #[test]
    fn sample_empty_regex() {
        let mut builder = ReBuilder::default();
        let regex = builder.none();

        assert_eq!(
            sample_regex(&regex, &mut builder, 0, false),
            SampleResult::Empty
        );
        assert_eq!(
            sample_regex(&regex, &mut builder, 20, false),
            SampleResult::Empty
        );
    }

    #[test]
    fn sample_all() {
        let mut builder = ReBuilder::default();
        let regex = builder.all();

        assert!(sample_regex(&regex, &mut builder, 0, false).success());
        assert!(sample_regex(&regex, &mut builder, 20, false).success());
    }

    #[test]
    fn sample_any() {
        let mut builder = ReBuilder::default();
        let regex = builder.allchar();
        assert!(sample_regex(&regex, &mut builder, 20, false).success());
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

        let sample = sample_nfa(&nfa, 10, false);
        assert_eq!(sample, SampleResult::Sampled(SmtString::from("a")));
    }

    #[test]
    fn test_sample_nfa_rejects_unreachable_final_state() {
        let mut nfa = NFA::new();
        let q0 = nfa.new_state();
        let q1 = nfa.new_state(); // Final state, but not reachable

        nfa.set_initial(q0).unwrap();
        nfa.add_final(q1).unwrap();

        let sample = sample_nfa(&nfa, 10, false);
        assert_eq!(sample, SampleResult::Empty);
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

        let sample = sample_nfa(&nfa, 10, false);
        assert_eq!(sample, SampleResult::Sampled(SmtString::from("b")));
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

        let sample = sample_nfa(&nfa, 1, false); // Very low max depth
        assert_eq!(sample, SampleResult::MaxDepth); // Should not reach q2 in one step
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

        let sample = sample_nfa(&nfa, 10, false);
        assert!(sample.success()); // Should produce a valid word
        if let SampleResult::Sampled(word) = sample {
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

        let sample = sample_nfa(&nfa, 10, false);
        assert!(
            sample == SampleResult::Sampled(SmtString::from("ab"))
                || sample == SampleResult::Sampled(SmtString::from("xy"))
        );
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

        match sample_nfa(&nfa, 100, false) {
            SampleResult::Sampled(w) => {
                let l = w.len();
                let mut expected = SmtString::from("a").repeat(l - 1);
                expected.push('b');
                assert_eq!(w, expected);
            }
            _ => unreachable!("Sample should not return None"),
        }
    }
}
