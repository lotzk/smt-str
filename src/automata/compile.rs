use std::collections::HashSet;
use std::vec;

use indexmap::IndexMap;

use crate::alphabet::{Alphabet, CharRange};

use crate::re::{union_of_chars, ReBuilder, ReOp, Regex};
use crate::SmtString;

use super::comp::complement;
use super::inter::intersect;
use super::{TransitionType, NFA};

use smallvec::smallvec;

/// Compiles the given regex into an NFA.
/// The NFA accepts exactly the language of the regex.
/// The NFA is constructed using the Thompson construction.
/// The returned automaton is trim but can contain epsilon transitions.
pub fn compile(re: &Regex, builder: &mut ReBuilder) -> NFA {
    let thompson = Thompson::default();
    thompson.compile(re, builder)
}

#[derive(Default)]
struct Thompson {
    cache: IndexMap<Regex, NFA>,
}

impl Thompson {
    /// Creates an NFA accepting no words.
    /// This is exactly the empty automaton.
    fn none(&mut self) -> NFA {
        NFA::new()
    }

    /// Creates an NFA accepting exactly the given word
    /// The automaton has a single initial state with transitions for each character in the word.
    /// The final state is the last state of the word.
    /// As such, the automaton has exactly `word.len() + 1` states.
    fn word(&mut self, word: &SmtString) -> NFA {
        let mut nfa = NFA::new();
        let q0 = nfa.new_state();
        nfa.set_initial(q0).unwrap();
        let mut q = q0;
        for c in word.iter() {
            let q1 = nfa.new_state();
            nfa.add_transition(q, q1, TransitionType::Range(CharRange::singleton(*c)))
                .unwrap();
            q = q1;
        }
        nfa.add_final(q).unwrap();
        nfa
    }

    /// Creates an NFA accepting any word of length 1 that is in the given range
    /// The automaton has two states: an initial state and a final state, connected by a transition that accepts any character in the range.
    /// If the range is empty, the automaton is empty.
    fn range(&mut self, range: CharRange) -> NFA {
        if range.is_empty() {
            return self.none();
        }
        let mut nfa = NFA::new();
        let q0 = nfa.new_state();
        nfa.set_initial(q0).unwrap();
        let q1 = nfa.new_state();
        nfa.add_transition(q0, q1, TransitionType::Range(range))
            .unwrap();
        nfa.add_final(q1).unwrap();
        nfa
    }

    /// Creates an NFA accepting the union of the given character ranges.
    /// The ranges are compressed into a list of ranges that are disjoint, non-adjacent, and non-empty.
    /// This is a special case of the `union` method that accepts only character ranges.
    /// The automaton has a single initial state and a single final state, connected by transitions for each range.
    /// Therefore, the automaton is slimmer than the general union method.
    fn ranges(&mut self, rs: Vec<CharRange>) -> NFA {
        // Compress the ranges
        let mut alphabet = Alphabet::default();
        rs.into_iter().for_each(|r| alphabet.insert(r));
        let mut nfa = NFA::new();
        let q0 = nfa.new_state();
        nfa.set_initial(q0).unwrap();
        let qf = nfa.new_state();
        nfa.add_final(qf).unwrap();
        for r in alphabet.iter_ranges() {
            nfa.add_transition(q0, qf, TransitionType::Range(r))
                .unwrap();
        }
        nfa
    }

    /// Creates an NFA accepting all words of length 1 that are not in the given range and all words with length != 1
    fn not_range(&mut self, range: CharRange) -> NFA {
        let mut nfa = NFA::new();
        let q0 = nfa.new_state();
        nfa.set_initial(q0).unwrap();
        let q1 = nfa.new_state();
        let q2 = nfa.new_state();
        nfa.add_transition(q0, q2, TransitionType::NotRange(range))
            .unwrap();
        nfa.add_transition(q0, q1, TransitionType::Range(range))
            .unwrap();
        nfa.add_transition(q1, q2, TransitionType::Range(CharRange::all()))
            .unwrap(); // Word of length 2
        nfa.add_transition(q2, q2, TransitionType::Range(CharRange::all()))
            .unwrap(); // Word of length > 2
        nfa.add_final(q2).unwrap();
        nfa.add_final(q0).unwrap();
        nfa
    }

    /// Creates an NFA accepting any symbol, i.e., any word of length 1
    fn any(&mut self) -> NFA {
        let mut nfa = NFA::new();
        let q0 = nfa.new_state();
        nfa.set_initial(q0).unwrap();
        let q1 = nfa.new_state();
        nfa.add_transition(q0, q1, TransitionType::Range(CharRange::all()))
            .unwrap();
        nfa.add_final(q1).unwrap();
        nfa
    }

    /// Creates the universal NFA, i.e., the NFA accepting all words.
    /// The automaton has a single state that is both initial and final, with a self-loop that accepts any character.
    fn all(&mut self) -> NFA {
        let mut nfa = NFA::new();
        let q0 = nfa.new_state();
        nfa.set_initial(q0).unwrap();
        nfa.add_final(q0).unwrap();
        nfa.add_transition(q0, q0, TransitionType::Range(CharRange::all()))
            .unwrap();
        nfa
    }

    /// Creates an NFA accepting the union of the given NFAs
    fn union(&mut self, rs: &[Regex], builder: &mut ReBuilder) -> NFA {
        match rs.len() {
            0 => return NFA::new(),                        // ∅
            1 => return self.compile_rec(&rs[0], builder), // no need to union
            _ => {}
        }

        let mut sub_nfas = Vec::with_capacity(rs.len());

        for r in rs {
            if r.none().unwrap_or(false) {
                continue;
            }
            let nfa_r = self.compile_rec(r, builder);
            if nfa_r.num_states() > 0 {
                sub_nfas.push(nfa_r);
            }
        }

        let mut nfa = NFA::new();
        let q0 = nfa.new_state();

        for nfa_r in sub_nfas {
            let offset = nfa.merge(&nfa_r);
            if let Some(init) = nfa_r.initial() {
                nfa.add_transition(q0, init + offset, TransitionType::Epsilon)
                    .unwrap();
            }
            for qf in nfa_r.finals() {
                nfa.add_final(qf + offset).unwrap();
            }
        }

        nfa.set_initial(q0).unwrap();
        nfa
    }

    /// Creates an NFA accepting the concatenation of the given NFAs
    fn concat(&mut self, rs: &[Regex], builder: &mut ReBuilder) -> NFA {
        let mut nfa = NFA::new();
        let mut last_finals = HashSet::new();
        let mut first = true;
        for r in rs {
            let nfa_r = self.compile_rec(r, builder);

            if first {
                nfa = nfa_r;
                first = false;
                last_finals = HashSet::from_iter(nfa.finals());
            } else {
                let offset = nfa.merge(&nfa_r);

                for qf in &last_finals {
                    if let Some(initial) = nfa_r.initial() {
                        nfa.add_transition(*qf, initial + offset, TransitionType::Epsilon)
                            .unwrap();
                    }
                }
                nfa.finals.clear();
                last_finals.clear();
                for qf in nfa_r.finals() {
                    nfa.add_final(qf + offset).unwrap();
                    last_finals.insert(qf + offset);
                }
            }
        }
        nfa
    }

    fn inter(&mut self, rs: &[Regex], builder: &mut ReBuilder) -> NFA {
        if rs.is_empty() {
            return self.none();
        }
        // we need to collect all sub-NFAs and intersect them

        let mut nfa = self.compile_rec(&rs[0], builder);

        for r in rs.iter().skip(1) {
            let nfa_r = self.compile_rec(r, builder);
            nfa = intersect(&nfa, &nfa_r)
        }

        nfa
    }

    /// Creates an NFA accepting the Kleene star of the given NFA
    fn star(&mut self, r: &Regex, builder: &mut ReBuilder) -> NFA {
        let mut nfa = self.compile_rec(r, builder);
        if let Some(q0) = nfa.initial() {
            let new_qf = nfa.new_state();
            let new_q0 = nfa.new_state();

            nfa.add_transition(new_q0, q0, TransitionType::Epsilon)
                .unwrap();
            nfa.add_transition(new_q0, new_qf, TransitionType::Epsilon)
                .unwrap();

            // Create a transition from the final states to the initial state and to the new final state
            // Need to store the new transitions in a separate vector to avoid modifying the NFA while iterating over it
            let mut new_transitions = vec![];
            for final_ in nfa.finals() {
                let initial = q0;
                new_transitions.push((final_, initial, TransitionType::Epsilon));
                new_transitions.push((final_, new_qf, TransitionType::Epsilon));
            }
            for (q1, q2, t) in new_transitions {
                nfa.add_transition(q1, q2, t).unwrap();
            }
            nfa.finals.clear();
            nfa.add_final(new_qf).unwrap();
            nfa.set_initial(new_q0).unwrap();
        }
        nfa
    }

    /// Creates and automaton recognizing the Kleene closure of the given character range
    /// This needs just as single state with a self-loop that accepts any character in the range.
    fn star_range(&mut self, range: CharRange) -> NFA {
        let mut nfa = NFA::new();
        let q0 = nfa.new_state();
        nfa.set_initial(q0).unwrap();
        nfa.add_final(q0).unwrap();
        nfa.add_transition(q0, q0, TransitionType::Range(range))
            .unwrap();
        nfa
    }

    /// Creates an NFA accepting the Kleene closure of the given word
    /// This needs |word| + 1 states, which is less than the general Kleene star.
    fn star_word(&mut self, word: &SmtString) -> NFA {
        let mut word_nfa = self.word(word);
        let q0 = word_nfa.initial().unwrap();
        let qf = word_nfa.finals().next().unwrap();

        word_nfa
            .add_transition(qf, q0, TransitionType::Epsilon)
            .unwrap();
        word_nfa
            .add_transition(q0, qf, TransitionType::Epsilon)
            .unwrap();

        word_nfa
    }

    /// Creates an NFA accepting the Kleene plus, i.e., the positive closure, of the given NFA
    fn plus(&mut self, r: &Regex, builder: &mut ReBuilder) -> NFA {
        // Same as Kleene star but we omit the transition that goes from the new initial to the new final, i.e., the transition that "skips" the inner regex.
        let mut nfa = self.compile_rec(r, builder);

        if let Some(q0) = nfa.initial() {
            let new_qf = nfa.new_state();
            let new_q0 = nfa.new_state();

            nfa.add_transition(new_q0, q0, TransitionType::Epsilon)
                .unwrap();

            // Create a transition from the final states to the initial state and to the new final state
            // Need to store the new transitions in a separate vector to avoid modifying the NFA while iterating over it
            let mut new_transitions = vec![];
            for final_ in nfa.finals() {
                let initial = q0;
                new_transitions.push((final_, initial, TransitionType::Epsilon));
                new_transitions.push((final_, new_qf, TransitionType::Epsilon));
            }
            for (q1, q2, t) in new_transitions {
                nfa.add_transition(q1, q2, t).unwrap();
            }
            nfa.finals.clear();
            nfa.add_final(new_qf).unwrap();
            nfa.set_initial(new_q0).unwrap();
        }
        nfa
    }

    /// Creates an NFA accepting the regex and the empty word
    /// This is achieved by simply adding the initial state as a final state.
    fn opt(&mut self, r: &Regex, builder: &mut ReBuilder) -> NFA {
        let mut nfa = self.compile_rec(r, builder);
        if let Some(q0) = nfa.initial() {
            nfa.add_final(q0).unwrap();
        }
        nfa
    }

    /// Creates an NFA accepting the `n`th power of the given regex
    fn pow(&mut self, r: &Regex, n: u32, builder: &mut ReBuilder) -> NFA {
        let concat = vec![r.clone(); n as usize];
        self.concat(&concat, builder)
    }

    /// Creates an NFA accepting the complement of the given regex
    fn comp(&mut self, r: &Regex, builder: &mut ReBuilder) -> NFA {
        let nfa = self.compile_rec(r, builder);
        // Complement the NFA we just compiled
        complement(&nfa)
    }

    /// Creates an NFA accepting the difference of the two given regexes
    fn diff(&mut self, r1: &Regex, r2: &Regex) -> NFA {
        let nfa1 = self.compile_rec(r1, &mut ReBuilder::default());

        let nfa2 = self.compile_rec(r2, &mut ReBuilder::default());

        // The difference is the intersection of the first automaton with the complement of the second automaton
        let comp = complement(&nfa2);
        intersect(&nfa1, &comp)
    }

    /// Creates an NFA accepting the loop of the given regex
    /// That is, accepts at least `lower` and at most `upper` repetitions of the regex.
    fn loop_(&mut self, regex: &Regex, lower: u32, upper: u32, builder: &mut ReBuilder) -> NFA {
        let pow_unrolling = self.bounded_loop(regex, lower, upper, builder);
        self.compile_rec(&pow_unrolling, builder)
    }

    /// Constructs a regex that accepts between `lower` and `upper` repetitions of `regex`,
    /// using power-of-two unrolling to minimize regex size.
    ///
    /// This avoids generating `upper` copies of `regex` by:
    /// - Concatenating `lower` required copies of `regex`
    /// - Adding optional chunks of `1`, `2`, `4`, `8`, ... repetitions, depending on `upper - lower`
    ///
    /// For example, with `lower = 0` and `upper = 253`, it builds:
    ///     (R^1)? (R^2)? (R^4)? (R^8)? (R^16)? (R^32)? (R^64)? (R^128)?
    fn bounded_loop(
        &mut self,
        regex: &Regex,
        lower: u32,
        upper: u32,
        builder: &mut ReBuilder,
    ) -> Regex {
        assert!(lower <= upper);

        // Required part: concat lower copies
        let mut acc = if lower > 0 {
            smallvec![regex.clone(); lower as usize]
        } else {
            smallvec![]
        };

        // Optional part: power-of-two unrolling
        let mut optional_reps = upper - lower;
        let mut chunk = 1;

        while optional_reps > 0 {
            let reps = std::cmp::min(chunk, optional_reps);
            let repeated = builder.concat(smallvec![regex.clone(); reps as usize]);
            let optional = builder.opt(repeated);
            acc.push(optional);
            optional_reps -= reps;
            chunk *= 2;
        }

        builder.concat(acc)
    }

    fn compile_rec(&mut self, regex: &Regex, builder: &mut ReBuilder) -> NFA {
        if let Some(nfa) = self.cache.get(regex) {
            return nfa.clone();
        }
        if let Some(uoc) = union_of_chars(regex) {
            return self.ranges(uoc);
        }
        let nfa = match regex.op() {
            ReOp::Literal(w) => self.word(w),
            ReOp::None => self.none(),
            ReOp::All => self.all(),
            ReOp::Any => self.any(),
            ReOp::Concat(rs) => self.concat(rs, builder),
            ReOp::Union(rs) => self.union(rs, builder),
            ReOp::Inter(rs) => self.inter(rs, builder),
            ReOp::Star(r) => match r.op() {
                ReOp::Range(r) => self.star_range(*r),
                ReOp::Literal(w) => self.star_word(w),
                _ => self.star(r, builder),
            },
            ReOp::Plus(r) => self.plus(r, builder),
            ReOp::Opt(r) => self.opt(r, builder),
            ReOp::Range(r) => self.range(*r),
            ReOp::Comp(c) => match c.op() {
                ReOp::Range(r) => self.not_range(*r),
                ReOp::Literal(w) if w.len() == 1 => self.not_range(CharRange::new(w[0], w[0])),
                _ => self.comp(c, builder),
            },
            ReOp::Diff(r1, r2) => self.diff(r1, r2),
            ReOp::Pow(r, n) => self.pow(r, *n, builder),
            ReOp::Loop(r, l, u) => self.loop_(r, *l, *u, builder),
        };

        self.cache.insert(regex.clone(), nfa.clone());

        nfa
    }

    /// Compiles the given regex into an NFA.
    pub fn compile(mut self, regex: &Regex, builder: &mut ReBuilder) -> NFA {
        let nfa = self.compile_rec(regex, builder);
        nfa.trim()
    }
}

#[cfg(test)]
mod test {
    use quickcheck::TestResult;
    use quickcheck_macros::quickcheck;
    use rand::prelude::Distribution;

    use super::*;
    use crate::{alphabet::CharRange, re::ReBuilder, SmtChar};
    use smallvec::smallvec;

    fn test_acceptance(
        re: &Regex,
        builder: &mut ReBuilder,
        pos: &[SmtString],
        neg: &[SmtString],
    ) -> NFA {
        let thompson = Thompson::default();
        let nfa = thompson.compile(re, builder);
        for p in pos {
            assert!(nfa.accepts(p), "Expected NFA for {} to accept {}", re, p);
        }
        for n in neg {
            assert!(
                !nfa.accepts(n),
                "Expected NFA for {} to reject {}\n{}",
                re,
                n,
                nfa.dot()
            );
        }
        nfa
    }

    #[quickcheck]
    fn test_compile_word(w: SmtString) {
        let mut builder = ReBuilder::default();
        let re = builder.to_re(w.clone());
        let nfa = test_acceptance(&re, &mut builder, &[w.clone()], &[]);
        assert_eq!(nfa.states().count(), w.len() + 1);
    }

    #[test]
    fn test_compile_any() {
        let mut builder = ReBuilder::default();
        let re = builder.allchar();
        let nfa = test_acceptance(&re, &mut builder, &["a".into()], &["".into(), "ab".into()]);
        assert_eq!(nfa.states().count(), 2);
    }

    #[quickcheck]
    fn test_compile_all(w: SmtString) {
        let mut builder = ReBuilder::default();
        let re = builder.all();
        let nfa = test_acceptance(&re, &mut builder, &[w], &[]);
        assert_eq!(nfa.states().count(), 1);
    }

    #[quickcheck]
    fn test_compile_none(w: SmtString) {
        let mut builder = ReBuilder::default();
        let re = builder.none();
        let nfa = test_acceptance(&re, &mut builder, &[], &[w]);
        assert_eq!(nfa.states().count(), 0);
    }

    #[quickcheck]
    fn test_compile_range(r: CharRange) -> TestResult {
        let mut builder = ReBuilder::default();
        let re = builder.range(r);

        // Pick random chars in the range
        let mut rng = rand::rng();
        let dist = rand::distr::Uniform::new_inclusive(r.start().0, r.end().0).unwrap();

        for _ in 0..100 {
            let c = dist.sample(&mut rng);
            let nfa = test_acceptance(&re, &mut builder, &[SmtChar::from(c).into()], &[]);
            assert_eq!(nfa.states().count(), 2);
        }

        TestResult::passed()
    }

    #[quickcheck]
    fn test_compile_ranges(rs: Vec<CharRange>) {
        let mut builder = ReBuilder::default();
        let re_rs = rs.iter().map(|r| builder.range(*r)).collect();
        let re = builder.union(re_rs);

        let mut alph = Alphabet::empty();
        for r in rs.iter() {
            alph.insert(*r);
        }

        let th = Thompson::default();
        let nfa = th.compile(&re, &mut builder);

        if alph.is_empty() {
            assert_eq!(nfa.num_states(), 0);
        } else {
            debug_assert_eq!(nfa.num_states(), 2, "{}", nfa);
        }

        for r in alph.iter_ranges() {
            let mut rng = rand::rng();
            let dist = rand::distr::Uniform::new_inclusive(r.start().0, r.end().0).unwrap();
            for _ in 0..100 {
                let c = dist.sample(&mut rng);
                assert!(nfa.accepts(&SmtChar::from(c).into()));
                assert_eq!(nfa.states().count(), 2);
            }
        }
    }

    #[test]
    fn test_compile_not_char() {
        let mut builder = ReBuilder::default();
        let re = builder.range_from_to('a', 'a');
        let re = builder.comp(re);
        test_acceptance(
            &re,
            &mut builder,
            &["b".into(), "aa".into(), "abc".into(), "bb".into()],
            &["a".into()],
        );
    }

    #[quickcheck]
    fn test_compile_not_range(r: CharRange) -> TestResult {
        assert!(r.start() <= r.end());

        let mut builder = ReBuilder::default();

        let re = builder.range(r);
        let re = builder.comp(re);

        let comp = r.complement();
        let th = Thompson::default();
        let nfa = th.compile(&re, &mut builder);

        for crange in comp {
            debug_assert!(crange.intersect(&r).is_empty());
            let mut rng = rand::rng();
            let dist_in =
                match rand::distr::Uniform::new_inclusive(crange.start().0, crange.end().0) {
                    Ok(d) => d,
                    Err(e) => panic!("Error: {}", e),
                };

            for _ in 0..100 {
                let c = dist_in.sample(&mut rng);
                assert!(nfa.accepts(&SmtChar::from(c).into()));
            }
        }

        TestResult::passed()
    }

    #[quickcheck]
    fn test_compile_comp_char(r: SmtChar) -> TestResult {
        let mut builder = ReBuilder::default();

        let re = builder.to_re(r.into());
        let re = builder.comp(re);

        let comp = CharRange::singleton(r).complement();
        let th = Thompson::default();
        let nfa = th.compile(&re, &mut builder);

        for crange in comp {
            let mut rng = rand::rng();
            let dist_in =
                match rand::distr::Uniform::new_inclusive(crange.start().0, crange.end().0) {
                    Ok(d) => d,
                    Err(e) => panic!("Error: {}", e),
                };

            for _ in 0..100 {
                let c = dist_in.sample(&mut rng);
                assert!(nfa.accepts(&SmtChar::from(c).into()));
            }
        }

        TestResult::passed()
    }

    #[test]
    fn test_compile_concat_comp_char() {
        let mut builder = ReBuilder::default();
        let a = builder.to_re("a".into());
        let b = builder.to_re("b".into());
        let not_a = builder.comp(a);
        let not_a_b = builder.concat(smallvec![not_a.clone(), b.clone()]);

        let th = Thompson::default();
        let nfa = th.compile(&not_a_b, &mut builder);

        assert!(nfa.accepts(&"b".into()));
        assert!(nfa.accepts(&"abb".into()));

        assert!(!nfa.accepts(&"a".into()));
        assert!(!nfa.accepts(&"".into()));
        assert!(!nfa.accepts(&"ab".into()));
    }

    #[test]
    fn test_compile_union_chars() {
        let mut builder = ReBuilder::default();
        let rw1 = builder.to_re("f".into());
        let rw2 = builder.to_re("b".into());
        let re = builder.union(smallvec![rw1.clone(), rw2.clone()]);
        let th = Thompson::default();
        let nfa = th.compile(&re, &mut builder);

        assert!(nfa.accepts(&"f".into()));
        assert!(nfa.accepts(&"b".into()));
        assert!(!nfa.accepts(&"fb".into()));
    }

    #[quickcheck]
    fn test_compile_union(w1: SmtString, w2: SmtString, wneg: SmtString) {
        let mut builder = ReBuilder::default();
        let rw1 = builder.to_re(w1.clone());
        let rw2 = builder.to_re(w2.clone());
        let re = builder.union(smallvec![rw1.clone(), rw2.clone()]);
        if w1 == wneg || w2 == wneg {
            test_acceptance(&re, &mut builder, &[w1, w2], &[]);
        } else {
            test_acceptance(&re, &mut builder, &[w1, w2], &[wneg]);
        }
    }

    #[quickcheck]
    fn test_compile_concat_char() {
        let mut builder = ReBuilder::default();
        let rw1 = builder.to_re("a".into());
        let rw2 = builder.to_re("a".into());
        let re = builder.concat(smallvec![rw1.clone(), rw2.clone()]);
        let th = Thompson::default();
        let nfa = th.compile(&re, &mut builder);
        assert!(nfa.accepts(&"aa".into()));
        assert!(!nfa.accepts(&"a".into()));
        assert!(!nfa.accepts(&"aaa".into()));
    }

    #[quickcheck]
    fn test_compile_concat(w1: SmtString, w2: SmtString, wneg: SmtString) {
        let mut builder = ReBuilder::default();
        let rw1 = builder.to_re(w1.clone());
        let rw2 = builder.to_re(w2.clone());
        let re = builder.concat(smallvec![rw1.clone(), rw2.clone()]);

        let w1w2 = w1.concat(&w2);
        if w1.concat(&w2) == wneg {
            test_acceptance(&re, &mut builder, &[w1w2], &[]);
        } else {
            test_acceptance(&re, &mut builder, &[w1w2], &[wneg]);
        }
    }

    #[quickcheck]
    fn test_compile_star(w: SmtString, n: usize) {
        let mut builder = ReBuilder::default();
        let rw = builder.to_re(w.clone());
        let re = builder.star(rw);
        test_acceptance(&re, &mut builder, &[w.repeat(n % 100)], &[]);
    }

    #[quickcheck]
    fn test_compile_plus(w: SmtString, n: usize) {
        let mut builder = ReBuilder::default();
        let rw = builder.to_re(w.clone());
        let re = builder.plus(rw);
        if w.is_empty() {
            test_acceptance(&re, &mut builder, &[w.repeat(n % 100)], &[]);
        } else {
            test_acceptance(&re, &mut builder, &[w.repeat((n % 100) + 1)], &[]);
        }
    }

    #[quickcheck]
    fn test_compile_plus_of_any(w: SmtString) {
        let mut builder = ReBuilder::default();
        let any = builder.allchar();
        let re = builder.plus(any);
        if w.is_empty() {
            test_acceptance(&re, &mut builder, &[], &[w]);
        } else {
            test_acceptance(&re, &mut builder, &[w], &[]);
        }
    }

    #[quickcheck]
    fn test_compile_opt(w: SmtString) {
        let mut builder = ReBuilder::default();
        let rw = builder.to_re(w.clone());
        let re = builder.opt(rw);
        test_acceptance(&re, &mut builder, &[w.clone(), "".into()], &[]);
    }

    #[quickcheck]
    fn test_compile_pow_char() {
        let w: SmtString = "a".into();
        let n = 2;
        let mut builder = ReBuilder::default();
        let rw = builder.to_re(w.clone());
        let re = builder.pow(rw, n as u32);
        let th = Thompson::default();
        let nfa = th.compile(&re, &mut builder);
        for i in 0..=n {
            if i < n {
                assert!(!nfa.accepts(&w.repeat(i)),);
            } else {
                assert!(nfa.accepts(&w.repeat(i)),);
            }
        }
        assert!(!nfa.accepts(&w.repeat(n + 1)),);
    }

    #[quickcheck]
    fn test_compile_pow_char_rejects_longer() {
        let w: SmtString = "a".into();
        let n = 2;
        let mut builder = ReBuilder::default();
        let rw = builder.to_re(w.clone());
        let re = builder.pow(rw, n as u32);

        let th = Thompson::default();
        let nfa = th.compile(&re, &mut builder);
        assert!(!nfa.accepts(&w.repeat(n + 1)),);
    }

    #[quickcheck]
    fn test_compile_pow(w: SmtString, n: usize) {
        let n = n % 50;
        let w = w.take(50);
        let mut builder = ReBuilder::default();
        let rw = builder.to_re(w.clone());
        let re = builder.pow(rw, n as u32);
        let th = Thompson::default();

        let nfa = th.compile(&re, &mut builder);

        for i in 0..=n {
            if i < n && !w.is_empty() {
                assert!(!nfa.accepts(&w.repeat(i)),);
            } else {
                assert!(nfa.accepts(&w.repeat(i)),);
            }
        }
    }

    #[quickcheck]
    fn test_compile_loop_char(l: usize, u: usize) {
        let w: SmtString = "a".into();

        let l = l % 50;
        let u = u % 50;

        let mut builder = ReBuilder::default();
        let rw = builder.to_re(w.clone());
        let re = builder.loop_(rw, l as u32, u as u32);
        let th = Thompson::default();
        let nfa = th.compile(&re, &mut builder);
        for i in 0..=u {
            if i < l {
                assert!(
                    !nfa.accepts(&w.repeat(i)),
                    "loop({}, {}, {}) expected to reject {}^{}",
                    w,
                    l,
                    u,
                    w,
                    i
                );
            } else {
                assert!(
                    nfa.accepts(&w.repeat(i)),
                    "loop({}, {}, {}) accept to reject {}^{}",
                    w,
                    l,
                    u,
                    w,
                    i
                );
            }
        }
    }

    #[quickcheck]
    fn test_compile_loop_word(w: SmtString, l: usize, u: usize) {
        let l = l % 50;
        let u = u % 50;
        let w = w.take(20);
        let (l, u) = if l > u { (u, l) } else { (l, u) };
        let mut builder = ReBuilder::default();
        let rw = builder.to_re(w.clone());
        let re = builder.loop_(rw, l as u32, u as u32);
        let th = Thompson::default();
        let nfa = th.compile(&re, &mut builder);
        for i in 0..=u {
            if i < l && !w.is_empty() {
                assert!(
                    !nfa.accepts(&w.repeat(i)),
                    "NFA accepts {} for i={}",
                    w.repeat(i),
                    i
                );
            } else {
                assert!(
                    nfa.accepts(&w.repeat(i)),
                    "NFA rejects {} for i={}",
                    w.repeat(i),
                    i
                );
            }
        }
    }
}
