use std::collections::HashSet;
use std::vec;

use crate::alphabet::{Alphabet, CharRange};

use crate::re::{union_of_chars, ReBuilder, ReOp, Regex};
use crate::SmtString;

use super::comp::complement;
use super::inter::intersect;
use super::{TransitionType, NFA};

#[derive(Default)]
pub struct Thompson {
    /// The NFA under construction
    /// The NFA accepts exactly the language of the regex that has been compiled last, but keeps the states of all previous compilations.
    /// This prevents that NFAs for multiple regexes need to be merged into a single automaton.
    nfa: NFA,
}

impl Thompson {
    /// Creates an NFA accepting no words.
    /// This is exactly the empty automaton.
    fn none(&mut self) {
        self.nfa.initial = None;
        self.nfa.finals.clear();
    }

    /// Creates an NFA accepting exactly the given word
    /// The automaton has a single initial state with transitions for each character in the word.
    /// The final state is the last state of the word.
    /// As such, the automaton has exactly `word.len() + 1` states.
    fn word(&mut self, word: &SmtString) {
        let q0 = self.nfa.new_state();
        self.nfa.set_initial(q0).unwrap();
        let mut q = q0;
        for c in word.iter() {
            let q1 = self.nfa.new_state();
            self.nfa
                .add_transition(q, q1, TransitionType::Range(CharRange::singleton(*c)))
                .unwrap();
            q = q1;
        }
        self.nfa.add_final(q).unwrap();
    }

    /// Creates an NFA accepting any word of length 1 that is in the given range
    /// The automaton has two states: an initial state and a final state, connected by a transition that accepts any character in the range.
    /// If the range is empty, the automaton is empty.
    fn range(&mut self, range: CharRange) {
        if range.is_empty() {
            return self.none();
        }
        let q0 = self.nfa.new_state();
        self.nfa.set_initial(q0).unwrap();
        let q1 = self.nfa.new_state();
        self.nfa
            .add_transition(q0, q1, TransitionType::Range(range))
            .unwrap();
        self.nfa.add_final(q1).unwrap();
    }

    /// Creates an NFA accepting the union of the given character ranges.
    /// The ranges are compressed into a list of ranges that are disjoint, non-adjacent, and non-empty.
    /// This is a special case of the `union` method that accepts only character ranges.
    /// The automaton has a single initial state and a single final state, connected by transitions for each range.
    /// Therefore, the automaton is slimmer than the general union method.
    fn ranges(&mut self, rs: Vec<CharRange>) {
        // Compress the ranges
        let mut alphabet = Alphabet::default();
        rs.into_iter().for_each(|r| alphabet.insert(r));
        let q0 = self.nfa.new_state();
        self.nfa.set_initial(q0).unwrap();
        let qf = self.nfa.new_state();
        self.nfa.add_final(qf).unwrap();
        for r in alphabet.iter_ranges() {
            self.nfa
                .add_transition(q0, qf, TransitionType::Range(r))
                .unwrap();
        }
    }

    /// Creates an NFA accepting all words of length 1 that are not in the given range and all words with length != 1
    fn not_range(&mut self, range: CharRange) {
        let q0 = self.nfa.new_state();
        self.nfa.set_initial(q0).unwrap();
        let q1 = self.nfa.new_state();
        let q2 = self.nfa.new_state();
        self.nfa
            .add_transition(q0, q2, TransitionType::NotRange(range))
            .unwrap();
        self.nfa
            .add_transition(q0, q1, TransitionType::Range(range))
            .unwrap();
        self.nfa
            .add_transition(q1, q2, TransitionType::Range(CharRange::all()))
            .unwrap(); // Word of length 2
        self.nfa
            .add_transition(q2, q2, TransitionType::Range(CharRange::all()))
            .unwrap(); // Word of length > 2
        self.nfa.add_final(q2).unwrap();
        self.nfa.add_final(q0).unwrap();
    }

    /// Creates an NFA accepting any symbol, i.e., any word of length 1
    fn any(&mut self) {
        let q0 = self.nfa.new_state();
        self.nfa.set_initial(q0).unwrap();
        let q1 = self.nfa.new_state();
        self.nfa
            .add_transition(q0, q1, TransitionType::Range(CharRange::all()))
            .unwrap();
        self.nfa.add_final(q1).unwrap();
    }

    /// Creates the universal NFA, i.e., the NFA accepting all words.
    /// The automaton has a single state that is both initial and final, with a self-loop that accepts any character.
    fn all(&mut self) {
        let q0 = self.nfa.new_state();
        self.nfa.set_initial(q0).unwrap();
        self.nfa.add_final(q0).unwrap();
        self.nfa
            .add_transition(q0, q0, TransitionType::Range(CharRange::all()))
            .unwrap();
    }

    /// Creates an NFA accepting the union of the given NFAs
    fn union(&mut self, rs: &[Regex], builder: &mut ReBuilder) {
        let q0 = self.nfa.new_state();
        let mut finals = HashSet::new();
        for r in rs {
            self.compile_rec(r, builder);
            if let Some(initial) = self.nfa.initial() {
                self.nfa
                    .add_transition(q0, initial, TransitionType::Epsilon)
                    .unwrap();
            }
            for qf in self.nfa.finals() {
                finals.insert(qf);
            }
        }
        self.nfa.set_initial(q0).unwrap();
        self.nfa.finals = finals;
    }

    /// Creates an NFA accepting the concatenation of the given NFAs
    fn concat(&mut self, rs: &[Regex], builder: &mut ReBuilder) {
        let q0 = self.nfa.new_state();
        let mut finals = HashSet::new();
        finals.insert(q0);
        for r in rs {
            self.compile_rec(r, builder);
            if let Some(initial) = self.nfa.initial() {
                for qf in finals {
                    self.nfa
                        .add_transition(qf, initial, TransitionType::Epsilon)
                        .unwrap();
                }
            }
            finals = self.nfa.finals.clone();
            self.nfa.finals.clear();
        }
        self.nfa.set_initial(q0).unwrap();
        self.nfa.finals = finals;
    }

    fn inter(&mut self, rs: &[Regex], builder: &mut ReBuilder) {
        if rs.is_empty() {
            return self.none();
        }
        // we need to collect all sub-NFAs and intersect them
        self.compile_rec(&rs[0], builder);
        let mut nfa = self.nfa.clone();

        for r in rs.iter().skip(1) {
            // we can clear the nfa as we have already intersected it with the first regex and don't need it anymore
            // this saves memory
            self.nfa = NFA::new();
            self.compile_rec(r, builder);
            nfa = intersect(&nfa, &self.nfa)
        }

        self.nfa = nfa;
    }

    /// Creates an NFA accepting the Kleene star of the given NFA
    fn star(&mut self, r: &Regex, builder: &mut ReBuilder) {
        self.compile_rec(r, builder);
        if let Some(q0) = self.nfa.initial() {
            let new_qf = self.nfa.new_state();
            let new_q0 = self.nfa.new_state();

            self.nfa
                .add_transition(new_q0, q0, TransitionType::Epsilon)
                .unwrap();
            self.nfa
                .add_transition(new_q0, new_qf, TransitionType::Epsilon)
                .unwrap();

            // Create a transition from the final states to the initial state and to the new final state
            // Need to store the new transitions in a separate vector to avoid modifying the NFA while iterating over it
            let mut new_transitions = vec![];
            for final_ in self.nfa.finals() {
                let initial = q0;
                new_transitions.push((final_, initial, TransitionType::Epsilon));
                new_transitions.push((final_, new_qf, TransitionType::Epsilon));
            }
            for (q1, q2, t) in new_transitions {
                self.nfa.add_transition(q1, q2, t).unwrap();
            }
            self.nfa.finals.clear();
            self.nfa.add_final(new_qf).unwrap();
            self.nfa.set_initial(new_q0).unwrap();
        }
    }

    /// Creates an NFA accepting the Kleene plus, i.e., the positive closure, of the given NFA
    fn plus(&mut self, r: &Regex, builder: &mut ReBuilder) {
        // Same as Kleene star but we omit the transition that goes from the new initial to the new final, i.e., the transition that "skips" the inner regex.
        self.compile_rec(r, builder);
        if let Some(q0) = self.nfa.initial() {
            let new_qf = self.nfa.new_state();
            let new_q0 = self.nfa.new_state();

            self.nfa
                .add_transition(new_q0, q0, TransitionType::Epsilon)
                .unwrap();

            // Create a transition from the final states to the initial state and to the new final state
            // Need to store the new transitions in a separate vector to avoid modifying the NFA while iterating over it
            let mut new_transitions = vec![];
            for final_ in self.nfa.finals() {
                let initial = q0;
                new_transitions.push((final_, initial, TransitionType::Epsilon));
                new_transitions.push((final_, new_qf, TransitionType::Epsilon));
            }
            for (q1, q2, t) in new_transitions {
                self.nfa.add_transition(q1, q2, t).unwrap();
            }
            self.nfa.finals.clear();
            self.nfa.add_final(new_qf).unwrap();
            self.nfa.set_initial(new_q0).unwrap();
        }
    }

    /// Creates an NFA accepting the regex and the empty word
    /// This is achieved by simply adding the initial state as a final state.
    fn opt(&mut self, r: &Regex, builder: &mut ReBuilder) {
        self.compile_rec(r, builder);
        if let Some(q0) = self.nfa.initial() {
            self.nfa.add_final(q0).unwrap();
        }
    }

    /// Creates an NFA accepting the `n`th power of the given regex
    fn pow(&mut self, r: &Regex, n: u32, builder: &mut ReBuilder) {
        let concat = vec![r.clone(); n as usize];
        self.concat(&concat, builder)
    }

    /// Creates an NFA accepting the complement of the given regex
    fn comp(&mut self, r: &Regex, builder: &mut ReBuilder) {
        self.compile_rec(r, builder);
        // Complement the NFA we just compiled
        let comp = complement(&self.nfa);
        self.nfa = comp;
    }

    /// Creates an NFA accepting the difference of the two given regexes
    fn diff(&mut self, r1: &Regex, r2: &Regex) {
        self.compile_rec(r1, &mut ReBuilder::default());
        let nfa1 = self.nfa.clone();
        self.compile_rec(r2, &mut ReBuilder::default());
        let nfa2 = self.nfa.clone();

        // The difference is the intersection of the first automaton with the complement of the second automaton
        let comp = complement(&nfa2);
        let inter = intersect(&nfa1, &comp);
        self.nfa = inter;
    }

    /// Creates an NFA accepting the loop of the given regex
    /// The loop is the union of the `lower`th to `upper`th power of the regex.
    fn loop_(&mut self, regex: &Regex, lower: u32, upper: u32, builder: &mut ReBuilder) {
        // Concat the automaton u times, but after lower times, insert an epsilon transition to the final final state
        let q0 = self.nfa.new_state();
        let mut optionals = HashSet::new();
        let mut last_finals = HashSet::new();
        last_finals.insert(q0);

        for i in 1..=upper {
            self.compile_rec(regex, builder);
            if let Some(initial) = self.nfa.initial() {
                for qf in last_finals {
                    self.nfa
                        .add_transition(qf, initial, TransitionType::Epsilon)
                        .unwrap();
                }
            }
            if i >= lower {
                optionals.extend(self.nfa.finals());
            }
            last_finals = self.nfa.finals.clone();
            self.nfa.finals.clear();
        }
        let qf = self.nfa.new_state();
        for final_ in &last_finals {
            self.nfa
                .add_transition(*final_, qf, TransitionType::Epsilon)
                .unwrap();
        }
        for opt in optionals.iter().filter(|q| !last_finals.contains(q)) {
            self.nfa
                .add_transition(*opt, qf, TransitionType::Epsilon)
                .unwrap();
        }
        self.nfa.set_initial(q0).unwrap();
        self.nfa.finals = HashSet::new();
        self.nfa.add_final(qf).unwrap();
        if lower == 0 {
            self.nfa.add_final(q0).unwrap();
        }
    }

    fn compile_rec(&mut self, regex: &Regex, builder: &mut ReBuilder) {
        if let Some(uoc) = union_of_chars(regex) {
            self.ranges(uoc);
            return;
        }
        match regex.op() {
            ReOp::Literal(w) => self.word(w),
            ReOp::None => self.none(),
            ReOp::All => self.all(),
            ReOp::Any => self.any(),
            ReOp::Concat(rs) => self.concat(rs, builder),
            ReOp::Union(rs) => self.union(rs, builder),
            ReOp::Inter(rs) => self.inter(rs, builder),
            ReOp::Star(r) => self.star(r, builder),
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
    }

    /// Compiles the given regex into an NFA.
    pub fn compile(mut self, regex: &Regex, builder: &mut ReBuilder) -> NFA {
        self.compile_rec(regex, builder);
        self.nfa.trim()
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
        let nfa = thompson.compile(&re, builder);
        for p in pos {
            assert!(nfa.accepts(&p), "Expected NFA for {} to accept {}", re, p);
        }
        for n in neg {
            assert!(
                !nfa.accepts(&n),
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
        let re = builder.any_char();
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
            alph.insert(r.clone());
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
                test_acceptance(&re, &mut builder, &[SmtChar::from(c).into()], &[]);
            }
        }

        TestResult::passed()
    }

    #[test]
    fn test_compile_union_chars() {
        let mut builder = ReBuilder::default();
        let rw1 = builder.to_re("f".into());
        let rw2 = builder.to_re("b".into());
        let re = builder.union(smallvec![rw1.clone(), rw2.clone()]);
        let th = Thompson::default();
        let nfa = th.compile(&re, &mut builder);
        println!("{}", nfa.dot());
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
        let any = builder.any_char();
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
                println!("{}", nfa.dot());
                assert!(!nfa.accepts(&w.repeat(i)), "{}", i);
            } else {
                assert!(nfa.accepts(&w.repeat(i)), "{}", i);
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
