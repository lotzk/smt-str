use std::collections::{HashMap, HashSet};
use std::vec;

use crate::alphabet::CharRange;
use crate::automata::NState;
use crate::re::{union_of_chars, ReBuilder, ReOp, Regex};
use crate::{SmtChar, SmtString};

use super::comp::complement;
use super::{Automaton, AutomatonError, State, Transition, TransitionType, NFA};

#[derive(Default)]
pub struct Thompson {
    cache: HashMap<Regex, NFA>,
}

impl Thompson {
    /// Creates an NFA accepting no words
    fn none() -> Result<NFA, AutomatonError> {
        let mut nfa = Automaton::new();
        let id = nfa.new_state();
        nfa.set_initial(id);
        Ok(nfa)
    }

    /// Creates an NFA accepting exactly the given word
    fn word(word: &SmtString) -> Result<NFA, AutomatonError> {
        fn word2states(word: &SmtString, mut states: Vec<NState>) -> Vec<NState> {
            if let Some(c) = word.first() {
                // add transition to next state
                states.push(NState::with_transitions(vec![Transition::char(
                    c,
                    states.len() + 1,
                )]));
                word2states(&word.drop(1), states)
            } else {
                states.push(NState::default());
                states
            }
        }
        let states = word2states(word, vec![]);
        let mut nfa = Automaton::from_states(states);
        nfa.set_initial(0);
        nfa.add_final(nfa.states().len() - 1);
        Ok(nfa)
    }

    fn range(l: SmtChar, u: SmtChar) -> Result<NFA, AutomatonError> {
        let states = vec![
            NState::with_transitions(vec![Transition::range_from(l, u, 1)]),
            NState::default(),
        ];
        let mut nfa = Automaton::from_states(states);
        nfa.set_initial(0);
        nfa.add_final(1);
        Ok(nfa)
    }

    fn not_range(range: CharRange) -> Result<NFA, AutomatonError> {
        let states = vec![
            NState::with_transitions(vec![
                Transition::not_range(range, 2),
                Transition::range(range, 1),
            ]),
            NState::with_transitions(vec![Transition::any(2)]),
            NState::with_transitions(vec![Transition::any(2)]), // self-loop
        ];
        let mut nfa = Automaton::from_states(states);
        nfa.set_initial(0);
        nfa.add_final(0);
        nfa.add_final(2);
        Ok(nfa)
    }

    /// Creates an NFA accepting any symbol, i.e., any word of length 1
    fn any() -> Result<NFA, AutomatonError> {
        let states = vec![
            NState::with_transitions(vec![Transition::any(1)]),
            NState::default(),
        ];
        let mut nfa = Automaton::from_states(states);
        nfa.set_initial(0);
        nfa.add_final(1);
        Ok(nfa)
    }

    /// Creates the universal NFA
    fn all() -> Result<NFA, AutomatonError> {
        let states = vec![NState::with_transitions(vec![Transition::any(0)])];
        let mut nfa = Automaton::from_states(states);
        nfa.set_initial(0);
        nfa.add_final(0);
        Ok(nfa)
    }

    fn union_of_chars(rs: Vec<CharRange>) -> Result<NFA, AutomatonError> {
        let mut nfa = Automaton::new();
        let q0 = nfa.new_state();
        nfa.set_initial(q0);
        let qf = nfa.new_state();
        nfa.add_final(qf);
        for r in rs {
            nfa.add_transition(q0, qf, TransitionType::Range(r))?;
        }
        Ok(nfa)
    }

    /// Creates an NFA accepting the union of the given NFAs
    fn union(&mut self, rs: &[Regex], builder: &mut ReBuilder) -> Result<NFA, AutomatonError> {
        if rs.is_empty() {
            return Self::word(&SmtString::empty());
        }
        let mut states = vec![NState::default()];
        let mut finals = HashSet::new();
        for r in rs {
            let nfa = self.compile(r, builder)?;
            let offset = states.len();
            for (id, state) in nfa.iter_states() {
                states.push(state.offset_transitions(offset));
                if nfa.is_initial(id) {
                    let states_len = states.len();
                    states
                        .get_mut(0)
                        .unwrap()
                        .add_transition(Transition::epsilon(states_len - 1));
                }
                if nfa.is_final(id) {
                    finals.insert(states.len() - 1);
                }
            }
        }
        let mut nfa = Automaton::from_states(states);
        nfa.set_initial(0);
        for qf in finals {
            nfa.add_final(qf);
        }
        Ok(nfa)
    }

    /// Creates an NFA accepting the concatenation of the given NFAs
    fn concat(&mut self, rs: &[Regex], builder: &mut ReBuilder) -> Result<NFA, AutomatonError> {
        if rs.is_empty() {
            return Self::word(&SmtString::empty());
        }
        let mut states = vec![];
        let mut initial = None;
        let mut finals = HashSet::new();
        for r in rs {
            let nfa = self.compile(r, builder)?;
            let offset = states.len();
            let mut new_finals = HashSet::new();
            for (id, state) in nfa.iter_states() {
                states.push(state.offset_transitions(offset));
                let new_id = states.len() - 1;
                if nfa.is_initial(id) {
                    for qfi in &finals {
                        let qfi: &mut NState = states.get_mut(*qfi).unwrap();
                        qfi.add_transition(Transition::epsilon(new_id));
                    }
                }
                if nfa.is_final(id) {
                    new_finals.insert(new_id);
                }
                if nfa.is_initial(id) && initial.is_none() {
                    // first initial state
                    initial = Some(new_id);
                }
            }
            finals = new_finals;
        }
        let mut nfa = Automaton::from_states(states);
        nfa.set_initial(initial.unwrap());
        for f in finals {
            nfa.add_final(f);
        }

        Ok(nfa)
    }

    fn inter(&mut self, rs: &[Regex], builder: &mut ReBuilder) -> Result<NFA, AutomatonError> {
        if rs.is_empty() {
            return Self::none();
        }

        // todo: optimize
        // - if one of the regexes is empty, return empty
        // - if one of the regexes is all, skip it

        let mut nfa = self.compile(&rs[0], builder)?.remove_epsilons()?;

        for r in rs.iter().skip(1) {
            nfa = nfa.intersect(&self.compile(r, builder)?.remove_epsilons()?)?;
        }
        Ok(nfa)
    }

    /// Creates an NFA accepting the Kleene star of the given NFA
    fn star(&mut self, r: &Regex, builder: &mut ReBuilder) -> Result<NFA, AutomatonError> {
        let mut nfa = self.compile(r, builder)?;
        if nfa.initial().is_none() {
            panic!("Initial state is None: {}", r);
        }
        let new_final = nfa.new_state();
        let new_init = nfa.new_state();
        nfa.add_transition(new_init, nfa.initial().unwrap(), TransitionType::Epsilon)?;
        nfa.add_transition(new_init, new_final, TransitionType::Epsilon)?;
        for final_ in nfa.finals().clone() {
            let initial = nfa.initial().unwrap();
            nfa.add_transition(final_, initial, TransitionType::Epsilon)?;
            nfa.add_transition(final_, new_final, TransitionType::Epsilon)?;
        }
        nfa.clear_finals();
        nfa.add_final(new_final);
        nfa.set_initial(new_init);

        Ok(nfa)
    }

    fn plus(&mut self, r: &Regex, builder: &mut ReBuilder) -> Result<NFA, AutomatonError> {
        // Same as Kleene star but we omit the transition that goes from the new initial to the new final, i.e., the transition that "skips" the inner regex.
        let mut nfa = self.compile(r, builder)?;
        if nfa.initial().is_none() {
            panic!("Initial state is None: {}", r);
        }
        let new_final = nfa.new_state();
        let new_init = nfa.new_state();
        nfa.add_transition(new_init, nfa.initial().unwrap(), TransitionType::Epsilon)?;

        for final_ in nfa.finals().clone() {
            let initial = nfa.initial().unwrap();
            nfa.add_transition(final_, initial, TransitionType::Epsilon)?;
            nfa.add_transition(final_, new_final, TransitionType::Epsilon)?;
        }
        nfa.clear_finals();
        nfa.add_final(new_final);
        nfa.set_initial(new_init);

        Ok(nfa)
    }

    fn opt(&mut self, r: &Regex, builder: &mut ReBuilder) -> Result<NFA, AutomatonError> {
        // let epsi = builder.epsilon();
        // let union = builder.union(vec![r.clone(), epsi]);
        // self.compile(&union, builder)

        // s -e-> r -e-> f*; s -e-> f*
        let mut nfa = self.compile(r, builder)?;
        let new_initial = nfa.new_state();
        let new_final = nfa.new_state();

        // Transition from new initial to old initial
        if let Some(old_initial) = nfa.initial() {
            nfa.get_state_mut(new_initial)
                .unwrap()
                .add_transition(Transition::epsilon(old_initial));
        }
        // Transition from new initial to new final
        nfa.get_state_mut(new_initial)
            .unwrap()
            .add_transition(Transition::epsilon(new_final));

        // Transitions from old finals to new final
        for final_ in nfa.finals().clone() {
            nfa.get_state_mut(final_)
                .unwrap()
                .add_transition(Transition::epsilon(new_final));
        }
        nfa.clear_finals();
        nfa.add_final(new_final);
        nfa.set_initial(new_initial);
        Ok(nfa)
    }

    fn pow(&mut self, r: &Regex, n: u32, builder: &mut ReBuilder) -> Result<NFA, AutomatonError> {
        let concat = vec![r.clone(); n as usize];
        self.concat(&concat, builder)
    }

    fn comp(&mut self, r: &Regex, builder: &mut ReBuilder) -> Result<NFA, AutomatonError> {
        let base = self.compile(r, builder)?.remove_epsilons()?;
        let det = base.determinize()?;
        let comp = complement(&det)?;
        Ok(comp.into())
    }

    fn diff(&mut self, r1: &Regex, r2: &Regex) -> Result<NFA, AutomatonError> {
        let mut nfa1 = self
            .compile(r1, &mut ReBuilder::default())?
            .remove_epsilons()?;
        let nfa2 = self
            .compile(r2, &mut ReBuilder::default())?
            .remove_epsilons()?
            .determinize()?;
        let nfa2 = complement(&nfa2)?;
        nfa1 = nfa1.intersect(&nfa2.into())?;
        Ok(nfa1)
    }

    fn loop_(
        &mut self,
        regex: &Regex,
        lower: u32,
        upper: u32,
        builder: &mut ReBuilder,
    ) -> Result<NFA, AutomatonError> {
        // Rewrite the loop as a union of powers

        // let mut union = vec![];
        // for i in lower..=upper {
        //     union.push(builder.pow(regex.clone(), i));
        // }
        // // We need a new regex to avoid using the cache, as the new regex is not the same as the old one
        // self.union(&union, builder)

        //Rewrite as concat of opts
        let mut rs = vec![];
        for i in 0..upper {
            if i < lower {
                rs.push(regex.clone());
            } else {
                rs.push(builder.opt(regex.clone()));
            }
        }
        self.concat(&rs, builder)
    }

    pub fn compile(
        &mut self,
        regex: &Regex,
        builder: &mut ReBuilder,
    ) -> Result<NFA, AutomatonError> {
        if let Some(nfa) = self.cache.get(regex) {
            return Ok(nfa.clone());
        }
        if let Some(uoc) = union_of_chars(regex) {
            return Self::union_of_chars(uoc);
        }
        let m = match regex.op() {
            ReOp::Literal(w) => Self::word(w),
            ReOp::None => Self::none(),
            ReOp::All => Self::all(),
            ReOp::Any => Self::any(),
            ReOp::Concat(rs) => self.concat(rs, builder),
            ReOp::Union(rs) => self.union(rs, builder),
            ReOp::Inter(rs) => self.inter(rs, builder),
            ReOp::Star(r) => self.star(r, builder),
            ReOp::Plus(r) => self.plus(r, builder),
            ReOp::Opt(r) => self.opt(r, builder),
            ReOp::Range(r) => Self::range(r.start(), r.end()),
            ReOp::Comp(c) => match c.op() {
                ReOp::Range(r) => return Self::not_range(*r),
                ReOp::Literal(w) if w.len() == 1 => {
                    return Self::not_range(CharRange::new(w[0], w[0]))
                }
                _ => self.comp(c, builder),
            },
            ReOp::Diff(r1, r2) => self.diff(r1, r2),
            ReOp::Pow(r, n) => self.pow(r, *n, builder),
            ReOp::Loop(r, l, u) => self.loop_(r, *l, *u, builder),
        }?;
        self.cache.insert(regex.clone(), m.clone());
        Ok(m)
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

    fn test_acceptance(re: &Regex, builder: &mut ReBuilder, pos: &[SmtString], neg: &[SmtString]) {
        let mut thompson = Thompson::default();
        let nfa = thompson.compile(&re, builder).unwrap();
        for p in pos {
            println!("{}", nfa.dot().unwrap());
            assert!(
                nfa.accepts(&p).unwrap(),
                "Expected NFA for {} to accept {}",
                re,
                p
            );
        }
        for n in neg {
            assert!(
                !nfa.accepts(&n).unwrap(),
                "Expected NFA for {} to reject {}\n{}",
                re,
                n,
                nfa.dot().unwrap()
            );
        }
    }

    #[quickcheck]
    fn word(w: SmtString) {
        let mut builder = ReBuilder::default();
        let re = builder.to_re(w.clone());
        test_acceptance(&re, &mut builder, &[w], &[]);
    }

    #[test]
    fn any() {
        let mut builder = ReBuilder::default();
        let re = builder.any_char();
        test_acceptance(&re, &mut builder, &["a".into()], &["".into(), "ab".into()]);
    }

    #[quickcheck]
    fn all(w: SmtString) {
        let mut builder = ReBuilder::default();
        let re = builder.all();
        test_acceptance(&re, &mut builder, &[w], &[]);
    }

    #[quickcheck]
    fn none(w: SmtString) {
        let mut builder = ReBuilder::default();
        let re = builder.none();
        test_acceptance(&re, &mut builder, &[], &[w]);
    }

    #[quickcheck]
    fn range(l: SmtChar, u: SmtChar) -> TestResult {
        if l > u {
            return TestResult::discard();
        }

        let mut builder = ReBuilder::default();
        let re = builder.range_from_to(l, u);

        // Pick random chars in the range
        let mut rng = rand::rng();
        let dist = rand::distr::Uniform::new_inclusive(l.0, u.0).unwrap();

        for _ in 0..100 {
            let c = dist.sample(&mut rng);
            test_acceptance(&re, &mut builder, &[SmtChar::from(c).into()], &[]);
        }

        TestResult::passed()
    }

    #[test]
    fn not_char() {
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
    fn not_range(r: CharRange) -> TestResult {
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

    #[quickcheck]
    fn union(w1: SmtString, w2: SmtString, wneg: SmtString) {
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
    fn concat(w1: SmtString, w2: SmtString, wneg: SmtString) {
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
    fn star(w: SmtString, n: usize) {
        let mut builder = ReBuilder::default();
        let rw = builder.to_re(w.clone());
        let re = builder.star(rw);
        test_acceptance(&re, &mut builder, &[w.repeat(n % 100)], &[]);
    }

    #[quickcheck]
    fn plus(w: SmtString, n: usize) {
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
    fn plus_of_any(w: SmtString) {
        let mut builder = ReBuilder::default();
        let any = builder.any_char();
        let re = builder.plus(any);
        if w.is_empty() {
            test_acceptance(&re, &mut builder, &[], &[w]);
        } else {
            test_acceptance(&re, &mut builder, &[w], &[]);
        }
    }
}
