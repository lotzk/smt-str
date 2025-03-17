//! Brzozowski derivatives for regular expression.
use std::collections::HashMap;

use smallvec::{smallvec, SmallVec};

use crate::{SmtChar, SmtString};

use super::{union_of_chars, ReBuilder, ReOp, Regex};

#[derive(Debug, Clone, Default)]
pub(crate) struct Deriver {
    cache: HashMap<(Regex, SmtChar), Regex>,
}

impl Deriver {
    pub fn deriv(&mut self, r: &Regex, c: SmtChar, builder: &mut ReBuilder) -> Regex {
        if let Some(derived) = self.cache.get(&(r.clone(), c)) {
            derived.clone()
        } else if let Some(chrs) = union_of_chars(r) {
            if chrs.iter().any(|r| r.contains(c)) {
                return builder.epsilon();
            } else {
                return builder.none();
            }
        } else {
            let deriv = match r.op() {
                ReOp::Literal(w) => {
                    if w.first() == Some(c) {
                        builder.to_re(w.drop(1))
                    } else {
                        builder.none()
                    }
                }
                ReOp::None => builder.none(),
                ReOp::All => builder.all(),
                ReOp::Any => builder.epsilon(),
                ReOp::Union(rs) => {
                    let ch: SmallVec<[Regex; 2]> =
                        rs.iter().map(|r| self.deriv(r, c, builder)).collect();
                    builder.union(ch)
                }
                ReOp::Concat(rs) => {
                    let mut union = smallvec![];
                    for (n, r) in rs.iter().enumerate() {
                        let current = self.deriv(r, c, builder);
                        let remaining = rs.iter().skip(n + 1);
                        // Concatenate the current deriv with the remaining children
                        let mut v = SmallVec::with_capacity(remaining.len() + 1);
                        v.push(current);
                        v.extend(remaining.cloned());
                        let deriv = builder.concat(v);
                        union.push(deriv);
                        // Continue deriving the next child if the current child is nullable
                        if !r.nullable() {
                            break;
                        }
                    }

                    if union.len() == 1 {
                        union.pop().unwrap()
                    } else if !union.is_empty() {
                        builder.union(union)
                    } else {
                        // That only happens if the concatenation is empty, at which point the derivative is none
                        builder.none()
                    }
                }
                ReOp::Inter(rs) => {
                    let ch = rs.iter().map(|r| self.deriv(r, c, builder)).collect();
                    builder.inter(ch)
                }
                ReOp::Star(r) | ReOp::Plus(r) => {
                    let rderiv = self.deriv(r, c, builder);
                    let star = builder.star(r.clone());
                    builder.concat(smallvec![rderiv, star])
                }
                ReOp::Opt(r) => self.deriv(r, c, builder),
                ReOp::Range(r) => {
                    let l = r.start();
                    let u = r.end();
                    if l <= c && c <= u {
                        builder.epsilon()
                    } else {
                        builder.none()
                    }
                }
                ReOp::Comp(r) => {
                    let r = self.deriv(r, c, builder);
                    builder.comp(r)
                }
                ReOp::Diff(d1, d2) => {
                    // rewrite as intersection with complement
                    let d2c = builder.comp(d2.clone());
                    let inter = builder.inter(smallvec![d1.clone(), d2c.clone()]);
                    self.deriv(&inter, c, builder)
                }
                ReOp::Pow(r, e) => {
                    if *e == 0 {
                        builder.none()
                    } else {
                        let derivr = self.deriv(r, c, builder);
                        let pow = builder.pow(r.clone(), e - 1);
                        builder.concat(smallvec![derivr, pow])
                    }
                }
                ReOp::Loop(r, l, u) => {
                    if l > u || *u == 0 {
                        // is either none or epsilon, in both cases the derivative is none
                        return builder.none();
                    } else {
                        let rderiv = self.deriv(r, c, builder);
                        let loop_ =
                            builder.loop_(r.clone(), l.saturating_sub(1), u.saturating_sub(1));
                        builder.concat(smallvec![rderiv, loop_])
                    }
                }
            };
            self.cache.insert((r.clone(), c), deriv.clone());
            deriv
        }
    }
}

/// Calculates the derivative of a regex w.r.t. to a symbol.
/// The derivative of a regex `r` w.r.t. to a symbol `c` is a regex that matches the suffix of all words that `r` matches when `c` is removed from the beginning.
/// New expressions are built using the given [RegexBuilder].
/// It is a precondition that the given regex is managed by the same [RegexBuilder] as the one passed to this function.
pub fn deriv(r: &Regex, c: impl Into<SmtChar>, builder: &mut ReBuilder) -> Regex {
    let mut deriver = Deriver::default();
    deriver.deriv(r, c.into(), builder)
}

/// Calculates the derivative of a regex w.r.t. to a word by repeatedly calculating the derivative of the regex w.r.t. to each symbol in the word.
/// All derivatives are built using the given [RegexBuilder].
/// It is a precondition that the given regex is managed by the same [RegexBuilder] as the one passed to this function.
pub fn deriv_word(r: &Regex, w: impl Into<SmtString>, builder: &mut ReBuilder) -> Regex {
    let mut result = r.clone();
    let w = w.into();
    for c in w.iter() {
        result = deriv(&result, *c, builder);
    }
    result
}

#[cfg(test)]
mod test {

    use std::rc::Rc;

    use quickcheck_macros::quickcheck;

    use super::*;
    #[test]
    fn deriv_const() {
        let mut builder = ReBuilder::non_optimizing();
        let r = builder.to_re("foo".into());
        let expected = builder.to_re("oo".into());
        let derived = deriv(&r, 'f', &mut builder);
        assert_eq!(derived, expected);
    }

    #[test]
    fn deriv_const_builder() {
        let mut builder = ReBuilder::non_optimizing();
        let r = builder.to_re("foo".into());
        let expected = builder.to_re("oo".into());
        let derived = deriv(&r, 'f', &mut builder);
        assert!(Rc::ptr_eq(&derived, &expected));
    }

    #[quickcheck]
    fn deriv_none(c: char) {
        let mut builder = ReBuilder::non_optimizing();
        let r = builder.none();
        let expected = builder.none();
        assert_eq!(deriv(&r, c, &mut builder), expected);
    }

    #[quickcheck]
    fn deriv_all(c: char) {
        let mut builder = ReBuilder::non_optimizing();
        let r = builder.all();
        let expected = builder.all();
        assert_eq!(deriv(&r, c, &mut builder), expected);
    }

    #[test]
    fn deriv_all_char_builder() {
        let mut builder = ReBuilder::non_optimizing();
        let r = builder.any_char();
        let expected = builder.epsilon();
        assert_eq!(deriv(&r, 'f', &mut builder), expected);
    }

    #[test]
    fn deriv_range() {
        let mut builder = ReBuilder::non_optimizing();
        let r = builder.range_from_to('a', 'z');
        assert_eq!(deriv(&r, 'f', &mut builder), builder.epsilon());
        assert_eq!(deriv(&r, '1', &mut builder), builder.none());
    }

    #[test]
    fn deriv_concat() {
        let mut builder = ReBuilder::non_optimizing();

        let chs = smallvec![builder.to_re("foo".into()), builder.to_re("bar".into())];
        let r = builder.concat(chs);

        let chs = smallvec![builder.to_re("oo".into()), builder.to_re("bar".into())];
        let expected = builder.concat(chs);

        assert_eq!(deriv(&r, 'f', &mut builder), expected);
        assert!(deriv(&r, 'g', &mut builder).none().unwrap());
        assert!(deriv(&r, 'o', &mut builder).none().unwrap());
    }

    #[test]
    fn deriv_concat_nullable() {
        let mut builder = ReBuilder::default();
        let foo = builder.to_re("foo".into());

        // foo?
        let opt = builder.opt(foo.clone());

        let r = smallvec![opt.clone(), builder.to_re("far".into())];
        // (foo)?far
        let r = builder.concat(r);

        // oofar|ar
        let expected = smallvec![builder.to_re("oo".into()), builder.to_re("far".into())];
        let concat = builder.concat(expected);
        let expected = smallvec![concat, builder.to_re("ar".into())];
        let expected = builder.union(expected);

        let derived = deriv(&r, 'f', &mut builder);
        assert_eq!(
            expected.op(),
            derived.op(),
            "Expected: {} but was {}",
            expected,
            derived
        );
        assert!(deriv(&r, 'g', &mut builder).none().unwrap());
        assert!(deriv(&r, 'o', &mut builder).none().unwrap());
    }

    #[test]
    fn deriv_union() {
        let mut builder = ReBuilder::non_optimizing();
        let foo = builder.to_re("foo".into());
        let bar = builder.to_re("bar".into());
        let baz = builder.to_re("baz".into());
        let r = builder.union(smallvec![foo, bar, baz]);

        let ar = builder.to_re("ar".into());
        let az = builder.to_re("az".into());
        let none = builder.none();
        let expected = builder.union(smallvec![none.clone(), ar, az]);
        assert_eq!(deriv(&r, 'b', &mut builder), expected);

        let oo = builder.to_re("oo".into());
        let expected = builder.union(smallvec![oo, none.clone(), none.clone()]);
        assert_eq!(deriv(&r, 'f', &mut builder), expected);

        assert!(deriv(&r, 'g', &mut builder).none().unwrap());
    }

    #[test]
    fn deriv_star() {
        let mut builder = ReBuilder::non_optimizing();
        let foo = builder.to_re("foo".into());
        let r = builder.star(foo.clone());

        let oo = builder.to_re("oo".into());
        let expected = builder.concat(smallvec![oo, r.clone()]);
        assert_eq!(deriv(&r, 'f', &mut builder), expected);
        assert!(deriv(&r, 'g', &mut builder).none().unwrap());
    }

    #[test]
    fn deriv_plus() {
        let mut builder = ReBuilder::non_optimizing();
        let foo = builder.to_re("foo".into());
        let r = builder.plus(foo.clone());

        let expected = smallvec![builder.to_re("oo".into()), builder.star(foo.clone())];
        let expected = builder.concat(expected);
        assert_eq!(deriv(&r, 'f', &mut builder), expected);
        assert!(deriv(&r, 'g', &mut builder).none().unwrap());
    }

    #[test]
    fn deriv_opt() {
        let mut builder = ReBuilder::non_optimizing();
        let foo = builder.to_re("foo".into());
        let r = builder.opt(foo.clone());

        let expected = builder.to_re("oo".into());
        assert_eq!(deriv(&r, 'f', &mut builder), expected);
        assert!(deriv(&r, 'g', &mut builder).none().unwrap());
    }

    #[test]
    fn deriv_pow() {
        let mut builder = ReBuilder::non_optimizing();
        let foo = builder.to_re("foo".into());
        let r = builder.pow(foo.clone(), 3);

        let oo = builder.to_re("oo".into());
        let pow2 = builder.pow(foo.clone(), 2);
        let expected = builder.concat(smallvec![oo, pow2]);
        assert_eq!(deriv(&r, 'f', &mut builder), expected);
        assert!(deriv(&r, 'g', &mut builder).none().unwrap());
    }

    #[test]
    fn deriv_inter() {
        let mut builder = ReBuilder::non_optimizing();
        let foo = builder.to_re("foo".into());
        let far = builder.to_re("far".into());
        let r = builder.inter(smallvec![foo.clone(), far.clone()]);
        let oo = builder.to_re("oo".into());
        let ar = builder.to_re("ar".into());
        let expected = builder.inter(smallvec![oo, ar]);
        assert_eq!(deriv(&r, 'f', &mut builder), expected);
        assert!(deriv(&r, 'g', &mut builder).none().unwrap());
    }

    #[test]
    fn deriv_diff() {
        let mut builder = ReBuilder::non_optimizing();
        let foo = builder.to_re("foo".into());
        let far = builder.to_re("far".into());
        let r = builder.diff(foo.clone(), far.clone());
        let oo = builder.to_re("oo".into());
        let ar = builder.to_re("ar".into());
        let ar_comp = builder.comp(ar.clone());
        let expected = builder.inter(smallvec![oo, ar_comp]);
        assert_eq!(deriv(&r, 'f', &mut builder), expected);
        assert!(deriv(&r, 'g', &mut builder).none().unwrap());
    }

    #[test]
    fn deriv_pow_0() {
        let mut builder = ReBuilder::non_optimizing();
        let foo = builder.to_re("foo".into());
        let r = builder.pow(foo.clone(), 0);
        assert_eq!(deriv(&r, 'f', &mut builder), builder.none());
    }

    #[test]
    fn deriv_loop() {
        let mut builder = ReBuilder::non_optimizing();
        let foo = builder.to_re("foo".into());
        let r = builder.loop_(foo.clone(), 1, 2);

        let loop2 = builder.loop_(foo.clone(), 0, 1);
        let oo = builder.to_re("oo".into());
        let expected = builder.concat(smallvec![oo, loop2]);
        assert_eq!(deriv(&r, 'f', &mut builder), expected);
        assert!(deriv(&r, 'b', &mut builder).none().unwrap());
    }

    #[test]
    fn deriv_loop_empty() {
        let mut builder = ReBuilder::non_optimizing();
        let foo = builder.to_re("foo".into());
        let r = builder.loop_(foo.clone(), 0, 0);
        assert_eq!(deriv(&r, 'f', &mut builder), builder.none());
    }

    #[test]
    fn deriv_loop_empty2() {
        let mut builder = ReBuilder::non_optimizing();
        let foo = builder.to_re("foo".into());
        let r = builder.loop_(foo.clone(), 1, 0);
        assert_eq!(deriv(&r, 'f', &mut builder), builder.none());
    }

    #[quickcheck]
    fn deriv_loops(l: u32, u: u32) {
        let l = l % 100;
        let u = u % 100;
        let mut builder = ReBuilder::non_optimizing();
        let foo = builder.to_re("foo".into());
        let r = builder.loop_(foo.clone(), l, u);

        if l > u || u == 0 {
            assert_eq!(deriv(&r, 'f', &mut builder), builder.none());
        } else {
            let loop_ = builder.loop_(foo.clone(), l.saturating_sub(1), u.saturating_sub(1));
            let oo = builder.to_re("oo".into());
            let expected = builder.concat(smallvec![oo, loop_]);
            assert_eq!(deriv(&r, 'f', &mut builder), expected);
        }
    }
}
