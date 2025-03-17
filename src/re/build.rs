use smallvec::smallvec;

use crate::SmtChar;

use super::*;
use std::collections::{BTreeSet, HashMap};

/// A builder for regular expressions that constructs unique regex instances.
/// It is the only way to create regex instances.
#[derive(Debug)]
pub struct ReBuilder {
    registry: Registry,
    optimize: bool,

    /* base expressions */
    re_none: Regex,
    re_all: Regex,
    re_allchar: Regex,
    re_epsilon: Regex,
}

impl Default for ReBuilder {
    /// Creates a new `RegexBuilder` with an empty internal registry.
    /// By default, on-the-fly optimization is enabled.
    fn default() -> Self {
        let mut registry = Registry::new();
        let re_none = registry.intern(ReOp::None);
        let re_all = registry.intern(ReOp::All);
        let re_allchar = registry.intern(ReOp::Any);
        let re_epsilon = registry.intern(ReOp::Literal(SmtString::empty()));

        Self {
            registry,
            optimize: true,
            re_none,
            re_all,
            re_allchar,
            re_epsilon,
        }
    }
}

impl ReBuilder {
    pub fn non_optimizing() -> Self {
        Self {
            optimize: false,
            ..Default::default()
        }
    }

    fn intern(&mut self, regex: ReOp) -> Regex {
        self.registry.intern(regex)
    }

    /// Checks if the builder manages the given regex.
    /// Returns true if the regex was constructed by this builder.
    /// Returns false otherwise.
    pub fn manages(&self, regex: &Regex) -> bool {
        // We need to check if the registry has the regex pattern stored with the same id.
        // After that we need to check that that the stored regex is pointer equal to the given regex
        // Only checking the id is not enough as the id is only unique to the builder instance
        // We could also recurse the structure and check if all children are managed by this builder.
        match self.registry.registry.get(regex.op()) {
            Some(r) => Rc::ptr_eq(r, regex),
            None => false,
        }
    }

    /// Builds a regex that was constructed by a different builder.
    /// Returns a new regex managed by this builder.
    pub fn regex(&mut self, regex: &Regex) -> Regex {
        match regex.op() {
            ReOp::Literal(w) => self.to_re(w.clone()),
            ReOp::None => self.none(),
            ReOp::All => self.all(),
            ReOp::Any => self.any_char(),
            ReOp::Concat(rs) => {
                let rs = rs.iter().map(|r| self.regex(r)).collect();
                self.concat(rs)
            }
            ReOp::Union(rs) => {
                let rs = rs.iter().map(|r| self.regex(r)).collect();
                self.union(rs)
            }
            ReOp::Inter(rs) => {
                let rs = rs.iter().map(|r| self.regex(r)).collect();
                self.inter(rs)
            }
            ReOp::Star(r) => {
                let r = self.regex(r);
                self.star(r)
            }
            ReOp::Plus(r) => {
                let r = self.regex(r);
                self.plus(r)
            }
            ReOp::Opt(r) => {
                let r = self.regex(r);
                self.opt(r)
            }
            ReOp::Range(r) => self.range_from_to(r.start(), r.end()),
            ReOp::Comp(r) => {
                let r = self.regex(r);
                self.comp(r)
            }
            ReOp::Diff(r1, r2) => {
                let r1 = self.regex(r1);
                let r2 = self.regex(r2);
                self.diff(r1, r2)
            }
            ReOp::Pow(r, e) => {
                let r = self.regex(r);
                self.pow(r, *e)
            }
            ReOp::Loop(r, l, u) => {
                let r = self.regex(r);
                self.loop_(r, *l, *u)
            }
        }
    }

    /// Constructs a regular expression of a constant word.
    pub fn to_re(&mut self, w: SmtString) -> Regex {
        self.intern(ReOp::Literal(w))
    }

    /// Constructs a regular expression denoting the empty word.
    pub fn epsilon(&self) -> Regex {
        self.re_epsilon.clone()
    }

    /// Constructs a regular expression denoting the empty set.
    pub fn none(&self) -> Regex {
        self.re_none.clone()
    }

    /// Constructs a regular expression denoting the set of all strings.
    pub fn all(&self) -> Regex {
        self.re_all.clone()
    }

    /// Constructs a regular expression denoting the set of all strings of length 1.
    pub fn any_char(&self) -> Regex {
        self.re_allchar.clone()
    }

    /// Constructs a regular expression denoting the set of all strings of length 1 such that that lexicographically bounded by the given parameters.
    pub fn range_from_to(&mut self, l: impl Into<SmtChar>, u: impl Into<SmtChar>) -> Regex {
        self.range(CharRange::new(l.into(), u.into()))
    }

    /// Constructs a regular expression denoting the set of all strings of length 1 such that that lexicographically bounded by the given parameters.
    pub fn range(&mut self, r: CharRange) -> Regex {
        if self.optimize {
            self.range_opt(r)
        } else {
            self.intern(ReOp::Range(r))
        }
    }

    fn range_opt(&mut self, r: CharRange) -> Regex {
        if r.is_empty() {
            self.none()
        } else if r.is_full() {
            self.any_char()
        } else {
            self.intern(ReOp::Range(r))
        }
    }

    /// Constructs a regular expression denoting the concatenation of the given regular expressions.
    pub fn concat(&mut self, rs: SmallVec<[Regex; 2]>) -> Regex {
        if self.optimize {
            self.concat_opt(rs)
        } else {
            self.intern(ReOp::Concat(rs))
        }
    }

    fn concat_opt(&mut self, rs: SmallVec<[Regex; 2]>) -> Regex {
        // Check if any of the children is the empty set, then the concatenation is the empty set
        if rs.iter().any(|i| i.none() == Some(true)) {
            return self.none();
        }

        // Filter out empty terms as they are the identity element of concatenation
        // Flatten nested concatenations
        let rs = rs
            .into_iter()
            .filter(|i| i.epsilon() != Some(true))
            .flat_map(|i| {
                if let ReOp::Concat(rs) = &i.op() {
                    rs.clone()
                } else {
                    smallvec![i]
                }
            })
            .collect_vec();

        // Collapse adjacent words
        // TODO: Do this in one iteration
        let mut collapsed: SmallVec<[Regex; 2]> = SmallVec::with_capacity(rs.len());
        let mut last_word: Option<SmtString> = None;
        for r in rs.into_iter() {
            if let ReOp::Literal(w) = &r.op() {
                if let Some(last) = last_word {
                    last_word = Some(last.concat(w));
                } else {
                    last_word = Some(w.clone());
                }
            } else {
                if let Some(last) = last_word {
                    collapsed.push(self.to_re(last));
                    last_word = None;
                }
                collapsed.push(r);
            }
        }
        if let Some(last) = last_word {
            collapsed.push(self.to_re(last));
        }

        if collapsed.is_empty() {
            self.epsilon()
        } else if collapsed.len() == 1 {
            collapsed[0].clone()
        } else {
            self.intern(ReOp::Concat(collapsed))
        }
    }

    /// Constructs a regular expression denoting the union of the given regular expressions.
    pub fn union(&mut self, rs: SmallVec<[Regex; 2]>) -> Regex {
        if self.optimize {
            self.union_opt(rs)
        } else {
            self.intern(ReOp::Union(rs))
        }
    }

    fn union_opt(&mut self, rs: SmallVec<[Regex; 2]>) -> Regex {
        // Filter out empty terms as they are the identity element of union

        // Filter out empty terms as they are the identity element of union
        // Collect the remaining terms into a set to deduplicate and make the order deterministic

        #[allow(clippy::mutable_key_type)]
        let rs: BTreeSet<Regex> = rs
            .into_iter()
            .filter(|i| i.none() != Some(true))
            .flat_map(|i| {
                if let ReOp::Union(rs) = &i.op() {
                    rs.clone()
                } else {
                    smallvec![i]
                }
            })
            .collect();

        let rs: SmallVec<[Rc<ReNode>; 2]> = rs.into_iter().collect();

        if rs.is_empty() {
            self.none()
        } else if rs.len() == 1 {
            rs[0].clone()
        } else {
            self.intern(ReOp::Union(rs))
        }
    }

    /// Constructs a regular expression denoting the intersection of the given regular expressions.
    pub fn inter(&mut self, rs: SmallVec<[Regex; 2]>) -> Regex {
        if self.optimize {
            self.inter_opt(rs)
        } else {
            self.intern(ReOp::Inter(rs))
        }
    }

    fn inter_opt(&mut self, rs: SmallVec<[Regex; 2]>) -> Regex {
        // Check if any of the children is the empty set, then the intersection is the empty set
        if rs.iter().any(|i| i.none() == Some(true)) {
            return self.none();
        }

        // Filter out re.all terms as they are the identity element of intersection
        // As as side effect, this also flattens nested intersections, and sorts them to make the order deterministic
        #[allow(clippy::mutable_key_type)]
        let rs: BTreeSet<_> = rs
            .into_iter()
            .filter(|i| i.universal() != Some(true))
            .flat_map(|i| {
                if let ReOp::Inter(rs) = &i.op() {
                    rs.clone()
                } else {
                    smallvec![i]
                }
            })
            .collect();

        let rs: SmallVec<[Rc<ReNode>; 2]> = rs.into_iter().collect();
        if rs.is_empty() {
            self.all()
        } else if rs.len() == 1 {
            rs[0].clone()
        } else {
            self.intern(ReOp::Inter(rs))
        }
    }

    /// Constructs a regular expression denoting the Kleene closure of the given regular expression.
    pub fn star(&mut self, r: Regex) -> Regex {
        if self.optimize {
            self.star_opt(r)
        } else {
            self.intern(ReOp::Star(r))
        }
    }

    fn star_opt(&mut self, r: Regex) -> Regex {
        if r.none().unwrap_or(false) || r.epsilon().unwrap_or(false) {
            self.epsilon()
        } else if let ReOp::Star(_) = &r.op() {
            // Flatten nested stars
            r.clone()
        } else if let ReOp::Plus(rr) = &r.op() {
            // Flatten (R+)* = R*
            self.intern(ReOp::Star(rr.clone()))
        } else {
            self.intern(ReOp::Star(r))
        }
    }

    /// Constructs a regular expression denoting the positive closure of the given regular expression.
    pub fn plus(&mut self, r: Regex) -> Regex {
        if self.optimize {
            self.plus_opt(r)
        } else {
            self.intern(ReOp::Plus(r))
        }
    }

    fn plus_opt(&mut self, r: Regex) -> Regex {
        if r.none().unwrap_or(false) {
            self.none()
        } else if r.epsilon().unwrap_or(false) {
            self.epsilon()
        } else if let ReOp::Plus(_) = &r.op() {
            // Flatten nested pluses
            r.clone()
        } else if let ReOp::Star(_) = &r.op() {
            // Flatten (R*)+ = R*
            r.clone()
        } else {
            self.intern(ReOp::Plus(r))
        }
    }

    /// Constructs a regular expression denoting the complement of the given regular expression.
    pub fn comp(&mut self, r: Regex) -> Regex {
        if self.optimize {
            self.comp_opt(r)
        } else {
            self.intern(ReOp::Comp(r))
        }
    }

    fn comp_opt(&mut self, r: Regex) -> Regex {
        if r.none().unwrap_or(false) {
            self.all()
        } else if r.universal().unwrap_or(false) {
            self.none()
        } else {
            self.intern(ReOp::Comp(r))
        }
    }

    /// Constructs a regular expression denoting the difference of the two given regular expressions.
    pub fn diff(&mut self, r1: Regex, r2: Regex) -> Regex {
        if self.optimize {
            self.diff_opt(r1, r2)
        } else {
            self.intern(ReOp::Diff(r1, r2))
        }
    }

    fn diff_opt(&mut self, r1: Regex, r2: Regex) -> Regex {
        if r1.none().unwrap_or(false) || r2.universal().unwrap_or(false) {
            self.none()
        } else if r2.none().unwrap_or(false) {
            r1.clone()
        } else {
            self.intern(ReOp::Diff(r1, r2))
        }
    }

    /// Constructs a regular expression that either accepts the given regular expression or the empty word.
    pub fn opt(&mut self, r: Regex) -> Regex {
        if self.optimize {
            self.opt_opt(r)
        } else {
            self.intern(ReOp::Opt(r))
        }
    }

    fn opt_opt(&mut self, r: Regex) -> Regex {
        if r.none().unwrap_or(false) || r.epsilon().unwrap_or(false) {
            self.epsilon()
        } else {
            self.intern(ReOp::Opt(r))
        }
    }

    /// Constructs a regular expression denoting the set of the n-th powers of the given regular expression, that is, the n-fold concatenation of the regular expression with itself.
    pub fn pow(&mut self, r: Regex, n: u32) -> Regex {
        if self.optimize {
            self.pow_opt(r, n)
        } else {
            self.intern(ReOp::Pow(r, n))
        }
    }

    fn pow_opt(&mut self, r: Regex, n: u32) -> Regex {
        if n == 0 {
            self.epsilon()
        } else if n == 1 {
            r.clone()
        } else {
            self.intern(ReOp::Pow(r, n))
        }
    }

    /// Constructs a regular expression denoting the union of languages `T^n, T^(n+1), ..., T^m`
    pub fn loop_(&mut self, r: Regex, l: u32, u: u32) -> Regex {
        if self.optimize {
            self.loop_opt(r, l, u)
        } else {
            self.intern(ReOp::Loop(r, l, u))
        }
    }

    fn loop_opt(&mut self, r: Regex, l: u32, u: u32) -> Regex {
        if l > u {
            self.none()
        } else if u == 0 {
            self.epsilon()
        } else if l == u {
            self.pow(r, u)
        } else {
            self.intern(ReOp::Loop(r, l, u))
        }
    }

    /// Unrolls a loop of the given regular expression.
    /// The loop is unrolled as a concatenation of the given regular expression repeated `l` times followed by the optional regular expression repeated `u-l` times.
    pub fn unroll_loop(&mut self, r: Regex, l: u32, u: u32) -> Regex {
        let mut concats: SmallVec<[Rc<ReNode>; 2]> = SmallVec::with_capacity(u as usize);
        let opt = self.opt(r.clone());
        for i in 0..u {
            if i < l {
                concats.push(r.clone());
            } else {
                concats.push(opt.clone());
            }
        }
        self.concat(concats)
    }

    /// Unrolls a power of the given regular expression.
    /// The power is unrolled as a concatenation of the given regular expression repeated `n` times.
    pub fn unroll_pow(&mut self, r: Regex, n: u32) -> Regex {
        let mut concats = SmallVec::with_capacity(n as usize);
        for _ in 0..n {
            concats.push(r.clone());
        }
        self.concat(concats)
    }

    /* Aux methods */

    /// Constructs a regular expression denoting the reverse of the given regular expression.
    /// For all words $w$ in the language of the given regular expression, the reverse language contains the reverse word $w^R$, where $w^R$ is the reverse of $w$.
    ///
    /// # Arguments
    /// * `r` - The regular expression to reverse.
    pub fn reversed(&mut self, r: &Regex) -> Regex {
        match r.op() {
            ReOp::Literal(word) => self.to_re(word.reversed()),
            ReOp::None | ReOp::All | ReOp::Any | ReOp::Range(_) => r.clone(),
            ReOp::Concat(rs) => {
                let rs = rs.iter().rev().map(|r| self.reversed(r)).collect();
                self.concat(rs)
            }
            ReOp::Union(rs) => {
                let rs = rs.iter().map(|r| self.reversed(r)).collect();
                self.union(rs)
            }
            ReOp::Inter(rs) => {
                let rs = rs.iter().map(|r| self.reversed(r)).collect();
                self.inter(rs)
            }
            ReOp::Star(r) => {
                let rev = self.reversed(r);
                self.star(rev)
            }
            ReOp::Plus(r) => {
                let rev = self.reversed(r);
                self.plus(rev)
            }
            ReOp::Opt(r) => {
                let rev = self.reversed(r);
                self.opt(rev)
            }
            ReOp::Comp(r) => {
                let rev = self.reversed(r);
                self.comp(rev)
            }
            ReOp::Diff(r, r1) => {
                let rev = self.reversed(r);
                let rev1 = self.reversed(r1);
                self.diff(rev, rev1)
            }
            ReOp::Pow(r, e) => {
                let rev = self.reversed(r);
                self.pow(rev, *e)
            }
            ReOp::Loop(r, l, u) => {
                let rev = self.reversed(r);
                self.loop_(rev, *l, *u)
            }
        }
    }
}

/// Ensures that identical regex patterns are reused and provides construction methods for each regex op(), optimizing memory usage and construction efficiency.
#[derive(Debug)]
struct Registry {
    /// Stores the unique instances of `Regex`.
    /// The key is the regex pattern itself, and the value is a shared reference-counted handle to the pattern.
    registry: HashMap<ReOp, Regex>,
    /// The id to assign to the next regex.
    next_id: usize,
}

impl Registry {
    /// Creates a new `Interner` with an empty internal registry.
    /// It starts without any regex patterns and will only populate its internal map when `intern` is called.
    fn new() -> Self {
        Registry {
            registry: HashMap::new(),
            next_id: 0,
        }
    }

    /// Interns a regex pattern, ensuring each unique regex is stored and reused.
    /// This method serves as the primary access point for adding new regex patterns to the builder.
    ///
    /// # Arguments
    /// * `regex` - The `Regex` pattern to intern.
    ///
    /// # Returns
    /// A [RegexRef] pointing to the stored or newly created regex instance.
    fn intern(&mut self, op: ReOp) -> Regex {
        if let Some(existing) = self.registry.get(&op) {
            existing.clone()
        } else {
            let regex = ReNode::new(self.next_id, op.clone());
            self.next_id += 1;
            let re = Rc::new(regex.clone());
            self.registry.insert(op, re.clone());
            re
        }
    }
}

#[cfg(test)]
mod test {
    use test::build::ReBuilder;

    use super::*;

    #[test]
    fn intern_word() {
        let mut builder = ReBuilder::default();
        let w1 = builder.to_re("abc".into());
        let w2 = builder.to_re("abc".into());
        assert!(Rc::ptr_eq(&w1, &w2));
    }

    #[test]
    fn intern_epsilon() {
        let builder = ReBuilder::default();
        let e1 = builder.epsilon();
        let e2 = builder.epsilon();
        assert!(Rc::ptr_eq(&e1, &e2));
    }

    #[test]
    fn intern_none() {
        let builder = ReBuilder::default();
        let n1 = builder.none();
        let n2 = builder.none();
        assert!(Rc::ptr_eq(&n1, &n2));
    }

    #[test]
    fn intern_all() {
        let builder = ReBuilder::default();
        let a1 = builder.all();
        let a2 = builder.all();
        assert!(Rc::ptr_eq(&a1, &a2));
    }

    #[test]
    fn intern_all_char() {
        let builder = ReBuilder::default();
        let ac1 = builder.any_char();
        let ac2 = builder.any_char();
        assert!(Rc::ptr_eq(&ac1, &ac2));
    }

    #[test]
    fn intern_range() {
        let mut builder = ReBuilder::default();
        let r1 = builder.range_from_to('a', 'z');
        let r2 = builder.range_from_to('a', 'z');
        assert!(Rc::ptr_eq(&r1, &r2));
    }

    #[test]
    fn intern_range_full() {
        let mut builder = ReBuilder::default();
        let r1 = builder.range_from_to('\0', '\u{10FFFF}');
        let r2 = builder.any_char();
        assert!(Rc::ptr_eq(&r1, &r2));
    }

    #[test]
    fn intern_concat() {
        let mut builder = ReBuilder::default();
        let w1 = builder.to_re("abc".into());
        let w2 = builder.to_re("def".into());
        let c1 = builder.concat(smallvec![w1.clone(), w2.clone()]);
        let c2 = builder.concat(smallvec![w1.clone(), w2.clone()]);
        assert!(Rc::ptr_eq(&c1, &c2));
    }

    #[test]
    fn intern_union() {
        let mut builder = ReBuilder::default();
        let w1 = builder.to_re("abc".into());
        let w2 = builder.to_re("def".into());
        let u1 = builder.union(smallvec![w1.clone(), w2.clone()]);
        let u2 = builder.union(smallvec![w1.clone(), w2.clone()]);
        assert!(Rc::ptr_eq(&u1, &u2));
    }

    #[test]
    fn intern_inter() {
        let mut builder = ReBuilder::default();
        let w1 = builder.to_re("abc".into());
        let w2 = builder.to_re("def".into());
        let args = smallvec![w1.clone(), w2.clone()];
        let i1 = builder.inter(args.clone());
        let i2 = builder.inter(args.clone());
        assert!(Rc::ptr_eq(&i1, &i2));
    }

    #[test]
    fn intern_star() {
        let mut builder = ReBuilder::default();
        let w1 = builder.to_re("abc".into());
        let s1 = builder.star(w1.clone());
        let s2 = builder.star(w1.clone());
        assert!(Rc::ptr_eq(&s1, &s2));
    }

    #[test]
    fn intern_plus() {
        let mut builder = ReBuilder::default();
        let w1 = builder.to_re("abc".into());
        let p1 = builder.plus(w1.clone());
        let p2 = builder.plus(w1.clone());
        assert!(Rc::ptr_eq(&p1, &p2));
    }

    #[test]
    fn intern_opt() {
        let mut builder = ReBuilder::default();
        let w1 = builder.to_re("abc".into());
        let o1 = builder.opt(w1.clone());
        let o2 = builder.opt(w1.clone());
        assert!(Rc::ptr_eq(&o1, &o2));
    }

    #[test]
    fn intern_comp() {
        let mut builder = ReBuilder::default();
        let w1 = builder.to_re("abc".into());
        let c1 = builder.comp(w1.clone());
        let c2 = builder.comp(w1.clone());
        assert!(Rc::ptr_eq(&c1, &c2));
    }

    #[test]
    fn builder_equal() {
        let mut builder = ReBuilder::default();
        let w1 = builder.to_re("abc".into());
        let w2 = builder.to_re("abc".into());
        assert_eq!(w1, w2);
        assert!(Rc::ptr_eq(&w1, &w2));

        let mut builder2 = ReBuilder::default();
        let w3 = builder2.to_re("abc".into());

        assert_eq!(w1, w3);
        assert!(!Rc::ptr_eq(&w1, &w3));
    }

    #[test]
    fn universal_concat() {
        let mut builder = ReBuilder::default();
        let a = builder.all();
        let c = builder.concat(smallvec![a.clone(), a.clone()]);
        assert_eq!(c.universal(), Some(true));
    }

    #[test]
    fn non_universal_concat() {
        let mut builder = ReBuilder::default();
        let a = builder.all();
        let b = builder.any_char();
        let c = builder.concat(smallvec![a.clone(), b.clone()]);
        assert_eq!(c.universal(), Some(false));
    }

    #[test]
    fn universal_union() {
        let mut builder = ReBuilder::default();
        let all = builder.all();
        let none = builder.none();
        let c = builder.union(smallvec![all, none]);
        assert_eq!(c.universal(), Some(true));
    }

    #[test]
    fn non_universal_union() {
        let mut builder = ReBuilder::default();
        let b = builder.any_char();
        let c = builder.concat(smallvec![b.clone(), b.clone()]);
        assert_eq!(c.universal(), Some(false));
    }

    #[test]
    fn test_reversed_constant() {
        let mut rb = ReBuilder::default();
        let regex = rb.to_re("abc".into());
        let reversed = rb.reversed(&regex);
        assert_eq!(reversed.to_string(), rb.to_re("cba".into()).to_string());
    }

    #[test]
    fn test_reversed_concat() {
        let mut rb = ReBuilder::default();
        let regex1 = rb.to_re("abc".into());
        let regex2 = rb.to_re("def".into());
        let concat = rb.concat(smallvec![regex1.clone(), regex2.clone()]);
        let reversed = rb.reversed(&concat);
        let args = smallvec![rb.reversed(&regex2), rb.reversed(&regex1)];
        let expected = rb.concat(args);
        assert_eq!(reversed.to_string(), expected.to_string());
    }

    #[test]
    fn test_reversed_concat_2() {
        let mut rb = ReBuilder::default();
        let a = rb.to_re("a".into());
        let astar = rb.star(a);

        let b = rb.to_re("b".into());
        let bstar = rb.star(b);

        // a*b*
        let concat = rb.concat(smallvec![astar.clone(), bstar.clone()]);
        // Should be b*a*
        let reversed = rb.reversed(&concat);

        // b*a*
        let args = smallvec![bstar, astar];
        let expected = rb.concat(args);
        assert_eq!(
            reversed, expected,
            "Expected: {}, Got: {}",
            expected, reversed
        );
    }

    #[test]
    fn test_reversed_union() {
        let mut rb = ReBuilder::default();
        let regex1 = rb.to_re("abc".into());
        let regex2 = rb.to_re("xyz".into());
        let union = rb.union(smallvec![regex1.clone(), regex2.clone()]);
        let reversed = rb.reversed(&union);
        let args = smallvec![rb.reversed(&regex1), rb.reversed(&regex2)];
        let expected = rb.union(args);
        assert_eq!(reversed.to_string(), expected.to_string());
    }

    #[test]
    fn test_reversed_star() {
        let mut rb = ReBuilder::default();
        let regex = rb.to_re("abc".into());
        let star = rb.star(regex.clone());
        let reversed = rb.reversed(&star);
        let revd = rb.reversed(&regex);
        let expected = rb.star(revd);
        assert_eq!(reversed.to_string(), expected.to_string());
    }

    #[test]
    fn test_reversed_plus() {
        let mut rb = ReBuilder::default();
        let regex = rb.to_re("abc".into());
        let plus = rb.plus(regex.clone());
        let reversed = rb.reversed(&plus);

        let rev = rb.reversed(&regex);
        let expected = rb.plus(rev);
        assert_eq!(reversed.to_string(), expected.to_string());
    }

    #[test]
    fn test_reversed_opt() {
        let mut rb = ReBuilder::default();
        let regex = rb.to_re("abc".into());
        let opt = rb.opt(regex.clone());
        let reversed = rb.reversed(&opt);

        let rev = rb.reversed(&regex);
        let expected = rb.opt(rev);
        assert_eq!(reversed.to_string(), expected.to_string());
    }

    #[test]
    fn test_reversed_inter() {
        let mut rb = ReBuilder::default();
        let regex1 = rb.to_re("abc".into());
        let regex2 = rb.to_re("xyz".into());
        let inter = rb.inter(smallvec![regex1.clone(), regex2.clone()]);
        let reversed = rb.reversed(&inter);

        let args = smallvec![rb.reversed(&regex1), rb.reversed(&regex2)];
        let expected = rb.inter(args);
        assert_eq!(reversed.to_string(), expected.to_string());
    }

    #[test]
    fn test_reversed_diff() {
        let mut rb = ReBuilder::default();
        let regex1 = rb.to_re("abc".into());
        let regex2 = rb.to_re("xyz".into());
        let diff = rb.diff(regex1.clone(), regex2.clone());
        let reversed = rb.reversed(&diff);

        let l = rb.reversed(&regex1);
        let r = rb.reversed(&regex2);
        let expected = rb.diff(l, r);
        assert_eq!(reversed.to_string(), expected.to_string());
    }

    #[test]
    fn test_reversed_comp() {
        let mut rb = ReBuilder::default();
        let regex = rb.to_re("abc".into());
        let comp = rb.comp(regex.clone());
        let reversed = rb.reversed(&comp);

        let rev = rb.reversed(&regex);
        let expected = rb.comp(rev);
        assert_eq!(reversed.to_string(), expected.to_string());
    }

    #[test]
    fn test_reversed_pow() {
        let mut rb = ReBuilder::default();
        let regex = rb.to_re("abc".into());
        let pow = rb.pow(regex.clone(), 3);
        let reversed = rb.reversed(&pow);

        let rev = rb.reversed(&regex);
        let expected = rb.pow(rev, 3);
        assert_eq!(reversed.to_string(), expected.to_string());
    }

    #[test]
    fn test_reversed_loop() {
        let mut rb = ReBuilder::default();
        let regex = rb.to_re("abc".into());
        let looped = rb.loop_(regex.clone(), 2, 5);
        let reversed = rb.reversed(&looped);

        let rev = rb.reversed(&regex);
        let expected = rb.loop_(rev, 2, 5);
        assert_eq!(reversed.to_string(), expected.to_string());
    }

    #[test]
    fn test_unroll_loop_zero() {
        let mut rb = ReBuilder::default();
        let regex = rb.to_re("abc".into());
        let unrolled = rb.unroll_loop(regex.clone(), 0, 0);
        assert_eq!(unrolled.to_string(), rb.epsilon().to_string());
    }

    #[test]
    fn test_unroll_loop_single() {
        let mut rb = ReBuilder::default();
        let regex = rb.to_re("abc".into());
        let unrolled = rb.unroll_loop(regex.clone(), 1, 1);
        assert_eq!(unrolled.to_string(), regex.to_string());
    }

    #[test]
    fn test_unroll_loop_multiple() {
        let mut rb = ReBuilder::default();
        let regex = rb.to_re("abc".into());
        let unrolled = rb.unroll_loop(regex.clone(), 2, 4);
        let opt = rb.opt(regex.clone());

        let expected = rb.concat(smallvec![
            regex.clone(),
            regex.clone(),
            opt.clone(),
            opt.clone()
        ]);
        assert_eq!(unrolled.to_string(), expected.to_string());
    }

    #[test]
    fn test_unroll_loop_full() {
        let mut rb = ReBuilder::default();
        let regex = rb.to_re("abc".into());
        let unrolled = rb.unroll_loop(regex.clone(), 3, 3);
        let expected = rb.concat(smallvec![regex.clone(), regex.clone(), regex.clone()]);
        assert_eq!(unrolled.to_string(), expected.to_string());
    }

    #[test]
    fn test_unroll_loop_opt() {
        let mut rb = ReBuilder::default();
        let regex = rb.to_re("abc".into());
        let unrolled = rb.unroll_loop(regex.clone(), 0, 2);
        let opt = rb.opt(regex.clone());
        let expected = rb.concat(smallvec![opt.clone(), opt.clone()]);
        assert_eq!(unrolled.to_string(), expected.to_string());
    }

    #[test]
    fn test_unroll_pow_zero() {
        let mut builder = ReBuilder::default();
        let r = builder.to_re("a".into());
        let result = builder.unroll_pow(r, 0);
        assert_eq!(builder.epsilon(), result);
    }

    #[test]
    fn test_unroll_pow_one() {
        let mut builder = ReBuilder::default();
        let r = builder.to_re("a".into());
        let result = builder.unroll_pow(r.clone(), 1);
        assert_eq!(r, result);
    }

    #[test]
    fn test_unroll_pow_multiple() {
        let mut builder = ReBuilder::default();
        let r = builder.to_re("a".into());
        let result = builder.unroll_pow(r.clone(), 3);
        let expected = builder.concat(smallvec![r.clone(), r.clone(), r.clone()]);
        assert_eq!(expected, result);
    }
}
