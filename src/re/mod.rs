//! Core module for SMT-LIB regular expressions.
//!
//! This module defines the core data structures and algorithms for working with regular expressions
//! in the SMT-LIB theory of strings. Regular expressions are represented as abstract syntax trees (ASTs),
//! where each node represents a regular operation or base case.
//!
//!  The module provides the following types:
//!
//! - [`Regex`] — a reference-counted pointer to a regular expression node ([`ReNode`]).
//! - [`ReNode`] — a node in a regular expression AST.
//! - [`ReOp`] — the enum describing the different operations in a regular expression (e.g., `re.*`, `re.union`, `re.diff`).
//!
//! Each [`ReNode`] contains a [`ReOp`], which may be either a base case (e.g., a literal or character range),
//! or a regular operation applied to one or more subexpressions.
//! In the latter case, subexpressions are stored as reference-counted pointer ([`Regex`]), allowing for structural sharing of common subtrees.
//! This means that if a regular expression contains the same subexpression multiple times, the corresponding [ReOp]s will all point to the same [ReNode] instance.
//!
//! For example, if the regex is
//!
//! ```text
//! re.union(re.*(str.to_re("a")), re.comp(str.to_re("a")))`
//! ```
//!
//! then the `re.*` and `re.comp` operations will point to the same exact same [ReNode] instance for the subexpression `str.to_re("a")`.
//!
//! Structural sharing enables:
//!
//! - Reduced memory usage
//! - Efficient equality and hashing (by node ID)
//! - Cheap cloning of regular expressions
//! - Caching of derived properties (e.g., nullable, alphabet, etc.)
//!
//! Because of this, regular expressions must be created using the [`ReBuilder`] and cannot be constructed directly.
//!
//! # Example
//!
//! ```
//! use smt_str::re::*;
//! use std::rc::Rc;
//!
//! // Create a regular expression builder
//! let mut builder = ReBuilder::default();
//!
//! // Construct the regular expression `a``
//! let re_a = builder.to_re("a".into());
//!
//! // Construct the regular expression `a*` and `comp(a)`
//! let re_star = builder.star(re_a.clone());
//! let re_comp = builder.comp(re_a.clone());
//!
//! // The operand of `re_star` and `re_comp` are not only structurally equal
//! assert_eq!(re_star[0], re_comp[0]);
//! // but also share the same node in memory
//! assert!(Rc::ptr_eq(&re_star[0], &re_comp[0]));
//! ```

mod build;
pub mod deriv;

use quickcheck::{Arbitrary, Gen};
use smallvec::{smallvec, SmallVec};

use itertools::Itertools;

use std::cell::RefCell;

use std::hash::Hash;
use std::ops::Index;
use std::{fmt::Display, rc::Rc};

pub use build::ReBuilder;

use crate::alphabet::{partition::AlphabetPartition, Alphabet, CharRange};
use crate::SmtString;

pub type ReId = usize;

type LazyProp<T> = RefCell<Option<T>>;

/// A shared pointer to a regular expression node (`ReNode`).
///
/// This alias is used throughout the crate to represent a node in a regular expression AST.
/// For details on construction and structure sharing see the module documentation.
pub type Regex = Rc<ReNode>;

/// A node in the abstract syntax tree (AST) of a regular expression.
///
/// Each `ReNode` stores a single operation ([`ReOp`]) and a set of lazily computed
/// properties, such as:
///
/// - Whether the expression is nullable (i.e., accepts the empty word)
/// - Whether it is universal (i.e., accepts all words)
/// - Whether it denotes the empty set or the empty word
/// - Whether it has a constant prefix, suffix, or word
/// - The first-character partition and alphabet
/// - Whether it contains a complement or difference operation
///
/// All these properties are cached and evaluated on demand to improve performance.
///
/// Every `ReNode` has a unique identifier assigned by [`ReBuilder`].
/// This is unique across all regular expressions created by the same builder, which enables
/// fast (i.e., **O(1)**) equality checking and hashing.
/// Instances of this type are also superficially order by their ID.
///
///**Note**: `ReNode` is not meant to be used directly.
/// Use the [`Regex`](crate::re::Regex) alias, which wraps it in a reference-counted pointer.
///
/// # Cheap Cloning
///
/// The subexpressions of a regular expression are stored as reference-counted pointers ([`Regex`]) in the [`ReOp`] variant that defines the operation.
/// All derived properties are lazily computed and cached using interior mutability.
/// As a result, cloning a regular expression is inexpensive: it only increments the reference count to shared subexpressions and cached data.
/// This makes it efficient to clone and pass around regular expressions, even when they are large.
///
/// # Caching and Interior Mutability
///
/// This structure uses interior mutability (`RefCell`) to cache the derived properties.
/// These caches do **not** affect the `Hash`, `Ord`, or `Eq` implementations, which
/// rely solely on the unique `id`.
///
/// Therefore, it is safe to use `ReNode` (via [`Regex`]) as a key in hash maps or sets,
/// even though Clippy may issue warnings about interior mutability.
/// The warning `clippy::mutable_key_type` can be safely suppressed for this type using `#[allow(clippy::mutable_key_type)]`
#[derive(Debug, Clone)]
pub struct ReNode {
    /// Unique identifier for the regular expression.
    id: ReId,
    /// The operation defining the regex structure.
    op: ReOp,

    /// Whether the regex is simple.
    simple: LazyProp<bool>,

    /// Whether the regular expression can accept the empty word (ε).
    nullable: LazyProp<bool>,
    /// Whether the regular expression accepts all possible strings.
    universal: LazyProp<Option<bool>>,
    /// Whether the regular expression is the empty set (∅), meaning it accepts no words.
    none: LazyProp<Option<bool>>,
    /// Whether the regular expression exclusively denotes the empty word (ε).
    epsi: LazyProp<Option<bool>>,
    /// The set of characters that can be first in a word accepted by the regular expression.
    /// Partitioned according to left quotients.
    first: LazyProp<Rc<AlphabetPartition>>,

    /// The alphabet of the regular expression.
    alphabet: LazyProp<Rc<Alphabet>>,
    /// The only word accepted by the regular expression, if it is a constant word.
    is_constant: LazyProp<Option<SmtString>>,

    /// The prefix of all words accepted by the regular expression.
    /// Not necessarily the longest prefix.
    prefix: LazyProp<Option<SmtString>>,

    /// The suffix of all words accepted by the regular expression.
    /// Not necessarily the longest suffix.
    suffix: LazyProp<Option<SmtString>>,

    /// Whether the regular expression contains a complement operation.
    /// Also true if the expression contains difference, which is equivalent to complement.
    contains_complement: LazyProp<bool>,
}

impl PartialEq for ReNode {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}
impl Eq for ReNode {}
impl Hash for ReNode {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state)
    }
}
impl PartialOrd for ReNode {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for ReNode {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.id.cmp(&other.id)
    }
}

impl Index<usize> for ReNode {
    type Output = Regex;

    /// Returns the subexpression at the given index.
    /// If the index is out of bounds, this function panics.
    /// This function always panics on regular expressions that have no subexpressions, i.e., literals, ranges, none, any, and all.
    fn index(&self, index: usize) -> &Self::Output {
        match self.op() {
            ReOp::Concat(v) | ReOp::Union(v) | ReOp::Inter(v) => &v[index],
            ReOp::Star(r)
            | ReOp::Plus(r)
            | ReOp::Opt(r)
            | ReOp::Comp(r)
            | ReOp::Pow(r, _)
            | ReOp::Loop(r, _, _)
                if index == 0 =>
            {
                r
            }
            ReOp::Diff(r, _) if index == 0 => r,
            ReOp::Diff(_, r) if index == 1 => r,
            _ => panic!("Index out of bounds"),
        }
    }
}

impl ReNode {
    /// Creates a new regular expression node with the given ID and operation.
    /// This function should only be used by the `ReBuilder`.
    fn new(id: ReId, op: ReOp) -> Self {
        Self {
            id,
            op,
            nullable: RefCell::new(None),
            universal: RefCell::new(None),
            simple: RefCell::new(None),
            none: RefCell::new(None),
            epsi: RefCell::new(None),
            first: RefCell::new(None),
            alphabet: RefCell::new(None),
            is_constant: RefCell::new(None),
            prefix: RefCell::new(None),
            suffix: RefCell::new(None),
            contains_complement: RefCell::new(None),
        }
    }

    /// Returns the unique identifier of the regular expression.
    /// The ID assigned by the `ReBuilder` and unique across all regular expressions created by the same builder.
    pub fn id(&self) -> ReId {
        self.id
    }

    /// Returns the operation defining the regular expression.
    ///
    /// # Example
    /// ```
    /// use smt_str::re::*;
    ///
    /// let mut builder = ReBuilder::default();
    /// let re = builder.to_re("a".into());
    /// assert_eq!(re.op(), &ReOp::Literal("a".into()));
    ///
    /// let re_star = builder.star(re.clone());
    /// assert_eq!(re_star.op(), &ReOp::Star(re));
    /// ```
    pub fn op(&self) -> &ReOp {
        &self.op
    }

    /// Returns whether the regular expression is nullable.
    /// A regular expression is nullable if it accepts the empty word.
    ///
    /// # Example
    /// ```
    /// use smt_str::re::*;
    /// let mut builder = ReBuilder::default();
    /// let re = builder.to_re("a".into());
    /// let re_star = builder.star(re.clone());
    /// assert!(!re.nullable());
    /// assert!(re_star.nullable());
    /// ```
    pub fn nullable(&self) -> bool {
        *self
            .nullable
            .borrow_mut()
            .get_or_insert_with(|| self.op.nullable())
    }

    /// Returns whether the regular expression is simple.
    /// A regular expression is simple if it does not contain complement, difference, or intersection.
    ///
    /// # Example
    /// ```
    /// use smt_str::re::*;
    /// use smallvec::smallvec;
    /// let mut builder = ReBuilder::default();
    /// let re_a = builder.to_re("a".into());
    /// let re_star = builder.star(re_a.clone());
    /// assert!(re_a.simple());
    /// assert!(re_star.simple());
    ///
    /// // Complement, difference, and intersection are not simple.
    ///
    /// let re_comp = builder.comp(re_a.clone());
    /// let re_diff = builder.diff(re_star.clone(), re_a.clone());
    /// let re_inter = builder.inter(smallvec![re_star.clone(), re_a.clone()]);
    /// assert!(!re_comp.simple());
    /// assert!(!re_diff.simple());
    /// assert!(!re_inter.simple());
    /// ```
    pub fn simple(&self) -> bool {
        *self
            .simple
            .borrow_mut()
            .get_or_insert_with(|| self.op.simple())
    }

    /// Returns whether the regular expression is universal.
    /// A regular expression is universal if it accepts every word.
    /// This is determined heuristically based on the structure of the regular expression.
    /// As such, this operation may return `None` if the universality cannot be determined.
    ///
    /// # Example
    /// ```
    /// use smt_str::re::*;
    /// use smallvec::smallvec;
    /// let mut builder = ReBuilder::default();
    ///
    /// assert_eq!(builder.all().universal(), Some(true));
    /// assert_eq!(builder.none().universal(), Some(false));
    ///
    /// // Universally can not always be determined.
    /// let re_a = builder.to_re("a".into());
    /// let diff = builder.diff(builder.all(), re_a.clone());
    /// let union = builder.union(smallvec![diff, re_a.clone()]);
    ///
    /// assert_eq!(union.universal(), None);
    /// ```
    pub fn universal(&self) -> Option<bool> {
        *self
            .universal
            .borrow_mut()
            .get_or_insert_with(|| self.op.universal())
    }

    /// Returns whether the regular expression is the empty set.
    /// A regular expression is the empty set if it does not accept any word.
    /// This operation may return `None` if the emptiness cannot be determined.
    ///
    /// # Example
    /// ```
    /// use smt_str::re::*;
    /// use smallvec::smallvec;
    /// let mut builder = ReBuilder::default();
    /// assert_eq!(builder.none().none(), Some(true));
    /// assert_eq!(builder.all().none(), Some(false));
    ///
    /// let re_a = builder.to_re("a".into());
    /// let re_comp = builder.comp(re_a.clone());
    /// let inter = builder.inter(smallvec![re_a.clone(), re_comp.clone()]);
    ///
    /// assert_eq!(re_a.none(), Some(false));
    /// assert_eq!(re_comp.none(), Some(false));
    /// assert_eq!(inter.none(), None);
    /// ```
    pub fn none(&self) -> Option<bool> {
        *self.none.borrow_mut().get_or_insert_with(|| self.op.none())
    }

    /// Returns whether the regular expression denotes the empty word.
    /// A regular expression denotes the empty word if it only accepts the empty word.
    /// This operation may return `None` if the property cannot be determined.
    ///
    /// # Example
    /// ```
    /// use smt_str::re::*;
    /// use smallvec::smallvec;
    /// let mut builder = ReBuilder::default();
    ///
    /// assert_eq!(builder.epsilon().epsilon(), Some(true));
    ///
    /// let re_a = builder.to_re("a".into());
    /// let re_b = builder.to_re("b".into());
    /// let re_a_star = builder.star(re_a.clone());
    /// let re_b_star = builder.opt(re_a.clone());
    /// // The intersection contains only the empty word but we can't determine it.
    /// let re_inter = builder.inter(smallvec![re_a_star.clone(), re_b_star.clone()]);
    /// assert_eq!(re_inter.epsilon(), None);
    /// ```
    pub fn epsilon(&self) -> Option<bool> {
        *self
            .epsi
            .borrow_mut()
            .get_or_insert_with(|| self.op.epsilon())
    }

    /// Returns the set of characters that can be first in a word accepted by the regular expression.
    /// The set of first characters is partitioned into disjoint subsets of the alphabet, such that for each two characters `a` and `b` in the same subset, the left quotient of the regular expression w.r.t. `a` is equal to the left quotient of the regular expression w.r.t. `b`.
    /// In other words, if  `a` and `b` are in the same subset, then for all words `u`, the regular expression accepts `a \cdot u` iff it accepts `b \cdot u`.
    pub fn first(&self) -> Rc<AlphabetPartition> {
        self.first
            .borrow_mut()
            .get_or_insert_with(|| self.op.first())
            .clone()
    }

    /// The alphabet of the regular expression.
    /// The alphabet is the set of characters that occur in the regular expression.
    /// This is not the alphabet of words that the regular expression can accept.
    /// For example, the alphabet of `[^a]` is `{'a'}`, but the alphabet of `[^a]` is every character except `'a'`.
    ///
    ///
    /// # Example
    /// ```
    /// use smt_str::re::*;
    /// use smt_str::alphabet::{Alphabet, CharRange};
    ///
    /// use smallvec::smallvec;
    /// let mut builder = ReBuilder::default();
    /// let re_a = builder.to_re("a".into()); // 'a'
    /// let re_b = builder.to_re("b".into()); // 'b'
    /// let re_a_comp = builder.star(re_a.clone()); // comp('a')
    /// let re_union = builder.union(smallvec![re_a_comp.clone(), re_b.clone()]); // comp('a') | 'b'
    ///
    /// assert_eq!(re_a.alphabet().as_ref(), &Alphabet::from(CharRange::singleton('a')));
    /// // even though `comp('a')` accepts words with any character from the SMT-LIB alphabet, the alphabet of the regex is still `{'a'}`.
    /// assert_eq!(re_a_comp.alphabet().as_ref(), &Alphabet::from(CharRange::singleton('a')));
    /// assert_eq!(
    ///     re_union.alphabet().as_ref(),
    ///     &Alphabet::from_iter(vec![CharRange::singleton('a'), CharRange::singleton('b')])
    /// );
    /// ```
    pub fn alphabet(&self) -> Rc<Alphabet> {
        self.alphabet
            .borrow_mut()
            .get_or_insert_with(|| self.op.alphabet())
            .clone()
    }

    /// Returns Some(word) if the regular expression accepts only the given constant word, None otherwise or if it cannot be determined.
    ///
    /// # Example
    /// ```
    /// use smt_str::re::*;
    /// use smallvec::smallvec;
    /// let mut builder = ReBuilder::default();
    ///
    /// let re_foo = builder.to_re("foo".into());
    /// assert_eq!(re_foo.is_constant(), Some("foo".into()));
    ///
    /// // Not a constant word
    /// let re_opt = builder.opt(re_foo.clone());
    /// assert_eq!(re_opt.is_constant(), None);
    ///
    /// // Cannot be determined
    /// let re_bar = builder.to_re("bar".into());
    /// let re_foo_or_bar = builder.union(smallvec![re_foo.clone(), re_bar.clone()]);
    /// let inter = builder.inter(smallvec![re_foo.clone(), re_foo_or_bar.clone()]); // foo & (foo | bar) <--> foo
    /// assert_eq!(inter.is_constant(), None);
    ///```
    pub fn is_constant(&self) -> Option<SmtString> {
        self.is_constant
            .borrow_mut()
            .get_or_insert_with(|| self.op().is_constant())
            .clone()
    }

    /// Returns the prefix of all words accepted by the regular expression.
    /// Makes a best effort to obtain the longest prefix, but does not guarantee it.
    /// Is `None` if the prefix cannot be determined.
    ///
    /// # Example
    /// ```
    /// use smt_str::re::*;
    /// use smallvec::smallvec;
    /// let mut builder = ReBuilder::default();
    /// let re_foo = builder.to_re("foo".into());
    ///
    /// assert_eq!(re_foo.prefix(), Some("foo".into()));
    ///
    /// let re_bar = builder.to_re("bar".into());
    /// let re_foobar = builder.concat(smallvec![re_foo.clone(), re_bar.clone()]);
    ///
    /// let union = builder.union(smallvec![re_foo.clone(), re_foobar.clone()]);
    /// assert_eq!(union.prefix(), Some("foo".into()));
    /// ```
    pub fn prefix(&self) -> Option<SmtString> {
        self.prefix
            .borrow_mut()
            .get_or_insert_with(|| self.op().prefix())
            .clone()
    }

    /// Returns the suffix of all words accepted by the regular expression.
    /// Makes a best effort to obtain the longest suffix, but does not guarantee it.
    /// Is `None` if the suffix cannot be determined, which is the case for some extended regexes.
    ///
    /// # Example
    /// ```
    /// use smt_str::re::*;
    /// use smallvec::smallvec;
    /// let mut builder = ReBuilder::default();
    /// let re_bar = builder.to_re("bar".into());
    ///
    /// assert_eq!(re_bar.suffix(), Some("bar".into()));
    ///
    /// let re_foo = builder.to_re("foo".into());
    /// let re_foobar = builder.concat(smallvec![re_foo.clone(), re_bar.clone()]);
    ///
    /// let union = builder.union(smallvec![re_bar.clone(), re_foobar.clone()]);
    /// assert_eq!(union.suffix(), Some("bar".into()));
    /// ```
    pub fn suffix(&self) -> Option<SmtString> {
        self.suffix
            .borrow_mut()
            .get_or_insert_with(|| self.op().suffix())
            .clone()
    }

    /// Returns whether the regular expression contains a complement operation.
    ///
    /// # Example
    /// ```
    /// use smt_str::re::*;
    /// use smallvec::smallvec;
    /// let mut builder = ReBuilder::default();
    /// let re_a = builder.to_re("a".into());
    /// let re_b = builder.to_re("b".into());
    /// let re_a_comp = builder.comp(re_a.clone());
    ///
    /// assert!(!re_a.contains_complement());
    /// assert!(re_a_comp.contains_complement());
    ///
    /// // Difference is equivalent to complement
    /// let diff = builder.diff(re_a.clone(), re_b.clone());
    /// assert!(diff.contains_complement());
    /// ```
    pub fn contains_complement(&self) -> bool {
        *self
            .contains_complement
            .borrow_mut()
            .get_or_insert_with(|| self.op().contains_complement())
    }

    /// Return whether the regular expression accepts a given word.
    ///
    /// # Example
    /// ```
    /// use smt_str::re::*;
    /// let mut builder = ReBuilder::default();
    /// let re_a = builder.to_re("a".into());
    /// let re_star = builder.star(re_a.clone());
    /// assert!(re_star.accepts(&"a".into()));
    /// assert!(re_star.accepts(&"aaaa".into()));
    /// assert!(re_star.accepts(&"".into()));
    /// ```
    pub fn accepts(&self, w: &SmtString) -> bool {
        let mut builder_tmp = ReBuilder::default();
        let mnged = builder_tmp.regex(&Rc::new(self.clone()));
        deriv::deriv_word(&mnged, w.clone(), &mut builder_tmp).nullable()
    }

    /// Returns an iterator over the subexpressions of the regular expression.
    ///
    /// The subexpressions of a regular expression are the direct children of the operation defining the regex.
    /// If the regex is a base case (literal, range, none, any, all), it has no subexpressions.
    ///
    /// # Example
    /// ```
    /// use smt_str::re::*;
    /// use smallvec::smallvec;
    ///
    /// let mut builder = ReBuilder::default();
    /// let re_a = builder.to_re("a".into());
    /// let re_star = builder.star(re_a.clone());
    /// let re_b = builder.to_re("b".into());
    /// let re_union = builder.union(smallvec![re_star.clone(), re_b.clone()]);
    ///
    /// let mut subexprs = re_union.subexpr();
    /// assert_eq!(subexprs.next(), Some(&re_star));
    /// assert_eq!(subexprs.next(), Some(&re_b));
    /// assert_eq!(subexprs.next(), None);
    /// ```
    pub fn subexpr(&self) -> SubExpressions {
        SubExpressions {
            node: self,
            index: 0,
        }
    }
}

impl Display for ReNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.op)
    }
}

/// An iterator over the subexpressions of a regular expression.
/// Only iterates over the immediate subexpressions, not recursively.
///
/// The subexpressions of a regular expression are the direct children of the operation defining the regex.
/// If the regex is a base case (literal, range, none, any, all), the iterator will return `None`.
pub struct SubExpressions<'a> {
    node: &'a ReNode,
    index: usize,
}

impl<'a> Iterator for SubExpressions<'a> {
    type Item = &'a Regex;

    fn next(&mut self) -> Option<Self::Item> {
        match self.node.op() {
            ReOp::Literal(_) | ReOp::Range(_) | ReOp::None | ReOp::Any | ReOp::All => None,
            ReOp::Concat(rs) | ReOp::Union(rs) | ReOp::Inter(rs) if self.index < rs.len() => {
                let r = &rs[self.index];
                self.index += 1;
                Some(r)
            }
            ReOp::Star(r)
            | ReOp::Plus(r)
            | ReOp::Opt(r)
            | ReOp::Comp(r)
            | ReOp::Pow(r, _)
            | ReOp::Loop(r, _, _)
                if self.index == 0 =>
            {
                Some(r)
            }
            ReOp::Diff(r1, r2) => {
                if self.index == 0 {
                    self.index += 1;
                    Some(r1)
                } else if self.index == 1 {
                    self.index += 1;
                    Some(r2)
                } else {
                    None
                }
            }
            _ => None,
        }
    }
}

/// The core operations that define a regular expression.
///
/// This enum represents the different building blocks of a regular expression,
/// including **base cases** (such as literals and predefined sets) and **derived**
/// operations (such as concatenation, union, and intersection).
///
/// ## Structure Sharing
/// Many variants use **reference-counted pointers (`Rc<Regex>`)** to enable
/// **structural sharing** and **deduplication** of sub-expressions. This ensures
/// that identical sub-expressions are only computed once, improving efficiency.
///
/// ## Base Cases
/// These represent fundamental building blocks of regular expressions:
/// - `Literal`: A constant word.
/// - `Range`: A set of characters in a specific range.
/// - `None`: The empty set, which matches no words.
/// - `Any`: The set of all possible one-character strings.
/// - `All`: The set of all words, including the empty word.
///
/// ## Derived Operations
/// These are higher-level operations built from the base cases:
/// - **Concatenation (`Concat`)**: Sequences two or more regular expressions.
/// - **Union (`Union`)**: Represents the union (alternation) of two or more regexes.
/// - **Intersection (`Inter`)**: Matches only strings present in both regexes.
/// - **Star (`Star`)**: The Kleene star (`r*`), allowing zero or more repetitions.
/// - **Plus (`Plus`)**: The positive closure (`r+`), requiring at least one repetition.
/// - **Optional (`Opt`)**: Matches either a regex or the empty word (`r?`).
/// - **Difference (`Diff`)**: Matches words in one regex but not in another.
/// - **Complement (`Comp`)**: Matches everything **except** what a regex matches.
/// - **Power (`Pow`)**: Repeats a regex exactly `n` times (`r^n`).
/// - **Loop (`Loop`)**: Matches repetitions of a regex within a given range `[l, u]`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ReOp {
    // Base cases
    /// A constant (possibly empty) word.
    Literal(SmtString),
    /// A range of characters.
    Range(CharRange),
    /// The empty set (∅), which matches no strings.
    None,
    /// The set of all one-character strings. Independent of the alphabet.
    Any,
    /// The set of all possible words, including the empty word.
    All,
    // Derived cases
    /// The concatenation of two or more regular expressions.
    /// Matches words formed by appending matches of each sub-expression in sequence.
    Concat(SmallVec<[Regex; 2]>),
    /// The union (alternation) of two or more regular expressions.
    /// Matches words accepted by at least one of the sub-expressions.
    Union(SmallVec<[Regex; 2]>),
    /// The intersection of two or more regular expressions.
    /// Matches only words present in all sub-expressions.
    Inter(SmallVec<[Regex; 2]>),
    /// The Kleene star (`r*`). Matches zero or more repetitions of the regex.
    Star(Regex),
    /// The positive closure (`r+`). Matches one or more repetitions of the regex.
    Plus(Regex),
    /// The optional operator (`r?`). Matches either the regex or the empty word (ε).
    Opt(Regex),
    /// The difference (`r1 - r2`). Matches words in `r1` that are **not** in `r2`.
    Diff(Regex, Regex),
    /// The complement (`¬r`). Matches all words **except** those in the regex.
    Comp(Regex),
    /// The power operation (`r^n`). Matches exactly `n` repetitions of the regex.
    Pow(Regex, u32),
    /// The bounded repetition (`r[l, u]`). Matches between `l` and `u` repetitions.
    Loop(Regex, u32, u32),
}

impl ReOp {
    /// Computes whether the regular expression is nullable.
    fn nullable(&self) -> bool {
        match self {
            ReOp::Literal(word) => word.is_empty(),
            ReOp::Range(_) | ReOp::None | ReOp::Any => false,
            ReOp::All | ReOp::Star(_) | ReOp::Opt(_) => true,
            ReOp::Inter(rs) | ReOp::Concat(rs) => rs.iter().all(|r| r.nullable()),
            ReOp::Union(rs) => rs.iter().any(|r| r.nullable()),
            ReOp::Plus(r) => r.nullable(),
            ReOp::Diff(r1, r2) => r1.nullable() && !r2.nullable(),
            ReOp::Comp(r) => !r.nullable(),
            ReOp::Pow(r, e) => *e == 0 || r.nullable(),
            ReOp::Loop(r, l, u) => {
                if l <= u {
                    *l == 0 || r.nullable()
                } else {
                    // Invalid loop, corresponding to the empty set.
                    false
                }
            }
        }
    }

    /// Compute whether the regular expression is universal, i.e., accepts all words.
    fn universal(&self) -> Option<bool> {
        match self {
            ReOp::Literal(_) | ReOp::None | ReOp::Range(_) | ReOp::Any => Some(false),
            ReOp::All => Some(true),
            ReOp::Concat(rs) | ReOp::Inter(rs) => {
                if rs.is_empty() {
                    return match self {
                        ReOp::Concat(_) => Some(false), // Concatenation of nothing is epsilon.
                        ReOp::Inter(_) => Some(true),   // Intersection of nothing  is universal.
                        _ => unreachable!(),
                    };
                }
                // All subexpressions must be universal.
                for r in rs {
                    if !r.universal()? {
                        return Some(false);
                    }
                }
                Some(true)
            }
            ReOp::Union(rs) => {
                if rs.is_empty() {
                    return Some(false);
                }
                // Any subexpression that is universal makes the union universal.
                if rs.iter().any(|r| r.universal().unwrap_or(false)) {
                    Some(true)
                } else {
                    // The union of the subexpressions may form the universal set, we can't determine.
                    None
                }
            }
            ReOp::Star(r) | ReOp::Plus(r) | ReOp::Opt(r) => r.universal(),
            ReOp::Diff(r1, r2) => match (r1.universal(), r2.none()) {
                (Some(false), _) => Some(false),
                (Some(true), Some(true)) => Some(true),
                (_, Some(false)) => Some(false),
                _ => None,
            },
            ReOp::Comp(r) => r.none(),
            ReOp::Pow(_, 0) => Some(false),
            ReOp::Pow(r, _) => r.universal(),
            ReOp::Loop(_, l, u) if l > u || *u == 0 => Some(false),
            ReOp::Loop(r, _, _) => r.universal(),
        }
    }

    /// Compute whether the regular expression is empty, i.e., accepts no words.
    fn none(&self) -> Option<bool> {
        match self {
            ReOp::None => Some(true), // The empty set is trivially empty.
            ReOp::Literal(_) | ReOp::Any | ReOp::All => Some(false),
            ReOp::Range(r) => Some(r.is_empty()),
            ReOp::Concat(rs) => rs.iter().try_fold(false, |a, r| r.none().map(|b| a || b)),
            ReOp::Inter(rs) => {
                if rs.iter().any(|r| r.none().unwrap_or(false)) {
                    Some(true)
                } else {
                    // The intersection of the subexpressions may be empty, we can't determine.
                    None
                }
            }
            ReOp::Union(rs) => rs.iter().all(|r| r.none().unwrap_or(false)).into(),
            ReOp::Star(_) | ReOp::Opt(_) => Some(false), // These always match at least ε.
            ReOp::Plus(r) => r.none(),
            ReOp::Diff(r1, r2) => match (r1.none(), r2.universal()) {
                (Some(true), _) => Some(true), // If `r1` is empty, the difference is empty.
                (_, Some(true)) => Some(true), // Subtracting a universal set leaves an empty set.
                _ => None,
            },
            ReOp::Comp(r) => r.universal(), // Complement of a universal regex is an empty set.
            ReOp::Pow(_, 0) => Some(false), // `r^0 = ε`, which isn't empty.
            ReOp::Pow(r, _) => r.none(),    // `r^n` is empty if `r` itself is empty.
            ReOp::Loop(_, l, u) if l > u || *u == 0 => Some(true), // Invalid loop is empty by definition.
            ReOp::Loop(r, _, _) => r.none(), // If `l > 0`, behavior follows `r`.
        }
    }

    /// Compute whether the regular expression accepts the empty word and only the empty word.
    fn epsilon(&self) -> Option<bool> {
        match self {
            ReOp::None | ReOp::Any | ReOp::All | ReOp::Range(_) => Some(false),
            ReOp::Literal(w) => Some(w.is_empty()),
            ReOp::Union(rs) | ReOp::Concat(rs) => {
                if rs.is_empty() {
                    return match self {
                        ReOp::Union(_) => Some(false),
                        ReOp::Concat(_) => Some(true),
                        _ => unreachable!(),
                    };
                }
                // All subexpressions must accept ε and only ε.
                for r in rs {
                    if !r.epsilon()? {
                        return Some(false);
                    }
                }
                Some(true)
            }
            ReOp::Inter(rs) => {
                if rs.is_empty() {
                    Some(false)
                } else if rs.iter().all(|r| r.epsilon().unwrap_or(false)) {
                    // If all subexpressions accept ε and only ε, the intersection does too.
                    Some(true)
                } else {
                    // We can't determine if the intersection accepts only ε.
                    None
                }
            }
            ReOp::Plus(r) | ReOp::Star(r) | ReOp::Opt(r) => r.epsilon(),
            ReOp::Diff(_, _) | ReOp::Comp(_) => None, // can't determine
            ReOp::Pow(_, 0) => Some(true),            // `r^0 = ε`
            ReOp::Pow(r, _) => r.epsilon(),           // `r^n` is ε iff `r` itself is ε.
            ReOp::Loop(_, l, u) if l > u => Some(false), // Invalid loop is empty by definition.
            ReOp::Loop(_, 0, 0) => Some(true),        // `r^0` is ε
            ReOp::Loop(r, _, _) => r.epsilon(),       // `r^l..u` is ε iff `r` itself is ε.
        }
    }

    /// Compute the set of characters that can be first in a word accepted by the regular expression.
    /// The character are partitioned by their left quotient.
    fn first(&self) -> Rc<AlphabetPartition> {
        match self {
            ReOp::Literal(w) => {
                if let Some(c) = w.first() {
                    Rc::new(AlphabetPartition::singleton(CharRange::singleton(c)))
                } else {
                    Rc::new(AlphabetPartition::empty())
                }
            }
            ReOp::None => Rc::new(AlphabetPartition::empty()),
            ReOp::All | ReOp::Any => Rc::new(AlphabetPartition::singleton(CharRange::all())),
            ReOp::Concat(rs) => {
                let mut first = rs
                    .first()
                    .map(|r| r.first().as_ref().clone()) // Get first() from the first regex
                    .unwrap_or_else(AlphabetPartition::empty); // Default to empty
                for (i, r) in rs.iter().enumerate().skip(1) {
                    if !rs[i - 1].nullable() {
                        break;
                    } else {
                        let r_first = r.first();
                        first = first.refine(&r_first);
                    }
                }
                Rc::new(first)
            }
            ReOp::Union(rs) | ReOp::Inter(rs) => {
                let mut first = AlphabetPartition::empty();
                for r in rs {
                    first = first.refine(&r.first());
                }
                Rc::new(first)
            }
            ReOp::Star(r) | ReOp::Plus(r) | ReOp::Opt(r) | ReOp::Comp(r) => r.first().clone(),
            ReOp::Range(rs) => Rc::new(AlphabetPartition::singleton(*rs)),
            ReOp::Diff(r, s) => {
                let first = r.first().as_ref().clone();
                Rc::new(first.refine(&s.first()))
            }
            ReOp::Pow(r, _) | ReOp::Loop(r, _, _) => r.first().clone(),
        }
    }

    fn alphabet(&self) -> Rc<Alphabet> {
        let mut alph = Alphabet::empty();
        match self {
            ReOp::Literal(word) => {
                for c in word.iter() {
                    alph.insert_char(*c);
                }
            }
            ReOp::Range(r) => {
                alph.insert(*r);
            }
            ReOp::None | ReOp::Any | ReOp::All => {}
            ReOp::Concat(rs) | ReOp::Union(rs) | ReOp::Inter(rs) => {
                for r in rs {
                    alph = alph.union(r.alphabet().as_ref());
                }
            }
            ReOp::Comp(r)
            | ReOp::Star(r)
            | ReOp::Plus(r)
            | ReOp::Opt(r)
            | ReOp::Pow(r, _)
            | ReOp::Loop(r, _, _) => return r.alphabet().clone(),
            ReOp::Diff(r1, r2) => {
                alph = alph.union(r1.alphabet().as_ref());
                alph = alph.union(r2.alphabet().as_ref());
            }
        }

        Rc::new(alph)
    }

    fn is_constant(&self) -> Option<SmtString> {
        match self {
            ReOp::Literal(word) => Some(word.clone()),
            ReOp::Range(r) => {
                if r.start() == r.end() {
                    Some(SmtString::new(vec![r.start()]))
                } else {
                    None
                }
            }
            ReOp::None | ReOp::Any | ReOp::All => None,
            ReOp::Concat(r) => {
                let mut word = SmtString::empty();
                for re in r {
                    if let Some(w) = re.is_constant() {
                        word = word.concat(&w);
                    } else {
                        return None;
                    }
                }
                Some(word)
            }
            ReOp::Union(rs) | ReOp::Inter(rs) => {
                let mut words = rs.iter().map(|r| r.is_constant()).collect_vec();
                if words.iter().all(|w| w.is_some()) {
                    let word = words.pop().unwrap().unwrap();
                    // check if all words are the same
                    for w in words {
                        if word != w.unwrap() {
                            return None;
                        }
                    }
                    Some(word)
                } else {
                    None
                }
            }
            ReOp::Star(r) | ReOp::Opt(r) | ReOp::Plus(r) => {
                if r.epsilon().unwrap_or(false) {
                    Some(SmtString::empty())
                } else {
                    None
                }
            }

            ReOp::Diff(_, _) | ReOp::Comp(_) => None, // can't determine
            ReOp::Pow(_, 0) | ReOp::Loop(_, _, 0) => Some(SmtString::empty()),
            ReOp::Pow(r, _) => {
                if r.epsilon().unwrap_or(false) {
                    Some(SmtString::empty())
                } else {
                    None
                }
            }
            ReOp::Loop(r, l, u) if l <= u => {
                if r.epsilon().unwrap_or(false) {
                    Some(SmtString::empty())
                } else {
                    None
                }
            }
            ReOp::Loop(_, _, _) => None,
        }
    }

    fn prefix(&self) -> Option<SmtString> {
        match self {
            ReOp::Literal(word) => Some(word.clone()),
            ReOp::None | ReOp::Any | ReOp::All => Some(SmtString::empty()),
            ReOp::Range(r) => r.is_singleton().map(|c| c.into()),
            ReOp::Concat(rs) => {
                let mut prefix = SmtString::empty();
                if rs.is_empty() {
                    return Some(SmtString::empty());
                }
                // while the regexes accept only one word, we can concatenate them
                let mut i = 0;
                while let Some(w) = rs[i].is_constant() {
                    prefix = prefix.concat(&w);
                    i += 1;
                }
                if i == rs.len() {
                    Some(prefix)
                } else {
                    prefix = prefix.concat(&rs[i].prefix()?);
                    Some(prefix)
                }
            }
            ReOp::Union(rs) => {
                let mut prefixes = Vec::with_capacity(rs.len());
                for r in rs {
                    prefixes.push(r.prefix()?);
                }
                if prefixes.is_empty() {
                    return Some(SmtString::empty());
                }
                // Find the longest common prefix
                let mut i = 0;
                let mut done = false;
                while !done {
                    if prefixes.iter().all(|p| i < p.len()) {
                        if !prefixes.iter().map(|p| p[i]).all_equal() {
                            // not all characters are equal
                            done = true;
                        } else {
                            // all characters are equal, check next character
                            i += 1;
                        }
                    } else {
                        done = true;
                    }
                }
                Some(prefixes.first().unwrap().take(i))
            }
            ReOp::Star(_) | ReOp::Opt(_) => Some(SmtString::empty()),
            ReOp::Plus(r) => r.prefix(),
            ReOp::Diff(_, _) | ReOp::Comp(_) | ReOp::Inter(_) => None, // can't determine
            ReOp::Pow(r, _) => r.prefix(),
            ReOp::Loop(r, l, u) if l <= u => r.prefix(),
            ReOp::Loop(_, _, _) => None,
        }
    }

    fn suffix(&self) -> Option<SmtString> {
        match self {
            ReOp::Literal(word) => Some(word.clone()),
            ReOp::None | ReOp::Any | ReOp::All => Some(SmtString::empty()),
            ReOp::Range(r) => r.is_singleton().map(|c| c.into()),
            ReOp::Concat(rs) => {
                let mut suffix = SmtString::empty();
                if rs.is_empty() {
                    return Some(SmtString::empty());
                }
                // while the regexes accept only one word, we can concatenate them
                let mut i = rs.len() - 1;
                while let Some(w) = rs[i].is_constant() {
                    suffix = w.concat(&suffix);
                    i -= 1;
                }
                if i == 0 {
                    Some(suffix)
                } else {
                    suffix = rs[i].suffix()?.concat(&suffix);
                    Some(suffix)
                }
            }
            ReOp::Union(rs) => {
                let mut suffixes = Vec::with_capacity(rs.len());
                for r in rs {
                    suffixes.push(r.suffix()?);
                }
                if suffixes.is_empty() {
                    return Some(SmtString::empty());
                }
                // Find the longest common suffix by reversing the words and searching for the longest common prefix
                let suffixed_revd = suffixes.iter_mut().map(|w| w.reversed()).collect_vec();
                let mut i = 0;
                let mut done = false;
                while !done {
                    if suffixed_revd.iter().all(|p| i < p.len()) {
                        if !suffixed_revd.iter().map(|p| p[i]).all_equal() {
                            // not all characters are equal
                            done = true;
                        } else {
                            // all characters are equal, check next character
                            i += 1;
                        }
                    } else {
                        done = true;
                    }
                }
                let lcs_rev = suffixed_revd.first().unwrap().take(i);
                // reverse the longest common prefix
                Some(lcs_rev.reversed())
            }
            ReOp::Star(_) | ReOp::Opt(_) => Some(SmtString::empty()),
            ReOp::Plus(r) => r.suffix(),
            ReOp::Diff(_, _) | ReOp::Comp(_) | ReOp::Inter(_) => None, // can't determine
            ReOp::Pow(r, _) => r.suffix(),
            ReOp::Loop(r, l, u) if l <= u => r.suffix(),
            ReOp::Loop(_, _, _) => None,
        }
    }

    fn contains_complement(&self) -> bool {
        match self {
            ReOp::Literal(_) | ReOp::Range(_) | ReOp::None | ReOp::Any | ReOp::All => false,
            ReOp::Concat(rs) | ReOp::Union(rs) | ReOp::Inter(rs) => {
                rs.iter().any(|r| r.contains_complement())
            }
            ReOp::Star(r) | ReOp::Plus(r) | ReOp::Opt(r) => r.contains_complement(),
            ReOp::Diff(_, _) | ReOp::Comp(_) => true,
            ReOp::Pow(_, 0) => false,
            ReOp::Pow(r, _) => r.contains_complement(),
            ReOp::Loop(_, l, u) if l > u || *u == 0 => false,
            ReOp::Loop(r, _, _) => r.contains_complement(),
        }
    }

    fn simple(&self) -> bool {
        match self {
            ReOp::Literal(_) | ReOp::Range(_) | ReOp::None | ReOp::Any | ReOp::All => true,
            ReOp::Diff(_, _) | ReOp::Comp(_) | ReOp::Inter(_) => false,
            ReOp::Concat(rs) | ReOp::Union(rs) => rs.iter().all(|r| r.simple()),
            ReOp::Star(r)
            | ReOp::Plus(r)
            | ReOp::Opt(r)
            | ReOp::Pow(r, _)
            | ReOp::Loop(r, _, _) => r.simple(),
        }
    }
}

impl Display for ReOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReOp::Literal(w) => write!(f, "(str.to_re \"{}\")", w),
            ReOp::None => write!(f, "re.none"),
            ReOp::All => write!(f, "re.all"),
            ReOp::Any => write!(f, "re.allchar"),
            ReOp::Concat(rs) => {
                write!(f, "(re.++")?;
                for r in rs {
                    write!(f, " {}", r)?;
                }
                write!(f, ")")
            }
            ReOp::Union(rs) => {
                write!(f, "(re.union")?;
                for r in rs.iter() {
                    write!(f, " {}", r)?;
                }
                write!(f, ")")
            }
            ReOp::Inter(rs) => {
                write!(f, "(re.inter")?;
                for r in rs.iter() {
                    write!(f, " {}", r)?;
                }
                write!(f, ")")
            }
            ReOp::Star(r) => write!(f, "(re.* {})", r),
            ReOp::Plus(p) => write!(f, "(re.+ {})", p),
            ReOp::Opt(r) => write!(f, "(re.opt {})", r),
            ReOp::Range(r) => write!(f, "(re.range \"{}\" \"{}\")", r.start(), r.end()),
            ReOp::Comp(c) => write!(f, "(re.comp {})", c),
            ReOp::Diff(r1, r2) => write!(f, "(re.diff {} {})", r1, r2),
            ReOp::Pow(r, n) => write!(f, "((_ re.^ {}) {})", n, r),
            ReOp::Loop(r, l, u) => write!(f, "((_ re.loop {} {}) {})", l, u, r),
        }
    }
}

/// Checks whether the regular operator denotes a union of characters, expressed as a ranges.
/// If it does, the ranges are returned as a vector. Otherwise, `None` is returned.
pub fn union_of_chars(re: &Regex) -> Option<Vec<CharRange>> {
    match re.op() {
        ReOp::Literal(w) if w.len() == 1 => {
            let c = w.first().unwrap();
            Some(vec![CharRange::singleton(c)])
        }
        ReOp::None => Some(vec![]),
        ReOp::Any => Some(vec![CharRange::all()]),
        ReOp::Union(rs) => {
            let mut ranges = vec![];
            for r in rs {
                ranges.append(&mut union_of_chars(r)?);
            }
            Some(ranges)
        }
        ReOp::Range(r) => Some(vec![*r]),
        _ => None,
    }
}

impl Arbitrary for ReNode {
    fn arbitrary(g: &mut Gen) -> Self {
        use smallvec::smallvec;

        fn gen_regex(g: &mut Gen, builder: &mut ReBuilder, depth: u8) -> Regex {
            fn base_case(g: &mut Gen, n: usize, builder: &mut ReBuilder) -> Regex {
                match n {
                    0 => builder.epsilon(),
                    1 => builder.none(),
                    2 => builder.allchar(),
                    3 => builder.to_re(SmtString::arbitrary(g)),
                    4 => builder.range(CharRange::arbitrary(g)),
                    _ => unreachable!(),
                }
            }
            let choice = usize::arbitrary(g) % if depth == 0 { 5 } else { 15 };

            match choice {
                n if n < 5 => base_case(g, n, builder),
                5 => {
                    let inner = gen_regex(g, builder, depth - 1);
                    builder.star(inner)
                }
                6 => {
                    let inner = gen_regex(g, builder, depth - 1);
                    builder.plus(inner)
                }
                7 => {
                    let inner = gen_regex(g, builder, depth - 1);
                    builder.opt(inner)
                }
                8 => {
                    let inner = gen_regex(g, builder, depth - 1);
                    builder.comp(inner)
                }
                9 => {
                    let left = gen_regex(g, builder, depth - 1);
                    let right = gen_regex(g, builder, depth - 1);
                    builder.concat(smallvec![left, right])
                }
                10 => {
                    let left = gen_regex(g, builder, depth - 1);
                    let right = gen_regex(g, builder, depth - 1);
                    builder.union(smallvec![left, right])
                }
                11 => {
                    let left = gen_regex(g, builder, depth - 1);
                    let right = gen_regex(g, builder, depth - 1);
                    builder.inter(smallvec![left, right])
                }
                12 => {
                    let left = gen_regex(g, builder, depth - 1);
                    let right = gen_regex(g, builder, depth - 1);
                    builder.diff(left, right)
                }
                13 => {
                    let inner = gen_regex(g, builder, depth - 1);
                    let count = u32::arbitrary(g) % 5;
                    builder.pow(inner, count)
                }
                14 => {
                    let inner = gen_regex(g, builder, depth - 1);
                    let lo = u32::arbitrary(g) % 4;
                    let hi = lo + (u32::arbitrary(g) % 4);
                    builder.loop_(inner, lo, hi)
                }
                _ => unreachable!(),
            }
        }

        let mut builder = ReBuilder::default();
        let depth = g.size().min(5) as u8;
        gen_regex(g, &mut builder, depth).as_ref().clone()
    }
}

#[cfg(test)]
mod tests {
    use crate::re::ReBuilder;

    use super::*;

    use quickcheck_macros::quickcheck;
    use smallvec::smallvec;

    #[test]
    fn test_union_of_ranges_const() {
        let mut builder = ReBuilder::non_optimizing();
        let re = builder.to_re("a".into());
        let result = union_of_chars(&re);
        assert_eq!(result, Some(vec![CharRange::singleton('a')]));
    }

    #[test]
    fn test_union_of_ranges_const_word() {
        let mut builder = ReBuilder::non_optimizing();
        let re = builder.to_re("abc".into());
        let result = union_of_chars(&re);
        assert_eq!(result, None);
    }

    #[test]
    fn test_union_of_ranges_none() {
        let builder = ReBuilder::non_optimizing();
        let re = builder.none();
        let result = union_of_chars(&re);
        assert_eq!(result, Some(vec![]));
    }

    #[test]
    fn test_union_of_ranges_all_char() {
        let builder = ReBuilder::non_optimizing();
        let re = builder.allchar();
        let result = union_of_chars(&re);
        assert_eq!(result, Some(vec![CharRange::all()]));
    }

    #[test]
    fn test_union_of_ranges_union() {
        let mut builder = ReBuilder::non_optimizing();
        let alternatives = smallvec![
            builder.to_re("a".into()),
            builder.to_re("b".into()),
            builder.to_re("c".into()),
        ];
        let reg = builder.union(alternatives);
        let result = union_of_chars(&reg);
        assert_eq!(
            result,
            Some(vec![
                CharRange::singleton('a'),
                CharRange::singleton('b'),
                CharRange::singleton('c')
            ])
        );
    }

    #[test]
    fn test_union_of_ranges_range() {
        let mut builder = ReBuilder::non_optimizing();
        let re = builder.range_from_to('a', 'z');
        let result = union_of_chars(&re);
        assert_eq!(result, Some(vec![CharRange::new('a', 'z')]));
    }

    /* Nullable */

    #[quickcheck]
    fn nullable_literal(s: SmtString) {
        let mut builder = ReBuilder::non_optimizing();
        let epsi = builder.to_re(s.clone());
        if s.is_empty() {
            assert!(epsi.nullable());
        } else {
            assert!(!epsi.nullable());
        }
    }

    #[quickcheck]
    fn nullable_range(r: CharRange) {
        let mut builder = ReBuilder::non_optimizing();
        let r = builder.range(r);
        assert!(!r.nullable());
    }

    #[test]
    fn nullable_base_cases() {
        let builder = ReBuilder::non_optimizing();
        assert!(!builder.none().nullable());
        assert!(!builder.allchar().nullable());
        assert!(builder.epsilon().nullable());
        assert!(builder.all().nullable());
    }

    #[quickcheck]
    fn nullable_star(r: ReNode) {
        let mut builder = ReBuilder::non_optimizing();
        let r = builder.regex(&Rc::new(r));
        let star = builder.star(r);
        assert!(star.nullable());
    }

    #[quickcheck]
    fn nullable_opt(r: ReNode) {
        let mut builder = ReBuilder::non_optimizing();
        let r = builder.regex(&Rc::new(r));
        let opt = builder.opt(r);
        assert!(opt.nullable());
    }

    #[quickcheck]
    fn nullable_concat(rs: Vec<ReNode>) {
        let mut builder = ReBuilder::non_optimizing();
        let rs: SmallVec<_> = rs.into_iter().map(|r| builder.regex(&Rc::new(r))).collect();
        let concat = builder.concat(rs.clone());
        if rs.iter().all(|r| r.nullable()) {
            assert!(concat.nullable());
        } else {
            assert!(!concat.nullable());
        }
    }

    #[test]
    fn nullable_concat_all_parts_nullable() {
        let mut builder = ReBuilder::non_optimizing();
        let epsi = builder.epsilon();
        let a = builder.to_re("a".into());

        let r1 = builder.concat(smallvec![epsi.clone(), epsi.clone()]);
        assert!(r1.nullable());

        let r2 = builder.concat(smallvec![epsi, a]);
        assert!(!r2.nullable());
    }

    #[test]
    fn nullable_inter_all_parts_nullable() {
        let mut builder = ReBuilder::non_optimizing();
        let epsi = builder.epsilon();
        let also_epsi = builder.to_re("".into());

        let r1 = builder.inter(smallvec![epsi.clone(), also_epsi.clone()]);
        assert!(r1.nullable());

        let a = builder.to_re("a".into());
        let r2 = builder.inter(smallvec![epsi, a]);
        assert!(!r2.nullable());
    }

    #[quickcheck]
    fn nullable_inter(rs: Vec<ReNode>) {
        let mut builder = ReBuilder::non_optimizing();
        let rs: SmallVec<_> = rs.into_iter().map(|r| builder.regex(&Rc::new(r))).collect();
        let inter = builder.inter(rs.clone());
        if rs.iter().all(|r| r.nullable()) {
            assert!(inter.nullable());
        } else {
            assert!(!inter.nullable());
        }
    }

    #[quickcheck]
    fn nullable_union(rs: Vec<ReNode>) {
        let mut builder = ReBuilder::non_optimizing();
        let rs: SmallVec<_> = rs.into_iter().map(|r| builder.regex(&Rc::new(r))).collect();
        let union = builder.union(rs.clone());
        if rs.iter().any(|r| r.nullable()) {
            assert!(union.nullable());
        } else {
            assert!(!union.nullable());
        }
    }

    #[quickcheck]
    fn nullable_plus(r: ReNode) {
        let mut builder = ReBuilder::non_optimizing();
        let r = builder.regex(&Rc::new(r));
        let plus = builder.plus(r.clone());
        if r.nullable() {
            assert!(plus.nullable());
        } else {
            assert!(!plus.nullable());
        }
    }

    #[quickcheck]
    fn nullable_diff(l: ReNode, r: ReNode) {
        let mut builder = ReBuilder::non_optimizing();
        let l = builder.regex(&Rc::new(l));
        let r = builder.regex(&Rc::new(r));
        let diff = builder.diff(l.clone(), r.clone());

        if l.nullable() && !r.nullable() {
            assert!(diff.nullable());
        } else {
            assert!(!diff.nullable());
        }
    }

    #[quickcheck]
    fn nullable_comp(r: ReNode) {
        let mut builder = ReBuilder::non_optimizing();
        let r = builder.regex(&Rc::new(r));
        let comp = builder.comp(r.clone());
        if !r.nullable() {
            assert!(comp.nullable());
        } else {
            assert!(!comp.nullable());
        }
    }

    #[quickcheck]
    fn nullable_pow(r: ReNode, e: u32) {
        let mut builder = ReBuilder::non_optimizing();
        let r = builder.regex(&Rc::new(r));
        let pow = builder.pow(r.clone(), e);

        if e == 0 || r.nullable() {
            assert!(pow.nullable());
        } else {
            assert!(!pow.nullable());
        }
    }

    #[quickcheck]
    fn nullable_loop(r: ReNode, l: u32, u: u32) {
        let mut builder = ReBuilder::non_optimizing();
        let r = builder.regex(&Rc::new(r));
        let loop_ = builder.loop_(r.clone(), l, u);

        if l <= u {
            if l == 0 || r.nullable() {
                assert!(loop_.nullable());
            } else {
                assert!(!loop_.nullable());
            }
        } else {
            assert!(!loop_.nullable());
        }
    }

    /* Epsilon */

    #[quickcheck]
    fn epsilon_literal(s: SmtString) {
        let mut builder = ReBuilder::non_optimizing();
        let lit = builder.to_re(s.clone());
        assert_eq!(lit.epsilon(), Some(s.is_empty()));
    }

    #[quickcheck]
    fn epsilon_range(r: CharRange) {
        let mut builder = ReBuilder::non_optimizing();
        let r = builder.range(r);
        assert_eq!(r.epsilon(), Some(false));
    }

    #[test]
    fn epsilon_base_cases() {
        let builder = ReBuilder::non_optimizing();
        assert_eq!(builder.none().epsilon(), Some(false));
        assert_eq!(builder.allchar().epsilon(), Some(false));
        assert_eq!(builder.all().epsilon(), Some(false));
        assert_eq!(builder.epsilon().epsilon(), Some(true));
    }

    #[quickcheck]
    fn epsilon_star(r: ReNode) {
        let mut builder = ReBuilder::non_optimizing();
        let r = builder.regex(&Rc::new(r));
        let star = builder.star(r.clone());
        assert_eq!(star.epsilon(), r.epsilon());
    }

    #[quickcheck]
    fn epsilon_opt(r: ReNode) {
        let mut builder = ReBuilder::non_optimizing();
        let r = builder.regex(&Rc::new(r));
        let opt = builder.opt(r.clone());
        assert_eq!(opt.epsilon(), r.epsilon());
    }

    #[test]
    fn epsi_concat_bug() {
        let mut builder = ReBuilder::non_optimizing();
        let inner = smallvec![builder.none()];
        let concat = builder.concat(inner);
        assert_eq!(concat.epsilon(), Some(false));
    }

    #[quickcheck]
    fn epsilon_concat(rs: Vec<ReNode>) {
        let mut builder = ReBuilder::non_optimizing();
        let rs: SmallVec<_> = rs.into_iter().map(|r| builder.regex(&Rc::new(r))).collect();
        let concat = builder.concat(rs.clone());

        let mut expected = Some(true);
        for r in rs {
            match r.epsilon() {
                Some(true) => {}
                Some(false) => {
                    // if one the subexpressions in not epsilon, the whole expression is not epsilon
                    expected = Some(false);
                    break;
                }
                None => {
                    // if we can't determine if a subexpression is epsilon, thus we can't determine the epsilon of the whole expression
                    expected = None;
                    break;
                }
            }
        }
        assert_eq!(concat.epsilon(), expected);
    }

    #[quickcheck]
    fn epsilon_inter(rs: Vec<ReNode>) {
        let mut builder = ReBuilder::non_optimizing();
        let rs: SmallVec<_> = rs.into_iter().map(|r| builder.regex(&Rc::new(r))).collect();
        let inter = builder.inter(rs.clone());

        let mut expected = Some(!rs.is_empty());
        for r in rs {
            match r.epsilon() {
                Some(false) | None => {
                    // if one the subexpressions in not epsilon, the whole expression is not epsilon
                    expected = None;
                    break;
                }
                _ => {}
            }
        }
        assert_eq!(inter.epsilon(), expected);
    }

    #[quickcheck]
    fn epsilon_union(rs: Vec<ReNode>) {
        let mut builder = ReBuilder::non_optimizing();
        let rs: SmallVec<_> = rs.into_iter().map(|r| builder.regex(&Rc::new(r))).collect();
        let union = builder.union(rs.clone());

        let mut expected = Some(!rs.is_empty());
        for r in rs {
            match r.epsilon() {
                Some(true) => {}
                Some(false) => {
                    // if one the subexpressions in not epsilon, the whole expression is not epsilon
                    expected = Some(false);
                    break;
                }
                None => {
                    // if we can't determine the epsilon of a subexpression, we can't determine the epsilon of the whole expression
                    expected = None;
                    break;
                }
            }
        }

        assert_eq!(union.epsilon(), expected);
    }

    #[quickcheck]
    fn epsilon_plus(r: ReNode) {
        let mut builder = ReBuilder::non_optimizing();
        let r = builder.regex(&Rc::new(r));
        let plus = builder.plus(r.clone());
        assert_eq!(plus.epsilon(), r.epsilon());
    }

    #[quickcheck]
    fn epsilon_diff(r1: ReNode, r2: ReNode) {
        let mut builder = ReBuilder::non_optimizing();
        let r1 = builder.regex(&Rc::new(r1));
        let r2 = builder.regex(&Rc::new(r2));
        let diff = builder.diff(r1, r2);

        // Cannot determine in general
        assert_eq!(diff.epsilon(), None);
    }

    #[quickcheck]
    fn epsilon_comp(r: ReNode) {
        let mut builder = ReBuilder::non_optimizing();
        let r = builder.regex(&Rc::new(r));
        let comp = builder.comp(r);
        assert_eq!(comp.epsilon(), None);
    }

    #[quickcheck]
    fn epsilon_pow(r: ReNode, e: u32) {
        let mut builder = ReBuilder::non_optimizing();
        let r = builder.regex(&Rc::new(r));
        let pow = builder.pow(r.clone(), e);

        let expected = if e == 0 { Some(true) } else { r.epsilon() };
        assert_eq!(pow.epsilon(), expected);
    }

    #[quickcheck]
    fn epsilon_loop(r: ReNode, l: u32, u: u32) {
        let mut builder = ReBuilder::non_optimizing();
        let r = builder.regex(&Rc::new(r));
        let loop_ = builder.loop_(r.clone(), l, u);

        let expected = if l > u {
            Some(false)
        } else if u == 0 {
            Some(true)
        } else {
            r.epsilon()
        };
        assert_eq!(loop_.epsilon(), expected);
    }

    /* Universal */

    #[quickcheck]
    fn universal_literal(s: SmtString) {
        let mut builder = ReBuilder::non_optimizing();
        let lit = builder.to_re(s.clone());
        assert_eq!(lit.universal(), Some(false));
    }

    #[quickcheck]
    fn universal_range(r: CharRange) {
        let mut builder = ReBuilder::non_optimizing();
        let r = builder.range(r);
        assert_eq!(r.universal(), Some(false));
    }

    #[test]
    fn universal_base_cases() {
        let builder = ReBuilder::non_optimizing();
        assert_eq!(builder.none().universal(), Some(false));
        assert_eq!(builder.allchar().universal(), Some(false));
        assert_eq!(builder.all().universal(), Some(true));
        assert_eq!(builder.epsilon().universal(), Some(false));
    }

    #[quickcheck]
    fn universal_star(r: ReNode) {
        let mut builder = ReBuilder::non_optimizing();
        let r = builder.regex(&Rc::new(r));
        let star = builder.star(r.clone());
        assert_eq!(star.universal(), r.universal());
    }

    #[quickcheck]
    fn universal_opt(r: ReNode) {
        let mut builder = ReBuilder::non_optimizing();
        let r = builder.regex(&Rc::new(r));
        let opt = builder.opt(r.clone());
        assert_eq!(opt.universal(), r.universal());
    }

    #[quickcheck]
    fn universal_concat(rs: Vec<ReNode>) {
        let mut builder = ReBuilder::non_optimizing();
        let rs: SmallVec<_> = rs.into_iter().map(|r| builder.regex(&Rc::new(r))).collect();
        let concat = builder.concat(rs.clone());

        let mut expected = Some(false);
        for r in rs {
            match r.universal() {
                Some(true) => {
                    expected = Some(true);
                }
                Some(false) => {
                    // if one the subexpressions in not epsilon, the whole expression is not epsilon
                    expected = Some(false);
                    break;
                }
                None => {
                    // if we can't determine the epsilon of a subexpression, we can't determine the epsilon of the whole expression
                    expected = None;
                    break;
                }
            }
        }
        assert_eq!(concat.universal(), expected);
    }

    #[quickcheck]
    fn universal_inter(rs: Vec<ReNode>) {
        let mut builder = ReBuilder::non_optimizing();
        let rs: SmallVec<_> = rs.into_iter().map(|r| builder.regex(&Rc::new(r))).collect();
        let inter = builder.inter(rs.clone());

        let mut expected = Some(rs.is_empty());
        for r in rs {
            match r.universal() {
                Some(true) => {}
                Some(false) => {
                    // if one the subexpressions in not epsilon, the whole expression is not epsilon
                    expected = Some(false);
                    break;
                }
                None => {
                    // if we can't determine the epsilon of a subexpression, we can't determine the epsilon of the whole expression
                    expected = None;
                    break;
                }
            }
        }
        assert_eq!(inter.universal(), expected);
    }

    #[quickcheck]
    fn universal_union(rs: Vec<ReNode>) {
        let mut builder = ReBuilder::non_optimizing();
        let rs: SmallVec<_> = rs.into_iter().map(|r| builder.regex(&Rc::new(r))).collect();
        let union = builder.union(rs.clone());

        let mut expected = if rs.is_empty() { Some(false) } else { None };
        for r in rs {
            if let Some(true) = r.universal() {
                expected = Some(true);
                break;
            }
        }
        assert_eq!(union.universal(), expected);
    }

    #[quickcheck]
    fn universal_plus(r: ReNode) {
        let mut builder = ReBuilder::non_optimizing();
        let r = builder.regex(&Rc::new(r));
        let plus = builder.plus(r.clone());
        assert_eq!(plus.universal(), r.universal());
    }

    #[quickcheck]
    fn universal_diff(r1: ReNode, r2: ReNode) {
        let mut builder = ReBuilder::non_optimizing();
        let r1 = builder.regex(&Rc::new(r1));
        let r2 = builder.regex(&Rc::new(r2));
        let diff = builder.diff(r1.clone(), r2.clone());

        // Cannot determine in general
        let expected = match (r1.universal(), r2.none()) {
            (Some(false), _) => Some(false), // Removing something from non-universal is non-universal
            (Some(true), Some(true)) => Some(true), // Removing nothing from universal is universal
            (_, Some(false)) => Some(false), // Removing something from anything is non-universal
            _ => None,
        };
        assert_eq!(diff.universal(), expected);
    }

    #[quickcheck]
    fn universal_comp(r: ReNode) {
        let mut builder = ReBuilder::non_optimizing();
        let r = builder.regex(&Rc::new(r));
        let comp = builder.comp(r.clone());
        if let Some(true) = r.none() {
            assert_eq!(comp.universal(), Some(true));
        }
    }

    #[quickcheck]
    fn universal_pow(r: ReNode, e: u32) {
        let e = e % 20;
        let mut builder = ReBuilder::non_optimizing();
        let r = builder.regex(&Rc::new(r));
        let pow = builder.pow(r.clone(), e);

        if r.universal().unwrap_or(false) {
            if e == 0 {
                assert_eq!(pow.universal(), Some(false));
            } else {
                assert_eq!(pow.universal(), Some(true));
            }
        }
    }

    #[quickcheck]
    fn universal_loop(r: ReNode, l: u32, u: u32) {
        let mut builder = ReBuilder::non_optimizing();
        let r = builder.regex(&Rc::new(r));
        let loop_ = builder.loop_(r.clone(), l, u);
        if l > u {
            assert_eq!(loop_.universal(), Some(false));
        } else if r.universal().unwrap_or(false) {
            if u == 0 {
                assert_eq!(loop_.universal(), Some(false));
            } else {
                assert_eq!(loop_.universal(), Some(true));
            }
        }
    }

    #[test]
    fn test_accept() {
        // (assert (str.in_re X (re.++ (str.to_re "x") ((_ re.loop 4 4) (re.union (str.to_re "e") (str.to_re "d"))) (str.to_re "x"))))
        let mut builder = ReBuilder::non_optimizing();
        let x = builder.to_re("x".into());
        let e = builder.to_re("e".into());
        let d = builder.to_re("d".into());
        let union = builder.union(smallvec![e.clone(), d.clone()]);
        let looped = builder.loop_(union.clone(), 4, 4);
        let re = builder.concat(smallvec![x.clone(), looped.clone(), x.clone()]);

        assert!(
            re.accepts(&SmtString::from("xededx")),
            "Expected xededx to be accepted by {}",
            re
        );
    }

    #[test]
    fn test_is_const_none() {
        let builder = ReBuilder::non_optimizing();
        let none = builder.none();
        let result = none.is_constant();
        assert_eq!(result, None);
    }

    #[test]
    fn test_is_const_all() {
        let builder = ReBuilder::non_optimizing();
        let all = builder.all();
        let result = all.is_constant();
        assert_eq!(result, None);
    }

    #[test]
    fn test_is_const_allchar() {
        let builder = ReBuilder::non_optimizing();
        let any = builder.allchar();
        let result = any.is_constant();
        assert_eq!(result, None);
    }

    #[test]
    fn test_is_const_concat_consts() {
        let mut builder = ReBuilder::non_optimizing();
        let a = builder.to_re("a".into());
        let b = builder.to_re("b".into());
        let c = builder.to_re("c".into());
        let abc = builder.concat(smallvec![a.clone(), b.clone(), c.clone()]);
        let result = abc.is_constant();
        assert_eq!(result, Some(SmtString::from("abc")));
    }

    #[test]
    fn test_is_const_const() {
        let mut builder = ReBuilder::non_optimizing();
        let abc = builder.to_re("abc".into());

        let result = abc.is_constant();
        assert_eq!(result, Some(SmtString::from("abc")));
    }

    #[test]
    fn test_is_const_union() {
        let mut builder = ReBuilder::non_optimizing();
        let abc = builder.to_re("abc".into());
        let def = builder.to_re("def".into());

        let equal_union = builder.union(smallvec![abc.clone(), abc.clone()]);
        let different_union = builder.union(smallvec![abc.clone(), def.clone()]);
        assert_eq!(equal_union.is_constant(), Some(SmtString::from("abc")));
        assert_eq!(different_union.is_constant(), None);
    }

    #[test]
    fn test_is_const_inter() {
        let mut builder = ReBuilder::non_optimizing();
        let abc = builder.to_re("abc".into());
        let def = builder.to_re("def".into());

        let equal_union = builder.inter(smallvec![abc.clone(), abc.clone()]);
        let different_union = builder.inter(smallvec![abc.clone(), def.clone()]);
        assert_eq!(equal_union.is_constant(), Some(SmtString::from("abc")));
        assert_eq!(different_union.is_constant(), None);
    }

    #[test]
    fn test_is_const_star() {
        let mut builder = ReBuilder::non_optimizing();
        let epsi = builder.to_re("".into());
        let a = builder.to_re("a".into());
        let star_a = builder.star(a.clone());
        let star_epsi = builder.star(epsi.clone());
        assert_eq!(star_a.is_constant(), None);
        assert_eq!(star_epsi.is_constant(), Some(SmtString::empty()));
    }

    #[test]
    fn test_is_const_plus() {
        let mut builder = ReBuilder::non_optimizing();
        let epsi = builder.to_re("".into());
        let a = builder.to_re("a".into());
        let plus_a = builder.plus(a.clone());
        let plus_epsi = builder.plus(epsi.clone());
        assert_eq!(plus_a.is_constant(), None);
        assert_eq!(plus_epsi.is_constant(), Some(SmtString::empty()));
    }

    #[test]
    fn test_is_const_opt() {
        let mut builder = ReBuilder::non_optimizing();
        let epsi = builder.to_re("".into());
        let a = builder.to_re("a".into());
        let opt_a = builder.opt(a.clone());
        let opt_epsi = builder.opt(epsi.clone());
        assert_eq!(opt_a.is_constant(), None);
        assert_eq!(opt_epsi.is_constant(), Some(SmtString::empty()));
    }

    #[test]
    fn test_is_const_range() {
        let mut builder = ReBuilder::non_optimizing();
        let a = builder.range_from_to('a', 'a');
        let ab = builder.range_from_to('a', 'b');
        assert_eq!(a.is_constant(), Some(SmtString::from("a")));
        assert_eq!(ab.is_constant(), None);
    }

    #[test]
    fn prefix_bug() {
        // (re.++ (str.to_re "x") ((_ re.^ 4) (re.union (str.to_re "e") (str.to_re "d"))) (str.to_re "x"))
        let mut builder = ReBuilder::non_optimizing();
        let x = builder.to_re("x".into());
        let e = builder.to_re("e".into());
        let d = builder.to_re("d".into());
        let union = builder.union(smallvec![e.clone(), d.clone()]);
        let pow = builder.pow(union.clone(), 4);
        let re = builder.concat(smallvec![x.clone(), pow.clone(), x.clone()]);
        let prefix = re.prefix();
        assert_eq!(prefix, Some(SmtString::from("x")));
    }

    #[test]
    fn prefix_suf() {
        // (re.++ (str.to_re "x") ((_ re.^ 4) (re.union (str.to_re "e") (str.to_re "d"))) (str.to_re "x"))
        let mut builder = ReBuilder::non_optimizing();
        let x = builder.to_re("x".into());
        let e = builder.to_re("e".into());
        let d = builder.to_re("d".into());
        let union = builder.union(smallvec![e.clone(), d.clone()]);
        let pow = builder.pow(union.clone(), 4);
        let re = builder.concat(smallvec![x.clone(), pow.clone(), x.clone()]);
        let prefix = re.suffix();
        assert_eq!(prefix, Some(SmtString::from("x")));
    }
}
