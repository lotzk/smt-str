//! Regular Expressions

mod build;
pub mod deriv;

#[cfg(feature = "sampling")]
pub mod sampling;

use smallvec::SmallVec;

use itertools::Itertools;

use std::cell::RefCell;

use std::hash::Hash;
use std::{fmt::Display, rc::Rc};

pub use build::ReBuilder;

use crate::alphabet::{Alphabet, AlphabetPartition, CharRange};
use crate::SmtString;

pub type ReId = usize;

type LazyProp<T> = RefCell<Option<T>>;

/// A reference-counted regular expression.
///
/// This is a type alias for [`Rc<ReNode>`], meaning every `Regex` instance is
/// a reference-counted pointer to an immutable regex node. This allows for
/// **structural sharing** and **deduplication** of sub-expressions, reducing
/// redundant computation and improving efficiency.
///
/// Using `Rc<ReNode>` ensures that identical sub-expressions are stored only
/// once, making operations like equality checks and hashing **O(1)**.
///
/// ## Immutability
/// Since `Regex` is an `Rc`-wrapped structure, it is **immutable** after creation.
/// To construct a new regex, use [`ReBuilder`].
pub type Regex = Rc<ReNode>;

/// The internal representation of a regular expression.
///
/// This structure holds a **unique identifier**, an **operation type** (`ReOp`),
/// and **lazily computed properties** to optimize performance. Each `ReNode`
/// represents a **single node** in the abstract syntax tree (AST) of a regex.
///
/// Every `ReNode` has a unique identifier assigned by [`ReBuilder`].
/// This identifier is used for **equality comparison** and **hashing**, enabling
/// these operations to run in **O(1) time** instead of recursively traversing
/// the regex structure.
///
/// **Note:** `ReNode` should not be used directly. Instead, use the [`Regex`] alias,
/// which wraps it in an `Rc` for reference counting.
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

impl ReNode {
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

    pub fn id(&self) -> ReId {
        self.id
    }

    pub fn op(&self) -> &ReOp {
        &self.op
    }

    /// Returns whether the regular expression is nullable.
    /// A regular expression is nullable if it accepts the empty word.
    pub fn nullable(&self) -> bool {
        *self
            .nullable
            .borrow_mut()
            .get_or_insert_with(|| self.op.nullable())
    }

    /// Returns whether the regular expression is simple.
    /// A regular expression is simple if it does not contain complement, difference, or intersection.
    pub fn simple(&self) -> bool {
        *self
            .simple
            .borrow_mut()
            .get_or_insert_with(|| self.op.simple())
    }

    /// Returns whether the regular expression is universal.
    /// A regular expression is universal if it accepts every word.
    /// If it is a simple regex, this operation always returns `Some`.
    /// For extended regexes, this operation may return `None` if the universality cannot be determined.
    pub fn universal(&self) -> Option<bool> {
        *self
            .universal
            .borrow_mut()
            .get_or_insert_with(|| self.op.universal())
    }

    /// Returns whether the regular expression is the empty set.
    /// A regular expression is the empty set if it does not accept any word.
    /// If it is a simple regex, this operation always returns `Some`.
    /// For extended regexes, this operation may return `None` if the emptiness cannot be determined.
    pub fn none(&self) -> Option<bool> {
        *self.none.borrow_mut().get_or_insert_with(|| self.op.none())
    }

    /// Returns whether the regular expression denotes the empty word.
    /// A regular expression denotes the empty word if it only accepts the empty word.
    /// If it is a simple regex, this operation always returns `Some`.
    /// For extended regexes, this operation may return `None` if the emptiness cannot be determined.
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
    pub fn alphabet(&self) -> Rc<Alphabet> {
        self.alphabet
            .borrow_mut()
            .get_or_insert_with(|| self.op.alphabet())
            .clone()
    }

    /// Returns Some(word) if the regular expression accepts only the given constant word, None otherwise or if it cannot be determined.
    pub fn is_constant(&self) -> Option<SmtString> {
        self.is_constant
            .borrow_mut()
            .get_or_insert_with(|| self.op().is_constant())
            .clone()
    }

    /// Returns the prefix of all words accepted by the regular expression.
    /// Makes a best effort to obtain the longest prefix, but does not guarantee it.
    /// Is `None` if the prefix cannot be determined, which is the case for some extended regexes.
    pub fn prefix(&self) -> Option<SmtString> {
        self.prefix
            .borrow_mut()
            .get_or_insert_with(|| self.op().prefix())
            .clone()
    }

    /// Returns the suffix of all words accepted by the regular expression.
    /// Makes a best effort to obtain the longest suffix, but does not guarantee it.
    /// Is `None` if the suffix cannot be determined, which is the case for some extended regexes.
    pub fn suffix(&self) -> Option<SmtString> {
        self.suffix
            .borrow_mut()
            .get_or_insert_with(|| self.op().suffix())
            .clone()
    }

    /// Returns whether the regular expression contains a complement operation.
    pub fn contains_complement(&self) -> bool {
        *self
            .contains_complement
            .borrow_mut()
            .get_or_insert_with(|| self.op().contains_complement())
    }

    /// Return whether the regular expression accepts a given word.
    pub fn accepts(&self, w: &SmtString) -> bool {
        let mut builder_tmp = ReBuilder::default();
        let mnged = builder_tmp.regex(&Rc::new(self.clone()));
        deriv::deriv_word(&mnged, w.clone(), &mut builder_tmp).nullable()
    }
}

impl Display for ReNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.op)
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
            ReOp::Loop(r, l, _) => *l == 0 || r.nullable(),
        }
    }

    /// Compute whether the regular expression is universal, i.e., accepts all words.
    fn universal(&self) -> Option<bool> {
        match self {
            ReOp::Literal(_) | ReOp::None | ReOp::Range(_) | ReOp::Any => Some(false),
            ReOp::All => Some(true),
            ReOp::Concat(rs) | ReOp::Inter(rs) => rs
                .iter()
                .try_fold(true, |a, r| r.universal().map(|b| a && b)),
            ReOp::Union(rs) => rs.iter().any(|r| r.universal().unwrap_or(false)).into(),
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
            ReOp::Concat(rs) | ReOp::Inter(rs) => {
                rs.iter().try_fold(false, |a, r| r.none().map(|b| a || b))
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
                // All subexpressions must accept ε and only ε.
                rs.iter().try_fold(true, |a, r| r.epsilon().map(|b| a && b))
            }
            ReOp::Inter(rs) => {
                // If all subexpressions accept only ε, then intersection does as well.
                if let Some(true) = rs.iter().try_fold(true, |a, r| r.epsilon().map(|b| a && b)) {
                    Some(true)
                } else {
                    // We can't determine if the intersection contains only ε.
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
            ReOp::Range(r) if r.is_singleton() => Some(r.start().into()),
            ReOp::Range(_) => None,
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
            ReOp::Union(rs) | ReOp::Inter(rs) => {
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
                    if prefixes.iter().all(|p| p.len() < i) {
                        done = !prefixes.iter().map(|p| p[i]).all_equal();
                        i += 1;
                    } else {
                        done = true;
                    }
                }
                Some(prefixes.first().unwrap().take(i))
            }
            ReOp::Star(_) | ReOp::Opt(_) => Some(SmtString::empty()),
            ReOp::Plus(r) => r.prefix(),
            ReOp::Diff(_, _) | ReOp::Comp(_) => None, // can't determine
            ReOp::Pow(r, _) => r.prefix(),
            ReOp::Loop(r, l, u) if l <= u => r.prefix(),
            ReOp::Loop(_, _, _) => None,
        }
    }

    fn suffix(&self) -> Option<SmtString> {
        match self {
            ReOp::Literal(word) => Some(word.clone()),
            ReOp::None | ReOp::Any | ReOp::All => Some(SmtString::empty()),
            ReOp::Range(r) if r.is_singleton() => Some(r.start().into()),
            ReOp::Range(_) => None,
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
            ReOp::Union(rs) | ReOp::Inter(rs) => {
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
                    if suffixed_revd.iter().all(|p| p.len() < i) {
                        done = !suffixes.iter().map(|p| p[i]).all_equal();
                        i += 1;
                    } else {
                        done = true;
                    }
                }
                let lcs_rev = suffixes.first().unwrap().take(i);
                // reverse the longest common prefix
                Some(lcs_rev.reversed())
            }
            ReOp::Star(_) | ReOp::Opt(_) => Some(SmtString::empty()),
            ReOp::Plus(r) => r.suffix(),
            ReOp::Diff(_, _) | ReOp::Comp(_) => None, // can't determine
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
            ReOp::Diff(_, _) | ReOp::Comp(_) => false,
            ReOp::Concat(rs) | ReOp::Union(rs) | ReOp::Inter(rs) => rs.iter().all(|r| r.simple()),
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
            ReOp::Literal(w) => write!(f, "'{}'", w),
            ReOp::None => write!(f, "∅"),
            ReOp::All => write!(f, "(Σ*)"),
            ReOp::Any => write!(f, "Σ"),
            ReOp::Concat(rs) => {
                write!(f, "(")?;
                for r in rs {
                    write!(f, "{}", r)?;
                }
                write!(f, ")")
            }
            ReOp::Union(rs) => {
                write!(f, "(")?;
                for (i, r) in rs.iter().enumerate() {
                    if i > 0 {
                        write!(f, " | ")?;
                    }
                    write!(f, "{}", r)?;
                }
                write!(f, ")")
            }
            ReOp::Inter(rs) => {
                write!(f, "(")?;
                for (i, r) in rs.iter().enumerate() {
                    if i > 0 {
                        write!(f, " & ")?;
                    }
                    write!(f, "{}", r)?;
                }
                write!(f, ")")
            }
            ReOp::Star(r) => write!(f, "{}*", r),
            ReOp::Plus(p) => write!(f, "{}+", p),
            ReOp::Opt(r) => write!(f, "{}?", r),
            ReOp::Range(r) => write!(f, "{}", r),
            ReOp::Comp(c) => write!(f, "¬{}", c),
            ReOp::Diff(r1, r2) => write!(f, "({} - {})", r1, r2),
            ReOp::Pow(r, n) => write!(f, "({}^{})", r, n),
            ReOp::Loop(r, l, u) => write!(f, "({}^({}..{}))", r, l, u),
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

#[cfg(test)]
mod tests {
    use crate::re::ReBuilder;

    use super::*;
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
        let re = builder.any_char();
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

    #[test]
    fn first_concat_nullalble() {
        let mut builder = ReBuilder::non_optimizing();
        let a2c = builder.range_from_to('a', 'c');
        let d2e = builder.range_from_to('d', 'e');
        let a2c_star = builder.star(a2c.clone());
        let re = builder.concat(smallvec![a2c_star.clone(), d2e.clone()]);

        let first = re.first();

        assert_eq!(first.iter().count(), 2);
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
        let any = builder.any_char();
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
}
