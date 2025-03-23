//! Character ranges and alphabets for SMT-LIB strings.
//!
//! This module provides types for describing and manipulating sets of characters.
//! In SMT-LIB, a single character is represented as a Unicode code point in the range from `0` to `0x2FFFF` inclusive (defined in the `SmtChar` type).
//! A [`CharRange`] is a contiguous inclusive range of characters, defined by a start and an end character.
//! An [`Alphabet`] is a set of `CharRange`s, representing the union of all characters in the ranges.
//!
//! ```rust
//! use smt_str::alphabet::{CharRange, Alphabet};
//!
//! // This range contains all characters from 'a' to 'z' inclusive
//! let alpha = CharRange::new('a', 'z');
//! assert_eq!(alpha.len(), 26);
//! assert!(alpha.contains('a'));
//! assert!(alpha.contains('z'));
//! assert!(!alpha.contains('A'));
//!
//! // This range contains all character from '0' to '9' inclusive
//! let digits = CharRange::new('0', '9');
//! assert_eq!(digits.len(), 10);
//! assert!(digits.contains('0'));
//! assert!(digits.contains('9'));
//! assert!(!digits.contains('a'));
//!
//! // The union of the two ranges is an alphabet that contains all lowercase letters and digits
//!
//! let mut alphabet = Alphabet::default();
//! alphabet.insert(alpha);
//! alphabet.insert(digits);
//! assert_eq!(alphabet.len(), 36);
//! assert!(alphabet.contains('a'));
//! assert!(alphabet.contains('z'));
//! assert!(alphabet.contains('0'));
//! assert!(alphabet.contains('9'));
//! ```
//!
//! The [`Alphabet`] provides operations supports operations such as union, intersection, and complement. The type always maintains the most compact representation of the alphabet, i.e., it
//! merges overlapping and adjacent ranges into a single range.
//! For example, the alphabet that contains all lowercase letters and digits is represented as a single range `[0-9a-z]`. See [`Alphabet`] for more details.
//!
//! A different representation of the alphabet is provided by the [`AlphabetPartition`] type, which divides the SMT-LIB alphabet into a set of disjoint partitions.
//! Each partition is a contiguous range of characters that are treated. The partitioning can be refined by splitting partitions into smaller partitions.
//! See [partition] module for more details.

pub mod partition;

use std::{cmp::Ordering, fmt::Display};

use crate::{CharIterator, SmtChar};

/// A range of characters [SmtChar]s defined by a start and an end character.
///
/// Every range covers all characters between the start and the end character, including the start and the end character themselves.
/// If the start character is greater than the end character, the range is empty, otherwise it spans exactly `end - start + 1` characters.
///
/// # Ordering
///
/// Ranges are ordered lexicographically.
/// That is, if `[s1, e1]` and `[s2, e2]` are two ranges, then `[s1, e1] < [s2, e2]` precisely if
///
/// - `s1 < s2`,
/// - or `s1 == s2` and `e1 < e2`.
///
/// ## Example
/// The range `['a', 'k']` is less than the range `['a', 'z']` which is less than the range `['b', 'z']`.
/// ```rust
/// use smt_str::alphabet::CharRange;
/// assert!(CharRange::new('a', 'k') < CharRange::new('a', 'z'));
/// assert!(CharRange::new('a', 'z') < CharRange::new('b', 'z'));
/// ```
///
/// # Set operations
///
/// `CharRange` supports several common set-theoretic operations:
///
/// - **Intersection**: The intersection of two ranges is the range that contains all characters that are in both ranges (see [intersect](#method.intersect)).
/// - **Complement**: The complement of a range is the range(s) containing all characters (in the SMT-LIB alphabet) that are not in the original range (see [complement](#method.complement)).
/// - **Difference**: The difference of two ranges is the range(s) containing all characters that are in the first range but not in the second range (see [subtract](#method.subtract)).
/// - **Subset check**: Check if a range is a subset of another range (see [covers](#method.covers)).
///
/// All above operations are performed in O(1) time.
/// Unions of ranges are not directly supported by the CharRange type but are represented by the [Alphabet] type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CharRange {
    start: SmtChar,
    end: SmtChar,
}

impl CharRange {
    /// Create a new range of characters between two characters `start` and `end` (inclusive), where `start` and `end` can be any type that can be converted into a [SmtChar].
    /// If `start > end`, the range is empty.
    /// If `start == end`, the range contains a single character.
    ///
    /// # Example
    /// ```
    /// use smt_str::alphabet::CharRange;
    /// use smt_str::SmtChar;
    ///
    /// let range = CharRange::new('a', 'z');
    /// assert_eq!(range.len(), 26);
    ///
    /// let empty = CharRange::new('z', 'a');
    /// assert!(empty.is_empty());
    ///
    /// let single = CharRange::new('a', 'a');
    /// assert_eq!(single.is_singleton(), Some('a'.into()));
    /// ```
    pub fn new(start: impl Into<SmtChar>, end: impl Into<SmtChar>) -> Self {
        CharRange {
            start: start.into(),
            end: end.into(),
        }
    }

    /// Creates an empty range that contains no characters.
    ///
    /// # Example
    /// ```
    /// use smt_str::alphabet::CharRange;
    ///
    /// let range = CharRange::empty();
    /// assert!(range.is_empty());
    /// ```
    pub fn empty() -> Self {
        CharRange {
            start: SmtChar::new(1),
            end: SmtChar::new(0),
        }
    }

    /// Create a range that contains a single character.
    ///
    /// # Example
    /// ```
    /// use smt_str::alphabet::CharRange;
    ///
    /// let range = CharRange::singleton('a');
    /// assert_eq!(range.len(), 1);
    /// assert!(range.contains('a'));
    /// ```
    pub fn singleton(c: impl Into<SmtChar>) -> Self {
        let c = c.into();
        CharRange { start: c, end: c }
    }

    /// Create a range that covers all characters in the SMT-LIB alphabet, i.e., all characters in the range `[0, 0x2FFFF]`.
    ///
    /// # Example
    /// ```
    /// use smt_str::alphabet::CharRange;
    /// use smt_str::SmtChar;
    ///
    /// let range = CharRange::all();
    /// assert_eq!(range.len(), 0x2FFFF + 1);
    /// assert!(range.contains(0));
    /// assert!(range.contains(SmtChar::MAX));
    /// ```
    pub fn all() -> Self {
        CharRange {
            start: SmtChar::new(0),
            end: SmtChar::new(0x2FFFF),
        }
    }

    /// Return the number of characters in the range.
    ///
    /// # Example
    /// ```
    /// use smt_str::alphabet::CharRange;
    /// use smt_str::SmtChar;
    ///
    /// assert_eq!(CharRange::new('a', 'z').len(), 26);
    /// assert_eq!(CharRange::singleton('a').len(), 1);
    /// assert_eq!(CharRange::new('z', 'a').len(), 0);
    /// ```
    pub fn len(&self) -> usize {
        if self.start > self.end {
            0
        } else {
            ((self.end.0 - self.start.0) + 1) as usize
        }
    }

    /// Returns an iterator over all characters in the range.
    /// The iterator returns all characters in the range, including the start and the end character.
    /// If the range is empty, the iterator returns no characters.
    ///
    /// # Example
    /// ```
    /// use smt_str::alphabet::CharRange;
    /// use smt_str::SmtChar;
    ///
    /// let range = CharRange::new('a', 'c');
    /// let mut iter = range.iter();
    /// assert_eq!(iter.next(), Some('a'.into()));
    /// assert_eq!(iter.next(), Some('b'.into()));
    /// assert_eq!(iter.next(), Some('c'.into()));
    /// assert_eq!(iter.next(), None);
    /// ```
    pub fn iter(&self) -> impl Iterator<Item = SmtChar> {
        CharIterator::new(self.start, self.end)
    }

    /// Check if the range is empty. The range is empty if the start character is greater than the end character.
    ///
    /// # Example
    /// ```
    /// use smt_str::alphabet::CharRange;
    /// use smt_str::SmtChar;
    ///
    /// assert!(CharRange::empty().is_empty());
    /// assert!(!CharRange::new('a', 'z').is_empty());
    /// assert!(!CharRange::singleton('a').is_empty());
    /// assert!(CharRange::new('z', 'a').is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the lower bound of the range.
    ///
    /// # Example
    /// ```
    /// use smt_str::alphabet::CharRange;
    /// use smt_str::SmtChar;
    ///
    /// let range = CharRange::new('a', 'z');
    /// assert_eq!(range.start(), 'a'.into());
    ///
    /// // Does not check for empty range
    /// let empty = CharRange::new('z', 'a');
    /// assert_eq!(empty.start(), 'z'.into());
    /// ```
    pub fn start(&self) -> SmtChar {
        self.start
    }

    /// Returns the upper bound of the range.
    ///
    /// # Example
    /// ```
    /// use smt_str::alphabet::CharRange;
    /// use smt_str::SmtChar;
    ///
    /// let range = CharRange::new('a', 'z');
    /// assert_eq!(range.end(), 'z'.into());
    ///
    /// // Does not check for empty range
    /// let empty = CharRange::new('z', 'a');
    /// assert_eq!(empty.end(), 'a'.into());
    /// ```
    pub fn end(&self) -> SmtChar {
        self.end
    }

    /// Returns a character from the range.
    /// If the range is empty, returns `None`.
    ///
    /// # Example
    /// ```
    /// use smt_str::alphabet::CharRange;
    /// use smt_str::SmtChar;
    ///
    /// let range = CharRange::new('a', 'z');
    /// assert!(range.choose().is_some());
    /// assert_eq!(CharRange::empty().choose(), None);
    /// ```
    pub fn choose(&self) -> Option<SmtChar> {
        if self.is_empty() {
            None
        } else {
            Some(self.start)
        }
    }

    /// Check if the range contains a single character.
    ///
    ///
    /// # Example
    /// ```
    /// use smt_str::alphabet::CharRange;
    /// use smt_str::SmtChar;
    ///
    /// assert!(CharRange::singleton('a').is_singleton().is_some());
    /// assert!(CharRange::new('a', 'z').is_singleton().is_none());
    /// assert!(CharRange::empty().is_singleton().is_none());
    /// ```
    pub fn is_singleton(&self) -> Option<SmtChar> {
        if self.start == self.end {
            Some(self.start)
        } else {
            None
        }
    }

    /// Check if the range contains all characters in the SMT-LIB alphabet.
    ///
    /// # Example
    /// ```
    /// use smt_str::alphabet::CharRange;
    /// use smt_str::SmtChar;
    ///
    /// assert!(CharRange::all().is_full());
    /// assert!(CharRange::new(SmtChar::MIN, SmtChar::MAX).is_full());
    /// assert!(!CharRange::empty().is_full());
    /// assert!(!CharRange::new('a', 'z').is_full());
    /// ```
    pub fn is_full(&self) -> bool {
        self.start == SmtChar::MIN && self.end == SmtChar::MAX
    }

    /// Check if a character is in the range.
    /// Returns true if the character is in the range, false otherwise.
    ///
    /// # Example
    /// ```
    /// use smt_str::alphabet::CharRange;
    /// use smt_str::SmtChar;
    ///
    /// let range = CharRange::new('a', 'z');
    /// assert!(range.contains('a'));
    /// assert!(range.contains('z'));
    /// assert!(range.contains('m'));
    /// assert!(range.contains(98)); // 'a'
    /// assert!(!range.contains('A'));
    /// assert!(!range.contains('0'));
    /// ``````
    pub fn contains(&self, c: impl Into<SmtChar>) -> bool {
        let c = c.into();
        self.start <= c && c <= self.end
    }

    /// Checks if this range is a superset of another range.
    /// Returns true if this range contains all characters in the other range.
    ///
    /// # Example
    /// ```
    /// use smt_str::alphabet::CharRange;
    /// use smt_str::SmtChar;
    ///
    /// let r1 = CharRange::new('a', 'c');
    /// let r2 = CharRange::new('a', 'b');
    /// let r3 = CharRange::new('b', 'f');
    ///
    /// assert!(r1.covers(&r2));
    /// assert!(!r1.covers(&r3));
    /// assert!(r1.covers(&r1));
    /// ```
    pub fn covers(&self, other: &Self) -> bool {
        self.start <= other.start && self.end >= other.end
    }

    /// Return the intersection of two ranges.
    /// The intersection of two ranges is the range that contains all characters that are in both ranges.
    /// If the two ranges do not overlap, the intersection is empty.
    ///
    /// # Example
    /// ```
    /// use smt_str::alphabet::CharRange;
    /// use smt_str::SmtChar;
    ///
    /// let r1 = CharRange::new('a', 'm');
    /// let r2 = CharRange::new('a', 'z');
    /// let r3 = CharRange::singleton('a');
    /// let r4 = CharRange::new('y', 'z');
    ///
    /// assert_eq!(r1.intersect(&r2), CharRange::new('a', 'm'));
    /// assert_eq!(r1.intersect(&r3), CharRange::singleton('a'));
    /// assert!(r1.intersect(&r4).is_empty());
    /// ```
    pub fn intersect(&self, other: &Self) -> Self {
        let start = self.start.max(other.start);
        let end = self.end.min(other.end);
        CharRange::new(start, end)
    }

    /// Returns the complement of the SMT-LIB alphabet w.r.t. this range.
    /// If this range is `[a, b]`, the complement is a union of ranges containing
    ///
    /// - `[0, a-1]`  if `a > 0`,
    /// - and `[b+1, MAX]` if `b < MAX`.
    ///
    /// Thus, the complement of an empty range is the full alphabet and the complement of the full alphabet is empty.
    ///
    /// # Example
    /// ```
    /// use smt_str::alphabet::CharRange;
    /// use smt_str::SmtChar;
    ///
    /// let range = CharRange::new('a', 'd');
    /// let complement = range.complement();
    /// let mut iter = complement.into_iter();
    /// assert_eq!(iter.next(), Some(CharRange::new(SmtChar::from(0), SmtChar::from('a').saturating_prev())));
    /// assert_eq!(iter.next(), Some(CharRange::new(SmtChar::from('d').saturating_next(), SmtChar::MAX)));
    /// assert_eq!(iter.next(), None);
    ///
    /// assert_eq!(CharRange::empty().complement(), vec![CharRange::all()]);
    /// assert_eq!(CharRange::all().complement(), vec![]);
    /// ```
    pub fn complement(&self) -> Vec<CharRange> {
        if self.is_empty() {
            return vec![CharRange::all()];
        }

        let mut result = Vec::new();
        if self.start > SmtChar::new(0) {
            result.push(Self::new(0u32, self.start.saturating_prev()));
        }
        if self.end < SmtChar::MAX {
            result.push(Self::new(self.end.saturating_next(), SmtChar::MAX));
        }
        result
    }

    /// Subtracts the other range from this ranges.
    /// Returns the difference of two ranges.
    /// The difference of two ranges is the range that contains all characters that are in the first range but not in the second range.
    /// If the two ranges do not overlap, the difference is the first range itself.
    /// If the first range is a subset of the second range, the difference is empty.
    /// If the second range is a subset of the first range, the difference is the two ranges that are not overlapping.
    /// If the two ranges are equal, the difference is empty.
    ///
    /// # Example
    /// ```
    /// use smt_str::alphabet::CharRange;
    /// use smt_str::SmtChar;
    ///
    /// let r1 = CharRange::new('a', 'z');
    /// let r2 = CharRange::new('a', 'm');
    /// let r3 = CharRange::new('m', 'z');
    /// let r4 = CharRange::singleton('c');
    ///
    /// assert_eq!(r1.subtract(&r2), vec![CharRange::new('n', 'z')]);
    /// assert_eq!(r1.subtract(&r3), vec![CharRange::new('a', 'l')]);
    /// assert_eq!(r2.subtract(&r3), vec![CharRange::new('a', 'l')]);
    /// assert_eq!(r1.subtract(&r4), vec![CharRange::new('a', 'b'), CharRange::new('d', 'z')]);
    /// assert_eq!(r2.subtract(&r2), vec![]);
    /// ```
    pub fn subtract(&self, other: &Self) -> Vec<CharRange> {
        if self.is_empty() {
            return vec![];
        } else if other.is_empty() {
            return vec![*self];
        }
        // No overlap, return self
        if self.end < other.start || self.start > other.end {
            return vec![*self];
        }
        let mut result = Vec::new();
        // Left part before `other`
        if self.start < other.start {
            result.push(Self::new(self.start, other.start.saturating_prev()));
        }
        // Right part after `other`
        if self.end > other.end {
            result.push(Self::new(other.end.saturating_next(), self.end));
        }
        result
    }
}

impl PartialOrd for CharRange {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for CharRange {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match self.start.cmp(&other.start) {
            std::cmp::Ordering::Equal => self.end.cmp(&other.end),
            o => o,
        }
    }
}

/// A finite set of characters from the SMT-LIB alphabet, represented as a normalized set of disjoint [`CharRange`]s.
///
/// An `Alphabet` models a subset of the SMT-LIB character domain `[0x0000, 0x2FFFF]`.
/// Internally, it is represented as a sorted list of [`CharRange`]s, where each range is:
/// - non-empty,
/// - non-overlapping with other ranges,
/// - and not adjacent to any other range.
///
/// ## Normalization
///
/// The internal representation is kept normalized at all times:
/// - Ranges are kept in ascending order by start character.
/// - Overlapping or adjacent ranges are merged automatically upon insertion.
/// - The internal structure is stored in a [`Vec`] of [`CharRange`]s.
///
// Keeping this invariant requires `O(n)` time for insertion, where `n` is the number of existing ranges in the alphabet.
///
/// ### Example
///
/// ```rust
/// use smt_str::alphabet::{Alphabet, CharRange};
///
/// let mut a = Alphabet::default();
/// a.insert(CharRange::new('a', 'f'));
///
/// // Overlapping range is merged
/// a.insert(CharRange::new('d', 'h'));
/// assert_eq!(a.iter_ranges().count(), 1);
///
/// // Adjacent range is also merged
/// a.insert(CharRange::new('h', 'k'));
/// assert_eq!(a.iter_ranges().count(), 1);
/// ```
///
/// ## Set Operations
///
/// This type supports the following set operations, each returning a new normalized [`Alphabet`]:
///
/// - [`union`](#method.union): characters in either alphabet.
/// - [`intersect`](#method.intersect): characters common to both alphabets.
/// - [`subtract`](#method.subtract): characters in `self` but not in the other alphabet.
/// - [`complement`](#method.complement): all characters not in the current alphabet (with respect to the SMT-LIB character domain).
///
/// All set operations are performed in O(n + m) time, where n and m are the number of ranges in the two alphabets.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Alphabet {
    ranges: Vec<CharRange>,
}

impl Alphabet {
    /// Returns the full alphabet, containing all characters in the SMT-LIB alphabet.
    pub fn full() -> Self {
        let mut alphabet = Alphabet::default();
        alphabet.insert(CharRange::all());
        alphabet
    }

    /// Returns an empty alphabet.
    pub fn empty() -> Self {
        Alphabet::default()
    }

    /// Check if the alphabet is empty.
    ///
    /// # Example
    /// ```
    /// use smt_str::alphabet::Alphabet;
    /// let alphabet = Alphabet::empty();
    /// assert!(alphabet.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }

    /// The number of characters in the alphabet.
    /// This is the sum of the number of characters in each range in the alphabet.
    /// If the alphabet is empty, the length is zero.
    ///
    /// # Example
    /// ```
    /// use smt_str::alphabet::{Alphabet, CharRange};
    ///
    /// let mut alphabet = Alphabet::default();
    /// alphabet.insert(CharRange::new('a', 'd')); // 4 characters
    /// alphabet.insert(CharRange::new('x', 'z')); // 3 characters
    /// assert_eq!(alphabet.len(), 7);
    /// ```
    pub fn len(&self) -> usize {
        self.ranges.iter().map(|r| r.len()).sum()
    }

    /// Check if a character is in the alphabet.
    /// Requires O(log n) time where n is the number of ranges in the alphabet.
    ///
    /// # Example
    /// ```
    /// use smt_str::alphabet::{Alphabet, CharRange};
    ///
    /// let mut alphabet = Alphabet::default();
    /// alphabet.insert(CharRange::new('a', 'd'));
    /// assert!(alphabet.contains('a'));
    /// assert!(alphabet.contains('d'));
    /// assert!(alphabet.contains('c'));
    /// assert!(!alphabet.contains('e'));
    /// ```
    pub fn contains(&self, c: impl Into<SmtChar>) -> bool {
        let c = c.into();
        self.ranges
            .binary_search_by(|range| {
                if c < range.start() {
                    // c is before the range
                    Ordering::Greater
                } else if c > range.end() {
                    // c is after the range
                    Ordering::Less
                } else {
                    // c is in the range
                    Ordering::Equal
                }
            })
            .is_ok()
    }

    /// Inserts a new character range into the alphabet, merging it with any overlapping or adjacent ranges.
    ///
    /// The internal representation is kept normalized: all ranges are non-overlapping, non-adjacent, and sorted by starting character.
    /// This means that if the inserted range touches or overlaps existing ranges, it will be merged into a single contiguous range.
    ///
    /// Keeping this invariant requires `O(n)` time for insertion, where `n` is the number of existing ranges in the alphabet.
    ///
    /// Inserting an empty range has no effect.
    ///
    ///
    /// # Examples
    ///
    /// ```rust
    /// use smt_str::alphabet::{Alphabet, CharRange};
    ///
    /// let mut a = Alphabet::default();
    /// a.insert(CharRange::new('a', 'c'));
    /// a.insert(CharRange::new('d', 'f')); // Adjacent: merged into ['a','f']
    ///
    /// let ranges: Vec<_> = a.iter_ranges().collect();
    /// assert_eq!(ranges, vec![CharRange::new('a', 'f')]);
    /// ```
    pub fn insert(&mut self, new: CharRange) {
        if new.is_empty() {
            return;
        }

        // Find where `new.start` fits in the sorted list
        let mut i = match self.ranges.binary_search_by(|r| r.start.cmp(&new.start)) {
            Ok(i) => i,
            Err(i) => i,
        };

        let mut start = new.start;
        let mut end = new.end;

        // Merge backwards if the previous range overlaps or is adjacent `new`
        if i > 0 {
            let prev = self.ranges[i - 1];
            if prev.end.saturating_next() >= new.start {
                i -= 1;
                start = start.min(prev.start);
                end = end.max(prev.end);
                self.ranges.remove(i); // remove prev
            }
        }

        // Move forward and remove all overlapping/adjacent ranges
        while i < self.ranges.len() {
            let r = self.ranges[i];
            if r.start > end.saturating_next() {
                break;
            }
            start = start.min(r.start);
            end = end.max(r.end);
            self.ranges.remove(i); // shift left, now `i` points to the next range
        }

        // Insert merged range
        self.ranges.insert(i, CharRange::new(start, end));
    }

    /// Insert a new character in the alphabet.
    /// Equivalent to `insert(CharRange::singleton(c))`.
    /// See [insert](#method.insert) for more details.
    pub fn insert_char(&mut self, c: impl Into<SmtChar>) {
        self.insert(CharRange::singleton(c));
    }

    /* Set operations */

    /// Computes the union of two alphabets and returns the result.
    ///
    /// The union contains all characters that appear in either `self` or `other`.
    ///
    /// This method performs a single linear pass over the ranges in both alphabets to merge overlapping and adjacent ranges.
    /// Requires O(n + m) space and time, where n and m are the number of ranges in the two alphabets.
    ///
    /// The resulting `Alphabet` preserves the invariant that its character ranges are sorted and normalized, i.e., non-overlapping and non-adjacent.
    ///
    ///
    /// # Examples
    /// ```
    /// use smt_str::alphabet::{Alphabet, CharRange};
    ///
    /// let mut a1 = Alphabet::default();
    /// a1.insert(CharRange::new('a', 'c'));
    ///
    /// let mut a2 = Alphabet::default();
    /// a2.insert(CharRange::new('d', 'f'));
    ///
    /// let union = a1.union(&a2);
    ///
    /// let mut iter = union.iter_ranges();
    /// assert_eq!(iter.next(), Some(CharRange::new('a', 'f'))); // merged
    /// assert_eq!(iter.next(), None);
    /// ```
    pub fn union(&self, other: &Self) -> Self {
        let mut merged = Vec::with_capacity(self.ranges.len() + other.ranges.len());
        let mut i = 0;
        let mut j = 0;
        let mut active = CharRange::empty();

        /// Merge helper: adds next range into active or flushes it to the merged vector
        /// If the active range is empty, it is set to the next range.
        /// If the active range overlaps with the next range, it is extended.
        /// Otherwise, the active range is flushed to the merged vector and the next range becomes the new active range.
        fn merge_into(active: &mut CharRange, next: CharRange, merged: &mut Vec<CharRange>) {
            if active.is_empty() {
                *active = next;
            } else if active.end.saturating_next() >= next.start {
                active.end = active.end.max(next.end);
            } else {
                merged.push(*active);
                *active = next;
            }
        }

        // Merge two sorted sequences
        while i < self.ranges.len() && j < other.ranges.len() {
            let r1 = self.ranges[i];
            let r2 = other.ranges[j];
            debug_assert!(!r1.is_empty());
            debug_assert!(!r2.is_empty());

            if r1.start <= r2.start {
                merge_into(&mut active, r1, &mut merged);
                i += 1;
            } else {
                merge_into(&mut active, r2, &mut merged);
                j += 1;
            }
        }

        // Handle the remaining ranges
        while i < self.ranges.len() {
            merge_into(&mut active, self.ranges[i], &mut merged);
            i += 1;
        }
        while j < other.ranges.len() {
            merge_into(&mut active, other.ranges[j], &mut merged);
            j += 1;
        }

        if !active.is_empty() {
            merged.push(active);
        }

        Alphabet { ranges: merged }
    }

    /// Computes the intersection of two alphabets and returns the result.
    ///
    /// The result contains all characters that are present in both `self` and `other`.
    /// This implementation performs a linear-time merge over the sorted range lists
    /// and computes pairwise intersections of overlapping ranges.
    /// Hence, the function needs O(n + m) time and O(min(n,m)) space, where n and m are the number of ranges in the two alphabets.
    ///
    /// The result is guaranteed to be sorted, non-overlapping, and non-adjacent.
    ///
    /// # Example
    /// ```
    /// use smt_str::alphabet::{Alphabet, CharRange};
    /// use smt_str::SmtChar;
    ///
    /// let mut a1 = Alphabet::default();
    /// a1.insert(CharRange::new('a', 'd'));
    /// a1.insert(CharRange::new('x', 'z'));
    ///
    /// let mut a2 = Alphabet::default();
    /// a2.insert(CharRange::new('c', 'f'));
    /// a2.insert(CharRange::new('y', 'z'));
    ///     
    /// let intersection = a1.intersect(&a2);
    ///
    /// let mut iter = intersection.iter_ranges();
    /// assert_eq!(iter.next(), Some(CharRange::new('c', 'd')));
    /// assert_eq!(iter.next(), Some(CharRange::new('y', 'z')));
    /// assert_eq!(iter.next(), None);
    /// ```
    pub fn intersect(&self, other: &Self) -> Self {
        let mut result = Vec::new();
        let mut i = 0;
        let mut j = 0;

        while i < self.ranges.len() && j < other.ranges.len() {
            let r1 = self.ranges[i];
            let r2 = other.ranges[j];

            if r1.end < r2.start {
                // r1 ends before r2 starts
                i += 1;
            } else if r2.end < r1.start {
                // r2 ends before r1 starts
                j += 1;
            } else {
                // Ranges overlap, compute intersection is O(1)
                result.push(r1.intersect(&r2));
                // Move the range that ends first
                if r1.end < r2.end {
                    i += 1;
                } else {
                    j += 1;
                }
            }
        }
        Alphabet { ranges: result }
    }

    /// Computes the complement of the alphabet with respect to the full SMT-LIB character set.
    ///
    /// The result contains all characters that are **not present** in this alphabet.  
    /// That is, if this alphabet contains characters `A`, the complement contains the set `[0x0000, 0x2FFFF] \ A`.
    ///
    /// Computing the complement is done in a single pass over the ranges in the alphabet.
    /// It requires O(n) time and space, where n is the number of ranges in the alphabet.
    ///
    /// The output is a normalized `Alphabet` with non-overlapping, non-adjacent and sorted ranges.
    ///
    /// # Example
    /// ```
    /// use smt_str::alphabet::{Alphabet, CharRange};
    /// use smt_str::SmtChar;
    ///
    /// let mut a = Alphabet::default();
    /// a.insert(CharRange::new('a', 'd'));
    /// a.insert(CharRange::new('x', 'z'));
    ///
    /// let complement = a.complement();
    ///
    /// let mut iter = complement.iter_ranges();
    /// assert_eq!(iter.next(), Some(CharRange::new(SmtChar::from(0), SmtChar::from('a').saturating_prev())));
    /// assert_eq!(iter.next(), Some(CharRange::new(SmtChar::from('d').saturating_next(), SmtChar::from('x').saturating_prev())));
    /// assert_eq!(iter.next(), Some(CharRange::new(SmtChar::from('z').saturating_next(), SmtChar::MAX)));
    /// assert_eq!(iter.next(), None);
    /// ```
    pub fn complement(&self) -> Self {
        let mut result = Alphabet::default();
        let mut last = SmtChar::new(0);
        for r in &self.ranges {
            if last < r.start {
                result.insert(CharRange::new(last, r.start.saturating_prev()));
            }
            last = match r.end.try_next() {
                Some(s) => s,
                None => return result, // We have reached the end of the SMT-LIB alphabet
            }
        }
        if last <= SmtChar::MAX {
            result.insert(CharRange::new(last, SmtChar::MAX));
        }
        result
    }

    /// Computes the set difference `self \ other` and returns a new `Alphabet`.
    ///
    /// The result contains all characters that are present in `self` but **not** in `other`.
    ///
    /// This operation is analogous to set difference and preserves normalization
    /// (sorted, non-overlapping, non-adjacent ranges).
    /// It needs O(n + m)  and O(n + m) space, where n and m are the number of ranges in the two alphabets.
    ///
    /// # Example
    ///  ```
    /// use smt_str::alphabet::{Alphabet, CharRange};
    ///
    /// let mut a = Alphabet::default();
    /// a.insert(CharRange::new('a', 'f'));
    ///
    /// let mut b = Alphabet::default();
    /// b.insert(CharRange::new('c', 'd'));
    ///
    /// let diff = a.subtract(&b);
    /// let expected = [CharRange::new('a', 'b'), CharRange::new('e', 'f')];
    ///
    /// let ranges: Vec<_> = diff.iter_ranges().collect();
    /// assert_eq!(ranges, expected);
    /// ```
    pub fn subtract(&self, other: &Self) -> Self {
        // This could be done with a single pass over the ranges, which would not require allocating an alphabet for the complement.
        // However, the current implementation is simpler and has the same runtime complexity.
        let other_comp = other.complement();
        self.intersect(&other_comp)
    }

    /// Return an iterator over the ranges in the alphabet.
    pub fn iter_ranges(&self) -> impl Iterator<Item = CharRange> + '_ {
        self.ranges.iter().copied()
    }

    /// Return an iterator over all characters in the alphabet.
    pub fn iter(&self) -> impl Iterator<Item = SmtChar> + '_ {
        self.iter_ranges().flat_map(|r| r.iter())
    }
}

impl FromIterator<CharRange> for Alphabet {
    fn from_iter<T: IntoIterator<Item = CharRange>>(iter: T) -> Self {
        let mut alphabet = Alphabet::default();
        for r in iter {
            alphabet.insert(r);
        }
        alphabet
    }
}

impl FromIterator<SmtChar> for Alphabet {
    fn from_iter<T: IntoIterator<Item = SmtChar>>(iter: T) -> Self {
        let mut alphabet = Alphabet::default();
        for c in iter {
            alphabet.insert_char(c);
        }
        alphabet
    }
}

impl Display for CharRange {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_empty() {
            write!(f, "[]")
        } else if self.start == self.end {
            write!(f, "[{}]", self.start)
        } else {
            write!(f, "[{}-{}]", self.start, self.end)
        }
    }
}

impl Display for Alphabet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{{")?;
        for (i, r) in self.iter_ranges().enumerate() {
            write!(f, "{}", r)?;
            if i < self.ranges.len() - 1 {
                write!(f, ", ")?;
            }
        }
        write!(f, "}}")
    }
}

#[cfg(test)]
mod tests {
    use quickcheck::Arbitrary;
    use quickcheck_macros::quickcheck;

    impl Arbitrary for CharRange {
        fn arbitrary(g: &mut quickcheck::Gen) -> Self {
            let start = SmtChar::arbitrary(g);
            let end = SmtChar::arbitrary(g);
            CharRange::new(start.min(end), end.max(start))
        }
    }

    impl Arbitrary for Alphabet {
        fn arbitrary(g: &mut quickcheck::Gen) -> Self {
            let mut alphabet = Alphabet::default();
            let ranges: Vec<CharRange> = Arbitrary::arbitrary(g);
            for r in ranges {
                alphabet.insert(r);
            }
            alphabet
        }
    }

    use crate::{
        alphabet::{Alphabet, CharRange},
        CharIterator, SmtChar,
    };

    #[quickcheck]
    fn all_range_contains_all_chars(c: SmtChar) -> bool {
        let range = super::CharRange::all();
        range.contains(c)
    }

    #[quickcheck]
    fn range_contain_all_between(s: SmtChar, e: SmtChar) {
        let range = super::CharRange::new(s, e);
        let mut len = 0;
        for c in CharIterator::new(s, e) {
            len += 1;
            assert!(range.contains(c));
        }
        assert_eq!(len, range.len());
    }

    #[quickcheck]
    fn range_intersect_self(r: CharRange) -> bool {
        let i = r.intersect(&r);
        i == r
    }

    #[quickcheck]
    fn range_intersect_empty(r: CharRange) -> bool {
        let empty = CharRange::empty();
        let i = r.intersect(&empty);
        i.is_empty()
    }

    #[quickcheck]
    fn range_intersect_all(r: CharRange) -> bool {
        let all = CharRange::all();
        let i = r.intersect(&all);
        i == r
    }

    #[quickcheck]
    fn range_intersect_correct(r1: CharRange, r2: CharRange) {
        let i = r1.intersect(&r2);

        for c in r1.iter() {
            if r2.contains(c) {
                assert!(i.contains(c));
            } else {
                assert!(!i.contains(c));
            }
        }

        for c in r2.iter() {
            if r1.contains(c) {
                assert!(i.contains(c));
            } else {
                assert!(!i.contains(c));
            }
        }

        for c in i.iter() {
            assert!(r1.contains(c));
            assert!(r2.contains(c));
        }
    }

    #[quickcheck]
    fn range_difference_self(r: CharRange) -> bool {
        let d = r.subtract(&r);
        d.is_empty()
    }

    #[quickcheck]
    fn range_difference_empty(r: CharRange) {
        let empty = CharRange::empty();
        let d = r.subtract(&empty);
        assert_eq!(d.len(), 1);
        assert_eq!(d[0], r)
    }

    #[quickcheck]
    fn range_difference_all(r: CharRange) -> bool {
        let all = CharRange::all();
        let d = r.subtract(&all);
        d.is_empty()
    }

    #[quickcheck]
    fn range_difference_correct(r1: CharRange, r2: CharRange) {
        let diff = r1.subtract(&r2);

        for c in r1.iter() {
            if r2.contains(c) {
                assert!(!diff.iter().any(|r| r.contains(c)));
            } else {
                assert!(diff.iter().any(|r| r.contains(c)));
            }
        }

        for c in r2.iter() {
            if r1.contains(c) {
                assert!(!diff.iter().any(|r| r.contains(c)));
            }
        }

        for cr in diff.iter() {
            for c in cr.iter() {
                assert!(r1.contains(c), "{c} is diff {cr} but not in r1 = {r1}");
                assert!(!r2.contains(c));
            }
        }
    }

    /* Alphabet */

    fn alphabet(ranges: &[CharRange]) -> Alphabet {
        let mut a = Alphabet::default();
        for r in ranges {
            a.insert(*r);
        }
        a
    }

    fn ranges_eq(a: &Alphabet, expected: &[CharRange]) {
        let actual: Vec<_> = a.iter_ranges().collect();
        let expected: Vec<_> = expected.to_vec();
        assert_eq!(actual, expected);
    }

    fn range(start: char, end: char) -> CharRange {
        CharRange::new(start, end)
    }

    #[quickcheck]
    fn test_alphabet_insert_empty(r: CharRange) {
        let mut alphabet = Alphabet::default();
        alphabet.insert(r);
        assert_eq!(alphabet.iter_ranges().next(), Some(r));
        assert_eq!(alphabet.iter_ranges().count(), 1);
    }

    #[test]
    fn test_alphabet_insert_non_overlapping() {
        let a = alphabet(&[range('a', 'c'), range('e', 'g')]);
        ranges_eq(&a, &[range('a', 'c'), range('e', 'g')]);
    }

    #[test]
    fn test_alphabet_insert_adjacent_right_merges() {
        let a = alphabet(&[range('a', 'c'), range('d', 'f')]);
        ranges_eq(&a, &[range('a', 'f')]);
    }

    #[test]
    fn test_alphabet_insert_adjacent_left_merges() {
        let a = alphabet(&[range('d', 'f'), range('a', 'c')]);
        ranges_eq(&a, &[range('a', 'f')]);
    }

    #[test]
    fn test_alphabet_insert_overlapping_right_merges() {
        let a = alphabet(&[range('a', 'c'), range('b', 'd')]);
        ranges_eq(&a, &[range('a', 'd')]);
    }

    #[test]
    fn test_alphabet_insert_overlapping_left_merges() {
        let a = alphabet(&[range('b', 'd'), range('a', 'c')]);
        ranges_eq(&a, &[range('a', 'd')]);
    }

    #[test]
    fn test_alphabet_insert_touching_multiple_merges_all() {
        let a = alphabet(&[
            range('a', 'c'),
            range('e', 'g'),
            range('i', 'k'),
            range('d', 'j'),
        ]);
        ranges_eq(&a, &[range('a', 'k')]);
    }

    #[test]
    fn test_alphabet_insert_inside_existing_range_no_change() {
        let mut a = Alphabet::default();
        a.insert(range('a', 'z'));
        a.insert(range('d', 'f'));
        ranges_eq(&a, &[range('a', 'z')]);
    }

    #[test]
    fn test_alphabet_test_alphabet_insert_subsuming_range() {
        let mut a = Alphabet::default();
        a.insert(range('d', 'f'));
        a.insert(range('a', 'z'));
        ranges_eq(&a, &[range('a', 'z')]);
    }

    #[test]
    fn test_alphabet_insert_single_char_merges_with_existing() {
        let a = alphabet(&[range('a', 'c'), CharRange::singleton('d')]);
        ranges_eq(&a, &[range('a', 'd')]);
    }

    #[test]
    fn test_alphabet_insert_disjoint_apart() {
        let a = alphabet(&[range('a', 'b'), range('x', 'z')]);
        ranges_eq(&a, &[range('a', 'b'), range('x', 'z')]);
    }

    #[quickcheck]
    fn alphabet_non_normalized_two_ranges(r1: CharRange, r2: CharRange) {
        let mut alphabet = Alphabet::default();
        alphabet.insert(r1);
        alphabet.insert(r2);

        let mut iter = alphabet.iter_ranges();
        let mut last = iter.next();
        for r in iter {
            // Not overlapping
            let i = last.unwrap().intersect(&r);
            assert!(
                i.is_empty(),
                "{}: Intersection of {} and {} = {}",
                alphabet,
                last.unwrap(),
                r,
                i
            );
            // Not adjacent:
            assert!(last.unwrap().end.saturating_next() < r.start);
            last = Some(r);
        }
    }

    #[quickcheck]
    fn alphabet_non_normalized(ranges: Vec<CharRange>) {
        let mut alphabet = Alphabet::default();
        for r in ranges {
            alphabet.insert(r);
        }

        let mut iter = alphabet.iter_ranges();
        let mut last = iter.next();
        for r in iter {
            // Not overlapping
            let i = last.unwrap().intersect(&r);
            assert!(
                i.is_empty(),
                "{}: Intersection of {} and {} = {}",
                alphabet,
                last.unwrap(),
                r,
                i
            );
            // Not adjacent:
            assert!(last.unwrap().end.saturating_next() < r.start);
            last = Some(r);
        }
    }

    #[test]
    fn test_alphabet_contains() {
        let a = alphabet(&[range('a', 'f'), range('x', 'z')]);
        assert!(a.contains('a'));
        assert!(a.contains('e'));
        assert!(!a.contains('m'));
        assert!(a.contains('x'));
        assert!(!a.contains('w'));
    }

    #[quickcheck]
    fn test_alphabet_contains_2(ranges: Vec<CharRange>) {
        let alphabet = alphabet(&ranges);
        for r in ranges {
            if let Some(c) = r.choose() {
                assert!(alphabet.contains(c));
            }
        }
    }

    /* Set operations */

    #[quickcheck]
    fn alphabet_union_self(a: Alphabet) -> bool {
        let u = a.union(&a);
        u == a
    }

    #[quickcheck]
    fn alphabet_union_empty(a: Alphabet) -> bool {
        let empty = Alphabet::default();
        let u = a.union(&empty);
        u == a
    }

    #[quickcheck]
    fn alphabet_union_all(a: Alphabet) -> bool {
        let all = Alphabet::full();
        let u = a.union(&all);
        u == all
    }

    #[test]
    fn alphabet_union_disjoint() {
        let a1 = alphabet(&[range('a', 'c')]);
        let a2 = alphabet(&[range('x', 'z')]);
        let union = a1.union(&a2);
        ranges_eq(&union, &[range('a', 'c'), range('x', 'z')]);
    }

    #[test]
    fn alphabet_union_overlapping() {
        let a1 = alphabet(&[range('a', 'f')]);
        let a2 = alphabet(&[range('d', 'k')]);
        let union = a1.union(&a2);
        ranges_eq(&union, &[range('a', 'k')]);
    }

    #[test]
    fn alphabet_union_adjacent() {
        let a1 = alphabet(&[range('a', 'c')]);
        let a2 = alphabet(&[range('d', 'f')]);
        let union = a1.union(&a2);
        ranges_eq(&union, &[range('a', 'f')]);
    }

    #[test]
    fn alphabet_union_multiple_merge() {
        let a1 = alphabet(&[range('a', 'c'), range('f', 'h')]);
        let a2 = alphabet(&[range('b', 'g')]);
        let union = a1.union(&a2);
        ranges_eq(&union, &[range('a', 'h')]);
    }

    #[quickcheck]
    fn alphabet_union_correct(a1: Alphabet, a2: Alphabet) -> bool {
        let u = a1.union(&a2);
        for c in u.iter() {
            if a1.contains(c) || a2.contains(c) {
                assert!(u.contains(c));
            } else {
                assert!(!u.contains(c));
            }
        }
        true
    }

    #[quickcheck]
    fn alphabet_intersect_self(a: Alphabet) -> bool {
        let i = a.intersect(&a);
        i == a
    }

    #[quickcheck]
    fn alphabet_intersect_empty(a: Alphabet) -> bool {
        let empty = Alphabet::default();
        let i = a.intersect(&empty);
        i.is_empty()
    }

    #[quickcheck]
    fn alphabet_intersect_all(a: Alphabet) -> bool {
        let all = Alphabet::full();
        let i = a.intersect(&all);
        i == a
    }

    #[test]
    fn intersect_disjoint() {
        let a1 = alphabet(&[range('a', 'c')]);
        let a2 = alphabet(&[range('x', 'z')]);
        let intersection = a1.intersect(&a2);
        assert!(intersection.is_empty());
    }

    #[test]
    fn intersect_overlap() {
        let a1 = alphabet(&[range('a', 'f')]);
        let a2 = alphabet(&[range('d', 'k')]);
        let intersection = a1.intersect(&a2);
        ranges_eq(&intersection, &[range('d', 'f')]);
    }

    #[test]
    fn intersect_adjacent_is_empty() {
        let a1 = alphabet(&[range('a', 'c')]);
        let a2 = alphabet(&[range('d', 'f')]);
        let intersection = a1.intersect(&a2);
        assert!(intersection.is_empty());
    }

    #[test]
    fn intersect_subset() {
        let a1 = alphabet(&[range('a', 'z')]);
        let a2 = alphabet(&[range('g', 'm')]);
        let intersection = a1.intersect(&a2);
        ranges_eq(&intersection, &[range('g', 'm')]);
    }

    #[test]
    fn intersect_multiple_ranges() {
        let a1 = alphabet(&[range('a', 'd'), range('f', 'j')]);
        let a2 = alphabet(&[range('c', 'g')]);
        let intersection = a1.intersect(&a2);
        ranges_eq(&intersection, &[range('c', 'd'), range('f', 'g')]);
    }

    #[quickcheck]
    fn alphabet_intersect_correct(a1: Alphabet, a2: Alphabet) -> bool {
        let i = a1.intersect(&a2);
        for c in i.iter() {
            assert!(a1.contains(c));
            assert!(a2.contains(c));
        }
        for c in a1.iter() {
            if a2.contains(c) {
                assert!(i.contains(c));
            } else {
                assert!(!i.contains(c));
            }
        }
        for c in a2.iter() {
            if a1.contains(c) {
                assert!(i.contains(c));
            } else {
                assert!(!i.contains(c));
            }
        }
        true
    }

    #[quickcheck]
    fn alphabet_complement_self(a: Alphabet) -> bool {
        let c = a.complement();
        let u = a.union(&c);
        u == Alphabet::full()
    }

    #[test]
    fn alphabet_complement_self_full() {
        let a = Alphabet::full();
        let c = a.complement();
        let u = a.union(&c);
        assert_eq!(u, Alphabet::full())
    }

    #[test]
    fn complement_single_range() {
        let a = alphabet(&[range('b', 'd')]);
        let complement = a.complement();

        let expected = vec![
            CharRange::new(SmtChar::from(0), SmtChar::from('b').saturating_prev()),
            CharRange::new(SmtChar::from('d').saturating_next(), SmtChar::MAX),
        ];
        ranges_eq(&complement, &expected);
    }

    #[test]
    fn complement_multiple_ranges() {
        let a = alphabet(&[range('a', 'c'), range('f', 'h'), range('k', 'k')]);
        let complement = a.complement();

        let expected = vec![
            CharRange::new(SmtChar::from(0), SmtChar::from('a').saturating_prev()),
            CharRange::new(
                SmtChar::from('c').saturating_next(),
                SmtChar::from('f').saturating_prev(),
            ),
            CharRange::new(
                SmtChar::from('h').saturating_next(),
                SmtChar::from('k').saturating_prev(),
            ),
            CharRange::new(SmtChar::from('k').saturating_next(), SmtChar::MAX),
        ];

        ranges_eq(&complement, &expected);
    }

    #[test]
    fn complement_range_from_zero() {
        let a = alphabet(&[CharRange::new(SmtChar::from(0), SmtChar::from('x'))]);
        let complement = a.complement();

        let expected = vec![CharRange::new(
            SmtChar::from('x').saturating_next(),
            SmtChar::MAX,
        )];

        ranges_eq(&complement, &expected);
    }

    #[test]
    fn complement_range_to_max() {
        let a = alphabet(&[CharRange::new(SmtChar::from('x'), SmtChar::MAX)]);
        let complement = a.complement();

        let expected = vec![CharRange::new(
            SmtChar::from(0),
            SmtChar::from('x').saturating_prev(),
        )];

        ranges_eq(&complement, &expected);
    }

    #[test]
    fn alphabet_full_complement_empty() {
        let empty = Alphabet::default();
        let full = Alphabet::full();
        assert_eq!(full.complement(), empty);
        assert_eq!(empty.complement(), full);
    }

    #[quickcheck]
    fn alphabet_intersect_comp_self(a: Alphabet) -> bool {
        let i = a.intersect(&a.complement());
        i.is_empty()
    }

    #[quickcheck]
    fn alphabet_union_comp_self(a: Alphabet) {
        let u = a.union(&a.complement());
        assert_eq!(u, Alphabet::full());
    }

    #[test]
    fn alphabet_union_comp_self_but_last() {
        let range = CharRange::new(0u32, SmtChar::MAX.saturating_prev());
        let mut a = Alphabet::default();
        a.insert(range);
        let comp = a.complement();
        let u = a.union(&comp);
        assert_eq!(u, Alphabet::full());
    }

    #[quickcheck]
    fn alphabet_insert_complement(r: CharRange) -> bool {
        let mut alphabet = Alphabet::default();
        alphabet.insert(r);
        let c = alphabet.complement();
        let u = alphabet.union(&c);
        u == Alphabet::full()
    }

    #[test]
    fn subtract_disjoint() {
        let a = alphabet(&[range('a', 'f')]);
        let b = alphabet(&[range('x', 'z')]);
        let diff = a.subtract(&b);
        assert_eq!(diff, a);
    }

    #[test]
    fn subtract_complete_overlap() {
        let a = alphabet(&[range('a', 'f')]);
        let b = alphabet(&[range('a', 'f')]);
        let diff = a.subtract(&b);
        assert!(diff.is_empty());
    }

    #[quickcheck]
    fn subtract_empty_neutral(a: Alphabet) -> bool {
        let empty = Alphabet::default();
        let diff = a.subtract(&empty);
        diff == a
    }

    #[quickcheck]
    fn subtract_all_empty(a: Alphabet) -> bool {
        let all = Alphabet::full();
        let diff = a.subtract(&all);
        diff.is_empty()
    }

    #[test]
    fn subtract_partial_overlap_middle() {
        let a = alphabet(&[range('a', 'f')]);
        let b = alphabet(&[range('c', 'd')]);
        let diff = a.subtract(&b);
        ranges_eq(&diff, &[range('a', 'b'), range('e', 'f')]);
    }

    #[test]
    fn subtract_leading_overlap() {
        let a = alphabet(&[range('a', 'f')]);
        let b = alphabet(&[range('a', 'c')]);
        let diff = a.subtract(&b);
        ranges_eq(&diff, &[range('d', 'f')]);
    }

    #[test]
    fn subtract_trailing_overlap() {
        let a = alphabet(&[range('a', 'f')]);
        let b = alphabet(&[range('d', 'f')]);
        let diff = a.subtract(&b);
        ranges_eq(&diff, &[range('a', 'c')]);
    }

    #[test]
    fn subtract_multiple_ranges() {
        let a = alphabet(&[range('a', 'z')]);
        let b = alphabet(&[range('b', 'd'), range('x', 'z')]);
        let diff = a.subtract(&b);
        ranges_eq(&diff, &[range('a', 'a'), range('e', 'w')]);
    }

    #[test]
    fn subtract_self_is_empty() {
        let a = alphabet(&[range('a', 'z')]);
        assert!(a.subtract(&a).is_empty());
    }

    #[test]
    fn subtract_from_empty_is_empty() {
        let a = Alphabet::default();
        let b = alphabet(&[range('a', 'z')]);
        assert!(a.subtract(&b).is_empty());
    }

    #[quickcheck]
    fn subtract_correct(a: Alphabet, b: Alphabet) -> bool {
        let diff = a.subtract(&b);
        for c in diff.iter() {
            if a.contains(c) && !b.contains(c) {
                assert!(diff.contains(c));
            } else {
                assert!(!diff.contains(c));
            }
        }
        true
    }
}
