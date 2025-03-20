//! An alphabet is finite set of symbols.
//! In the context of SMT-LIB the alphabet are exactly the symbols with Unicode code points between 0 an 0x2FFFF.
//! This module provides a type to represent subsets of the alphabet and utilities to work with them.

use std::{
    collections::{btree_map, BTreeMap, BTreeSet},
    fmt::Display,
};

use crate::{CharIterator, SmtChar};

/// A range of characters [SmChar]s defined by a start and an end character.
/// The range covers all characters between the start and the end character, including the start and the end character themselves.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CharRange {
    start: SmtChar,
    end: SmtChar,
}

impl CharRange {
    /// Create a new range of characters.
    /// The range covers all characters between the start and the end character, including the start and the end character themselves.
    /// If the start character is greater than the end character, the range is empty.
    pub fn new(l: impl Into<SmtChar>, r: impl Into<SmtChar>) -> Self {
        CharRange {
            start: l.into(),
            end: r.into(),
        }
    }

    /// Creates an empty range.
    pub fn empty() -> Self {
        CharRange {
            start: SmtChar::from(1),
            end: SmtChar::from(0),
        }
    }

    /// Create a range that contains a single character.
    pub fn singleton(c: impl Into<SmtChar>) -> Self {
        let c = c.into();
        CharRange { start: c, end: c }
    }

    /// Create a range that covers all characters in the SMT-LIB alphabet.
    ///
    /// # Example
    /// ```
    /// use smtlib_str::alphabet::CharRange;
    /// use smtlib_str::SmtChar;
    ///
    /// let range = CharRange::all();
    /// assert!(range.contains(0));
    /// assert!(range.contains(SmtChar::MAX));
    /// ```
    pub fn all() -> Self {
        CharRange {
            start: SmtChar::from(0),
            end: SmtChar::from(0x2FFFF),
        }
    }

    /// Return the number of characters in the range.
    ///
    /// # Example
    /// ```
    /// use smtlib_str::alphabet::CharRange;
    /// use smtlib_str::SmtChar;
    ///
    /// assert_eq!(CharRange::new('a', 'z').size(), 26);
    /// assert_eq!(CharRange::singleton('a').size(), 1);
    /// assert_eq!(CharRange::new('z', 'a').size(), 0);
    /// ```
    pub fn size(&self) -> usize {
        if self.start > self.end {
            0
        } else {
            ((self.end.0 - self.start.0) + 1) as usize
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = SmtChar> {
        CharIterator::new(self.start, self.end)
    }

    /// Check if the range is empty. The range is empty if the start character is greater than the end character.
    ///
    /// # Example
    /// ```
    /// use smtlib_str::alphabet::CharRange;
    /// use smtlib_str::SmtChar;
    ///
    /// assert!(CharRange::empty().is_empty());
    /// assert!(!CharRange::new('a', 'z').is_empty());
    /// assert!(!CharRange::singleton('a').is_empty());
    /// assert!(CharRange::new('z', 'a').is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.size() == 0
    }

    /// Returns the lower bound of the range.
    pub fn start(&self) -> SmtChar {
        self.start
    }

    /// Returns the upper bound of the range.
    pub fn end(&self) -> SmtChar {
        self.end
    }

    /// Returns a character from the range.
    /// If the range is empty, returns `None`.
    ///
    /// # Example
    /// ```
    /// use smtlib_str::alphabet::CharRange;
    /// use smtlib_str::SmtChar;
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
    /// # Example
    /// ```
    /// use smtlib_str::alphabet::CharRange;
    /// use smtlib_str::SmtChar;
    ///
    /// assert!(CharRange::singleton('a').is_singleton());
    /// assert!(!CharRange::new('a', 'z').is_singleton());
    /// assert!(!CharRange::empty().is_singleton());
    /// ```
    pub fn is_singleton(&self) -> bool {
        self.start == self.end
    }

    /// Check if the range contains all characters in the SMT-LIB alphabet.
    ///
    /// # Example
    /// ```
    /// use smtlib_str::alphabet::CharRange;
    /// use smtlib_str::SmtChar;
    ///
    /// assert!(CharRange::all().is_full());
    /// assert!(CharRange::new(SmtChar::MIN, SmtChar::MAX).is_full());
    /// assert!(!CharRange::empty().is_full());
    /// assert!(!CharRange::new('a', 'z').is_full());
    /// ```
    pub fn is_full(&self) -> bool {
        self.start == SmtChar::from(0) && self.end == SmtChar::MAX
    }

    /// Check if a character is in the range.
    /// Returns true if the character is in the range, false otherwise.
    ///
    /// # Example
    /// ```
    /// use smtlib_str::alphabet::CharRange;
    /// use smtlib_str::SmtChar;
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

    /// Return the intersection of two ranges.
    /// The intersection of two ranges is the range that contains all characters that are in both ranges.
    /// If the two ranges do not overlap, the intersection is empty.
    ///
    /// # Example
    /// ```
    /// use smtlib_str::alphabet::CharRange;
    /// use smtlib_str::SmtChar;
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
    /// use smtlib_str::alphabet::CharRange;
    /// use smtlib_str::SmtChar;
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
        if self.start > SmtChar::from(0) {
            result.push(Self::new(0, self.start.saturating_prev()));
        }
        if self.end < SmtChar::MAX {
            result.push(Self::new(self.end.saturating_next(), SmtChar::MAX));
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

/// A set of characters, represented as [CharRange]s.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Alphabet {
    ranges: BTreeSet<CharRange>,
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
    pub fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }

    /// The number of characters in the alphabet.
    pub fn len(&self) -> usize {
        self.ranges.iter().map(|r| r.size()).sum()
    }

    /// Check if a character is in the alphabet.
    pub fn contains(&self, c: impl Into<SmtChar>) -> bool {
        let c = c.into();
        // Could do binary search to go from O(n) to O(log n) but the number of ranges is expected to be small and contains is O(1) anyway
        self.ranges.iter().any(|r| r.contains(c))
    }

    /// Insert a new range in the alphabet.
    /// Keeps the invariant that the alphabet is a set of non-overlapping, non-adjacent ranges.
    /// If the new range overlaps with an existing ranges (or is adjacent to them), then the new range is merged with the existing ranges.
    ///
    /// # Example
    /// ```
    /// use smtlib_str::alphabet::{Alphabet, CharRange};
    /// use smtlib_str::SmtChar;
    ///
    /// let mut alphabet = Alphabet::default();
    /// alphabet.insert(CharRange::new('a', 'd'));
    /// assert_eq!(alphabet.iter_ranges().next(), Some(CharRange::new('a', 'd')));
    /// alphabet.insert(CharRange::new('e', 'g'));
    /// assert_eq!(alphabet.iter_ranges().next(), Some(CharRange::new('a', 'g')));
    /// alphabet.insert(CharRange::new('c', 'f'));
    /// assert_eq!(alphabet.iter_ranges().next(), Some(CharRange::new('a', 'g')));
    /// alphabet.insert(CharRange::new('x', 'z'));
    /// let mut iter = alphabet.iter_ranges();
    /// assert_eq!(iter.next(), Some(CharRange::new('a', 'g')));
    /// assert_eq!(iter.next(), Some(CharRange::new('x', 'z')));
    /// assert_eq!(iter.next(), None);
    /// ```
    pub fn insert(&mut self, new: CharRange) {
        if new.is_empty() {
            return;
        }

        // we find the first range in the set that overlaps with the new range or is adjacent to it
        let mut iter = self.ranges.iter();
        let mut subsumed = Vec::new();
        let mut start = new.start;
        let mut end = new.end;

        // Find the first range that overlaps or is adjacent to the new range
        for existing in iter.by_ref() {
            if !existing.intersect(&new).is_empty()
                || existing.start.saturating_prev() == new.end
                || existing.end.saturating_next() == new.start
            {
                subsumed.push(*existing);
                start = existing.start.min(new.start);
                end = existing.end.max(new.end);
                break;
            }
            // Stop early if new range is strictly before without adjacency
            if existing.start.saturating_next() > new.end {
                break;
            }
        }

        // Continue searching for additional overlapping or adjacent ranges
        for existing in iter {
            if !existing.intersect(&new).is_empty()
                || existing.start.saturating_prev() == new.end
                || existing.end.saturating_next() == new.start
            {
                end = existing.end.max(new.end);
                subsumed.push(*existing);
            }
            // Stop early if we have passed beyond possible overlaps
            if new.end.saturating_next() < existing.start {
                break;
            }
        }

        for s in subsumed.into_iter() {
            self.ranges.remove(&s);
        }

        self.ranges.insert(CharRange::new(start, end));
    }

    /// Insert a new character in the alphabet.
    /// Equivalent to `insert(CharRange::singleton(c))`.
    /// See [insert](#method.insert) for more details.
    pub fn insert_char(&mut self, c: impl Into<SmtChar>) {
        self.insert(CharRange::singleton(c));
    }

    /* Set operations */

    /// Creates the union of two alphabets.
    /// The union of two alphabets is the alphabet that contains all characters that are in either of the two alphabets.
    ///
    /// # Example
    /// ```
    /// use smtlib_str::alphabet::{Alphabet, CharRange};
    /// use smtlib_str::SmtChar;
    ///
    /// let mut a1 = Alphabet::default();
    /// a1.insert(CharRange::new('a', 'd'));
    ///
    /// let mut a2 = Alphabet::default();
    /// a2.insert(CharRange::new('x', 'z'));
    ///
    /// let union = a1.union(&a2);
    ///
    /// let mut iter = union.iter_ranges();
    /// assert_eq!(iter.next(), Some(CharRange::new('a', 'd')));
    /// assert_eq!(iter.next(), Some(CharRange::new('x', 'z')));
    /// assert_eq!(iter.next(), None);
    /// ```
    pub fn union(&self, other: &Self) -> Self {
        let mut result = self.clone();
        for r in &other.ranges {
            result.insert(*r);
        }
        result
    }

    /// Creates the intersection of two alphabets.
    /// The intersection of two alphabets is the alphabet that contains all characters, and only those characters, that are in both alphabets.
    ///
    /// # Example
    /// ```
    /// use smtlib_str::alphabet::{Alphabet, CharRange};
    /// use smtlib_str::SmtChar;
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
        let mut result = Alphabet::default();

        // Insert all ranges that are in both alphabets
        // We could be more efficient by using that the ranges are sorted but range intersection is O(1) so the actual performance gain is small
        for r1 in &self.ranges {
            for r2 in &other.ranges {
                let i = r1.intersect(r2);
                result.insert(i);
            }
        }
        result
    }

    /// Creates the complement of the alphabet (w.r.t. to the SMt-LIB alphabet).
    /// The complement of an alphabet is the alphabet that contains all characters that are not in the original alphabet.
    ///
    /// # Example
    /// ```
    /// use smtlib_str::alphabet::{Alphabet, CharRange};
    /// use smtlib_str::SmtChar;
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
        let mut last = SmtChar::from(0);
        for r in &self.ranges {
            if last < r.start {
                result.insert(CharRange::new(last, r.start.saturating_prev()));
            }
            last = r.end.saturating_next();
        }
        if last < SmtChar::MAX {
            result.insert(CharRange::new(last, SmtChar::MAX));
        }
        result
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

/// Represents a partitioning of an alphabet into non-overlapping partitions, each represented by a [CharRange].
/// In difference to [Alphabet], the partitioning does not enforce that the partitions are non-adjacent.
#[derive(Clone, Default, Debug)]
pub struct AlphabetPartition {
    map: AlphabetPartitionMap<()>,
}

impl AlphabetPartition {
    /// Creates an empty partitioning.
    pub fn empty() -> Self {
        Self {
            map: AlphabetPartitionMap::empty(),
        }
    }

    /// Creates a partitioning with a single range.
    ///
    /// ```
    /// use smtlib_str::alphabet::{AlphabetPartition, CharRange};
    ///
    /// let range = CharRange::new('a', 'z');
    /// let partitioning = AlphabetPartition::singleton(range.clone());
    /// assert_eq!(partitioning.len(), 1);
    /// assert!(partitioning.contains(&range));
    /// ```
    pub fn singleton(r: CharRange) -> Self {
        let map = AlphabetPartitionMap::singleton(r, ());
        Self { map }
    }

    /// Inserts the given character range into the partitioning.
    /// Checks if the range can be inserted into the partitioning without overlapping with existing partitions.
    /// If the range can be inserted, it is inserted and `Ok(())` is returned.
    /// If the range overlaps with an existing partition `r``, the range is returned in `Err(r)`.
    /// This needs O(n) time, where n is the number of partitions in the partitioning.
    /// If the range is known to not overlap with any existing partition, `insert_unchecked` can be used, which is faster.
    ///
    /// # Arguments
    ///
    /// * `range` - The character range to insert.
    ///
    /// # Examples
    ///
    /// ```
    /// use smtlib_str::alphabet::{AlphabetPartition, CharRange};
    ///
    /// let mut partitioning = AlphabetPartition::empty();
    ///
    /// // Insert a non-overlapping range
    /// let range = CharRange::new('a', 'z');
    /// assert_eq!(partitioning.insert(range.clone()), Ok(()));
    /// assert!(partitioning.contains(&range));
    ///
    /// // Insert an overlapping range
    /// assert_eq!(partitioning.insert(CharRange::new('m', 'p')), Err(CharRange::new('a', 'z')));
    /// ```
    pub fn insert(&mut self, range: CharRange) -> Result<(), CharRange> {
        self.map.insert(range, ())
    }

    /// Inserts the given character range into the partitioning, without checking if the partitioning is still valid.
    /// Takes O(log n) time, where n is the number of partitions in the partitioning.
    ///
    /// This method must be used with caution, as it can lead to an invalid partitioning if the range overlaps with an existing partition.
    ///
    /// # Arguments
    ///
    /// * `range` - The character range to insert.
    /// * `v` - The value associated with the character range.
    ///
    /// # Examples
    ///
    /// ```
    /// use smtlib_str::alphabet::{AlphabetPartition, CharRange};
    ///
    /// let mut partitioning = AlphabetPartition::empty();
    /// partitioning.insert_unchecked(CharRange::new('a','z'));
    /// assert!(partitioning.contains(&CharRange::new('a','z')));
    ///
    /// // This will lead to an invalid partitioning
    /// partitioning.insert_unchecked(CharRange::new('m','p'));
    /// assert!(partitioning.contains(&CharRange::new('m','p')));
    /// ```
    pub fn insert_unchecked(&mut self, range: CharRange) {
        self.map.insert_unchecked(range, ());
    }

    /// Returns whether the partitioning contains the given character range.
    /// Does not check for subranges.
    ///
    /// # Arguments
    /// - `range` - The character range to check for.
    ///
    /// # Examples
    ///
    /// ```
    /// use smtlib_str::alphabet::{AlphabetPartition, CharRange};
    ///
    /// let range = CharRange::new('a', 'z');
    /// let mut partitioning = AlphabetPartition::empty();
    /// partitioning.insert_unchecked(range.clone());
    /// assert!(partitioning.contains(&range));
    /// // subranges are not contained
    /// assert!(!partitioning.contains(&CharRange::new('a', 'y')));
    /// ```
    pub fn contains(&self, range: &CharRange) -> bool {
        self.map.get(range).is_some()
    }

    /// Returns the number of partitions in the partitioning.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Returns true if the partitioning is empty.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Removes the given character range from the partitioning.
    /// Only works if this exact range is in the partitioning.
    /// Returns true if the range was removed, false otherwise.
    ///
    /// # Arguments
    /// - `range` - The character range to remove.
    ///
    /// # Examples
    ///
    /// ```
    /// use smtlib_str::alphabet::{AlphabetPartition, CharRange};
    /// let mut partitioning = AlphabetPartition::empty();
    /// partitioning.insert_unchecked(CharRange::new('a','z'));
    ///
    /// assert!(partitioning.contains(&CharRange::new('a','z')));
    /// // We cannot remove "subranges"
    /// assert!(!partitioning.remove(CharRange::new('a','m')));
    ///
    /// assert!(partitioning.remove(CharRange::new('a','z')));
    /// assert!(!partitioning.contains(&CharRange::new('a','z')));
    pub fn remove(&mut self, range: CharRange) -> bool {
        self.map.remove(range).is_some()
    }

    /// Performs a partition refinement of this partitioning with the given partitioning.
    /// Let $P$ be this partitioning, $Q$  be the other partitioning.
    /// Then, for all partitions $p$ in $P$ or $Q$, there are ranges 'r1', ..., 'rn' in the refined partitioning such that:
    /// - $p = r_1 \cup ... \cup r_n$
    /// - $r_i \leq r_{i+1}$ for all $1 \leq i \le n$
    /// - For all $r_j$ with  $r_i \leq r_j \leq r_{i+1}$ either $ri = rj$ or $r_{i+1} = rj$
    ///
    /// # Arguments
    ///
    /// * `other` - The partitioning to refine with.
    ///
    /// # Examples
    ///
    /// ```
    /// use smtlib_str::alphabet::{AlphabetPartition, CharRange};
    ///
    /// let mut partitioning1 = AlphabetPartition::empty();
    /// partitioning1.insert_unchecked(CharRange::new('a', 'z'));
    ///
    /// let mut partitioning2 = AlphabetPartition::empty();
    /// partitioning2.insert_unchecked(CharRange::new('b', 'c'));
    ///
    /// let refined_partitioning = partitioning1.refine(&partitioning2);
    /// let mut iter = refined_partitioning.iter();
    /// assert_eq!(iter.next(), Some(&CharRange::new('a', 'a')));
    /// assert_eq!(iter.next(), Some(&CharRange::new('b', 'c')));
    /// assert_eq!(iter.next(), Some(&CharRange::new('d', 'z')));
    /// ```
    pub fn refine(&self, other: &Self) -> Self {
        let map = self.map.refine(&other.map, |_, _| ());
        Self { map }
    }

    /// Returns an iterator over the partitions in the partitioning.
    pub fn iter(&self) -> impl Iterator<Item = &CharRange> + '_ {
        self.map.iter().map(|(r, _)| r)
    }

    /// Returns an iterator over the partitions in the partitioning with a mutable reference to the values.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &CharRange> + '_ {
        self.map.iter_mut().map(|(r, _)| r)
    }
}

impl Display for AlphabetPartition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{{")?;
        for (i, r) in self.iter().enumerate() {
            write!(f, "{}", r)?;
            if i < self.len() - 1 {
                write!(f, ", ")?;
            }
        }
        write!(f, "}}")
    }
}

/// Represents a partitioning of an alphabet into non-overlapping partitions, each represented by a [CharRange].
/// In difference to [Alphabet], the partitioning does not enforce that the partitions are non-adjacent.
/// Additionally, each partition is associated with a value of type `T`.
#[derive(Clone, Default, Debug)]
pub struct AlphabetPartitionMap<T: Clone> {
    /// The character ranges in the partitioning and the associated values.
    /// The partitions are ordered in a BTreeMap by the start and end of the character range.
    parts: BTreeMap<CharRange, T>,
}

impl<T: Clone> AlphabetPartitionMap<T> {
    /// Creates an empty map.
    pub fn empty() -> Self {
        Self {
            parts: BTreeMap::new(),
        }
    }

    /// Creates a map  with a single range.
    /// The range is associated with the given value.
    ///
    /// ```
    /// use smtlib_str::alphabet::{AlphabetPartitionMap, CharRange};
    ///
    /// let range = CharRange::new('a', 'z');
    /// let partitioning = AlphabetPartitionMap::singleton(range.clone(), 1);
    /// assert_eq!(partitioning.len(), 1);
    /// assert_eq!(partitioning.get(&range), Some(&1));
    /// ```
    pub fn singleton(r: CharRange, v: T) -> Self {
        let parts = vec![(r, v)].into_iter().collect();
        Self { parts }
    }

    /// Inserts the given character range and the associated value into the partitioning.
    /// Checks if the range can be inserted into the partitioning without overlapping with existing partitions.
    /// If the range can be inserted, it is inserted and `Ok(())` is returned.
    /// If the range overlaps with an existing partition `r``, the range is returned in `Err(r)`.
    /// This needs O(n) time, where n is the number of partitions in the partitioning.
    /// If the range is known to not overlap with any existing partition, `insert_unchecked` can be used, which is faster.
    ///
    /// # Examples
    ///
    /// ```
    /// use smtlib_str::alphabet::{AlphabetPartitionMap, CharRange};
    ///
    /// let mut partitioning = AlphabetPartitionMap::empty();
    ///
    /// let range = CharRange::new('a', 'z');
    /// assert_eq!(partitioning.insert(range.clone(), 1), Ok(()));
    /// assert_eq!(partitioning.get(&range), Some(&1));
    ///
    /// // Insert an overlapping range
    /// assert_eq!(partitioning.insert(CharRange::new('m', 'p'), 1), Err(CharRange::new('a', 'z')));
    /// ```
    pub fn insert(&mut self, range: CharRange, v: T) -> Result<(), CharRange> {
        match self.overlaps(range) {
            Some((r, _)) => Err(*r),
            None => {
                self.insert_unchecked(range, v);
                Ok(())
            }
        }
    }

    /// Inserts the given character range and its associated value into the partitioning, without checking for overlaps with existing ranges.
    /// Takes O(log n) time, where n is the number of partitions in the partitioning.
    ///
    /// This method can lead to an invalid partitioning if the range overlaps with an existing partition.
    ///
    /// # Examples
    ///
    /// ```
    /// use smtlib_str::alphabet::{AlphabetPartitionMap, CharRange};
    ///
    /// let mut partitioning = AlphabetPartitionMap::empty();
    /// partitioning.insert_unchecked(CharRange::new('a','z'), 0);
    /// assert_eq!(partitioning.get(&CharRange::new('a','z')), Some(&0));
    ///
    /// // This will lead to an invalid partitioning
    /// partitioning.insert_unchecked(CharRange::new('m','p'), 1);
    /// assert_eq!(partitioning.get(&CharRange::new('m','p')), Some(&1));
    /// ```
    pub fn insert_unchecked(&mut self, range: CharRange, v: T) {
        self.parts.insert(range, v);
    }

    /// Returns the value associated with the given character range, if it exists. Returns `None` otherwise.
    pub fn get(&self, range: &CharRange) -> Option<&T> {
        self.parts.get(range)
    }

    /// Removes the given character range from the partitioning.
    /// Only works if this exact range is in the partitioning.
    ///
    /// # Arguments
    /// - `range` - The character range to remove.
    ///
    /// # Examples
    ///
    /// ```
    /// use smtlib_str::alphabet::{AlphabetPartitionMap, CharRange};
    /// let mut partitioning = AlphabetPartitionMap::empty();
    /// partitioning.insert_unchecked(CharRange::new('a','z'), 0);
    /// assert_eq!(partitioning.get(&CharRange::new('a','z')), Some(&0));
    ///
    /// // We cannot remove "subranges"
    /// assert_eq!(partitioning.remove(CharRange::new('a','m')), None);
    ///
    /// assert_eq!(partitioning.remove(CharRange::new('a','z')), Some(0));
    /// assert_eq!(partitioning.get(&CharRange::new('a','z')), None);
    pub fn remove(&mut self, range: CharRange) -> Option<T> {
        self.parts.remove(&range)
    }

    /// Returns the number of partitions in the partitioning.
    pub fn len(&self) -> usize {
        self.parts.len()
    }

    /// Returns true if the partitioning is empty.
    pub fn is_empty(&self) -> bool {
        self.parts.is_empty()
    }

    /// Performs a partition refinement of this partitioning with the given partitioning.
    /// Let $P$ be this partitioning, $Q$  be the other partitioning.
    /// Then, for all partitions $p$ in $P$ or $Q$, there are ranges 'r1', ..., 'rn' in the refined partitioning such that:
    ///
    /// - $p = r_1 \cup ... \cup r_n$
    /// - $r_i \leq r_{i+1}$ for all $1 \leq i \le n$
    /// - For all $r_j$ with  $r_i \leq r_j \leq r_{i+1}$ either $ri = rj$ or $r_{i+1} = rj$
    ///
    /// ## Handling of values
    /// Let $(r, v)$ be a key-value pair in the refined partitioning, then
    ///
    /// - if $r$ overlaps with a range $r_p$ in $P$ but not no range in $Q$, then $v = P(r_p)$
    /// - if $r$ overlaps with a range $r_q$ in $Q$ but with no range in $P$, then $v = Q(r_q)$
    /// - if $r$ overlaps with a range $r_p$ in $P$ and a range $r_q$ in $Q$, then $v = f(P(r_p), Q(r_q))$
    ///
    ///
    /// where $P(r_p)$ and $Q(r_q)$ are the values associated with the ranges $r_p$ and $r_q$ in the partitioning $P$ and $Q$, respectively, and $f$ is a function that refines the values. This function is passed as an argument to the method.
    /// In other words, the function f is called whenever there is a non-empty overlap between two ranges in the partitions.
    ///
    /// # Arguments
    ///
    /// * `other` - The partitioning to refine with.
    /// * `f` - A function that refines the values of overlapping partitions. See the description above for more details.
    ///
    /// # Examples
    ///
    /// ```
    /// use smtlib_str::alphabet::{AlphabetPartitionMap, CharRange};
    ///
    /// let mut partitioning1 = AlphabetPartitionMap::empty();
    /// partitioning1.insert_unchecked(CharRange::new('a', 'z'), 1);
    ///
    /// let mut partitioning2 = AlphabetPartitionMap::empty();
    /// partitioning2.insert_unchecked(CharRange::new('b', 'c'), 2);
    ///
    /// let refined_partitioning = partitioning1.refine(&partitioning2, |v1, v2| v1 + v2);
    /// let mut iter = refined_partitioning.iter();
    /// assert_eq!(iter.next(), Some((&CharRange::new('a', 'a'), &1)));
    /// assert_eq!(iter.next(), Some((&CharRange::new('b', 'c'), &3)));
    /// assert_eq!(iter.next(), Some((&CharRange::new('d', 'z'), &1)));
    /// ```
    #[allow(clippy::comparison_chain)]
    pub fn refine<F>(&self, other: &Self, f: F) -> Self
    where
        F: Fn(&T, &T) -> T,
    {
        debug_assert!(
            self.valid(),
            "invalid partitioning: {}",
            self.parts
                .keys()
                .map(|k| k.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );
        debug_assert!(other.valid());
        let mut refined = Self::empty();
        let mut left_iter = self.parts.iter();
        let mut right_iter = other.parts.iter();

        let mut left = left_iter.next().map(|(l, value)| (l.start, l.end, value));
        let mut right = right_iter.next().map(|(r, value)| (r.start, r.end, value));

        while let (Some(l), Some(r)) = (left, right) {
            let (l_start, l_end, val_l) = l;
            let (r_start, r_end, val_r) = r;
            if l_end < r_start {
                // No overlap, left is before right
                refined.insert_unchecked(CharRange::new(l_start, l_end), val_l.clone());
                // Advance left
                left = left_iter.next().map(|(l, v)| (l.start, l.end, v));
            } else if r_end < l_start {
                // No overlap, right is before left
                refined.insert_unchecked(CharRange::new(r_start, r_end), val_r.clone());
                // Advance right
                right = right_iter.next().map(|(r, v)| (r.start, r.end, v))
            } else {
                // Overlapping ranges
                if l_start < r_start {
                    // (l_start < r_start < l_end < r_end) or (l_start < r_start < r_end < l_end)
                    // Add [l_start, r_start-1], set left to [r_start, l_end]
                    let prefix = CharRange::new(l_start, r_start.saturating_prev());
                    refined.insert_unchecked(prefix, val_l.clone());
                    left = Some((r_start, l_end, val_l));
                } else if r_start < l_start {
                    // (r_start < l_start < r_end < l_end) or (r_start < l_start < l_end < r_end)
                    // Add [r_start, l_start-1], set right to [l_start, r_end]
                    let prefix = CharRange::new(r_start, l_start.saturating_prev());
                    refined.insert_unchecked(prefix, val_r.clone());
                    right = Some((l_start, r_end, val_r));
                } else {
                    // l_start == r_start, one is a prefix of the other
                    let refined_v = f(val_l, val_r);
                    if l_end < r_end {
                        // [l_start, l_end] is a prefix of [r_start, r_end]
                        // Add [l_start, l_end] to the refined partitioning, advance left, and set right to [l_end+1, r_end]
                        let prefix = CharRange::new(l_start, l_end);
                        refined.insert_unchecked(prefix, refined_v);
                        left = left_iter.next().map(|(l, v)| (l.start, l.end, v));
                        right = Some((l_end.saturating_next(), r_end, val_r));
                    } else if r_end < l_end {
                        // [r_start, r_end] is a prefix of [l_start, l_end]
                        // Add [r_start, r_end] to the refined partitioning, advance right, and set left to [r_end+1, l_end]
                        let prefix = CharRange::new(r_start, r_end);
                        refined.insert_unchecked(prefix, refined_v);
                        right = right_iter.next().map(|(r, v)| (r.start, r.end, v));
                        left = Some((r_end.saturating_next(), l_end, val_l));
                    } else {
                        // l_start == r_start && l_end == r_end
                        // Add [l_start, l_end] to the refined partitioning, advance both
                        refined.insert_unchecked(CharRange::new(l_start, l_end), refined_v);
                        left = left_iter.next().map(|(l, v)| (l.start, l.end, v));
                        right = right_iter.next().map(|(r, v)| (r.start, r.end, v))
                    }
                }
            }
        }

        // Add remaining partitions
        while let Some((start, end, v)) = left {
            debug_assert!(right.is_none());
            refined.insert_unchecked(CharRange::new(start, end), v.clone());
            left = left_iter.next().map(|(l, v)| (l.start, l.end, v));
        }
        while let Some((start, end, v)) = right {
            debug_assert!(left.is_none());
            refined.insert_unchecked(CharRange::new(start, end), v.clone());
            right = right_iter.next().map(|(r, v)| (r.start, r.end, v))
        }
        refined
    }

    /// Refines the partitioning with a single partition.
    /// This is a convenience method that creates a single-partition partitioning and refines the current partitioning with it.
    pub fn refine_single<F>(&self, rn: CharRange, val: T, f: F) -> Self
    where
        F: Fn(&T, &T) -> T,
    {
        let temp_part = AlphabetPartitionMap::singleton(rn, val);
        self.refine(&temp_part, f)
    }

    /// Returns an iterator over the partitions in the partitioning.
    pub fn iter(&self) -> impl Iterator<Item = (&CharRange, &T)> + '_ {
        self.parts.iter()
    }

    /// Returns an iterator over the partitions in the partitioning with a mutable reference to the values.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&CharRange, &mut T)> + '_ {
        self.parts.iter_mut()
    }

    /// Checks if the given character range overlaps with any partition in the partitioning.
    /// Returns the (first) overlapping partition, if it exists. Returns `None` otherwise.
    /// This needs O(n) time, where n is the number of partitions in the partitioning.
    /// Could be improved by using a binary search.
    fn overlaps(&self, range: CharRange) -> Option<(&CharRange, &T)> {
        self.parts
            .iter()
            .find(|(r, _)| !r.intersect(&range).is_empty())
    }

    /// Returns true if the partitioning is valid.
    /// That is, if no two partitions overlap.
    /// This needs O(n) time, where n is the number of partitions in the partitioning.
    fn valid(&self) -> bool {
        self.parts
            .keys()
            .zip(self.parts.keys().skip(1))
            .all(|(r1, r2)| r1.end < r2.start)
    }
}

impl<T: Clone> IntoIterator for AlphabetPartitionMap<T> {
    type Item = (CharRange, T);

    type IntoIter = btree_map::IntoIter<CharRange, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.parts.into_iter()
    }
}

impl<T: Display + Clone> Display for AlphabetPartitionMap<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{{")?;
        for (i, (r, v)) in self.iter().enumerate() {
            write!(f, "{}:{}", r, v)?;
            if i < self.len() - 1 {
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
        alphabet::{Alphabet, AlphabetPartitionMap, CharRange},
        CharIterator, SmtChar,
    };

    use super::AlphabetPartition;

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
        assert_eq!(len, range.size());
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

    /* Alphabet */

    #[quickcheck]
    fn test_alphabet_insert_empty(r: CharRange) {
        let mut alphabet = Alphabet::default();
        alphabet.insert(r);
        assert_eq!(alphabet.iter_ranges().next(), Some(r));
        assert_eq!(alphabet.iter_ranges().count(), 1);
    }

    #[test]
    fn test_alphabet_insert_non_overlapping() {
        let mut alphabet = Alphabet::default();
        let r1 = CharRange::new('a', 'c');
        let r2 = CharRange::new('x', 'z');
        alphabet.insert(r1);
        assert_eq!(alphabet.iter_ranges().next(), Some(r1));

        alphabet.insert(r2);
        let mut iter = alphabet.iter_ranges();
        assert_eq!(iter.next(), Some(r1));
        assert_eq!(iter.next(), Some(r2));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_alphabet_insert_overlapping_left() {
        let mut alphabet = Alphabet::default();
        let r1 = CharRange::new('a', 'e');
        let r2 = CharRange::new('c', 'z');

        alphabet.insert(r1);
        alphabet.insert(r2);

        let expected = CharRange::new('a', 'z');
        let mut iter = alphabet.iter_ranges();
        assert_eq!(iter.next(), Some(expected),);
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_alphabet_insert_overlapping_right() {
        let mut alphabet = Alphabet::default();

        let r1 = CharRange::new('a', 'g');
        let r2 = CharRange::new('c', 'z');
        alphabet.insert(r2);
        alphabet.insert(r1);

        let mut iter = alphabet.iter_ranges();
        assert_eq!(iter.next(), Some(CharRange::new('a', 'z')), "{}", alphabet);
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_alphabet_insert_left_adjacent() {
        let mut alphabet = Alphabet::default();
        alphabet.insert(CharRange::new('a', 'c'));
        alphabet.insert(CharRange::new('d', 'z'));

        let expected = CharRange::new('a', 'z');
        let mut iter = alphabet.iter_ranges();
        assert_eq!(
            iter.next(),
            Some(expected),
            "Expected {} but is {}",
            expected,
            alphabet
        );
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_alphabet_insert_right_adjacent() {
        let mut alphabet = Alphabet::default();
        alphabet.insert(CharRange::new('d', 'z'));
        alphabet.insert(CharRange::new('a', 'c'));

        let expected = CharRange::new('a', 'z');
        let mut iter = alphabet.iter_ranges();
        assert_eq!(
            iter.next(),
            Some(expected),
            "Expected {} but is {}",
            expected,
            alphabet
        );
        assert_eq!(iter.next(), None);
    }

    #[quickcheck]
    fn alphabet_non_overlapping_two(r1: CharRange, r2: CharRange) {
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
    fn alphabet_non_overlapping_or_adjacent(ranges: Vec<CharRange>) {
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
        let mut a = Alphabet::empty();
        a.insert(CharRange::new(0, SmtChar(196606)));
        let c = a.complement();
        let u = a.union(&c);
        assert_eq!(u, Alphabet::full())
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
    fn alphabet_union_comp_self(a: Alphabet) -> bool {
        let u = a.union(&a.complement());
        u == Alphabet::full()
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
    fn refine_al_subsumed() {
        let r1 = CharRange::new(2, 5);
        let r2 = CharRange::new(3, 6);
        let r3 = CharRange::new(1, 4);
        let mut part = AlphabetPartition::empty();
        part = part.refine(&AlphabetPartition::singleton(r1.clone()));

        part = part.refine(&AlphabetPartition::singleton(r2.clone()));

        part = part.refine(&AlphabetPartition::singleton(r3.clone()));

        let mut iter = part.iter();
        assert_eq!(iter.next(), Some(&CharRange::new(1, 1)));
        assert_eq!(iter.next(), Some(&CharRange::new(2, 2)));
        assert_eq!(iter.next(), Some(&CharRange::new(3, 4)));
        assert_eq!(iter.next(), Some(&CharRange::new(5, 5)));
        assert_eq!(iter.next(), Some(&CharRange::new(6, 6)));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_empty_partitioning() {
        let partitioning: AlphabetPartitionMap<i32> = AlphabetPartitionMap::empty();
        assert!(partitioning.parts.is_empty());
    }

    #[test]
    fn test_singleton_partitioning() {
        let range = CharRange::new('a', 'z');
        let partitioning = AlphabetPartitionMap::singleton(range.clone(), 1);
        assert_eq!(partitioning.get(&range), Some(&1));
    }

    #[test]
    fn test_insert_non_overlapping() {
        let mut partitioning = AlphabetPartitionMap::empty();

        // Insert non-overlapping ranges
        let range1 = CharRange::new('a', 'f');
        let range2 = CharRange::new('g', 'z');

        assert_eq!(partitioning.insert(range1.clone(), 1), Ok(()));
        assert_eq!(partitioning.insert(range2.clone(), 2), Ok(()));
        assert_eq!(partitioning.get(&range1), Some(&1));
        assert_eq!(partitioning.get(&range2), Some(&2));
    }

    #[test]
    fn test_insert_overlapping() {
        let mut partitioning = AlphabetPartitionMap::empty();

        // Insert initial range
        let range1 = CharRange::new('a', 'm');
        let overlapping_range = CharRange::new('g', 'z');

        assert_eq!(partitioning.insert(range1.clone(), 1), Ok(()));
        println!("{}", partitioning);
        // Insert overlapping range, expect an error
        assert_eq!(
            partitioning.insert(overlapping_range.clone(), 2),
            Err(range1)
        );
    }

    #[test]
    fn test_remove_partition() {
        let mut partitioning = AlphabetPartitionMap::empty();

        let range = CharRange::new('a', 'z');
        partitioning.insert_unchecked(range.clone(), 1);
        assert_eq!(partitioning.get(&range), Some(&1));

        // Now remove the range and check if it's gone
        partitioning.remove(range.clone());
        assert_eq!(partitioning.get(&range), None);
    }

    #[test]
    fn test_refine_fully_overlapping() {
        let mut partitioning1 = AlphabetPartitionMap::empty();
        partitioning1.insert_unchecked(CharRange::new('a', 'z'), 1);

        let mut partitioning2 = AlphabetPartitionMap::empty();
        partitioning2.insert_unchecked(CharRange::new('a', 'z'), 2);

        // Fully overlapping, so the function should combine the values (1 + 2).
        let refined_partitioning = partitioning1.refine(&partitioning2, |v1, v2| v1 + v2);

        let mut iter = refined_partitioning.iter();
        assert_eq!(iter.next(), Some((&CharRange::new('a', 'z'), &3))); // 1 + 2 = 3
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_refine_partial_overlap() {
        let mut partitioning1 = AlphabetPartitionMap::empty();
        partitioning1.insert_unchecked(CharRange::new('a', 'm'), 1);

        let mut partitioning2 = AlphabetPartitionMap::empty();
        partitioning2.insert_unchecked(CharRange::new('g', 'z'), 2);

        // Partial overlap: 'g' to 'm' is overlapping, other parts are non-overlapping.
        let refined_partitioning = partitioning1.refine(&partitioning2, |v1, v2| v1 + v2);

        let mut iter = refined_partitioning.iter();
        assert_eq!(iter.next(), Some((&CharRange::new('a', 'f'), &1))); // non-overlapping from partitioning1
        assert_eq!(iter.next(), Some((&CharRange::new('g', 'm'), &3))); // overlapping part (1 + 2)
        assert_eq!(iter.next(), Some((&CharRange::new('n', 'z'), &2))); // non-overlapping from partitioning2
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_refine_complex_overlaps() {
        let mut part1 = AlphabetPartitionMap::empty();
        part1.insert_unchecked(CharRange::new('a', 'e'), 1);
        part1.insert_unchecked(CharRange::new('f', 'j'), 3);

        let mut part2 = AlphabetPartitionMap::empty();
        part2.insert_unchecked(CharRange::new('d', 'g'), 2);
        part2.insert_unchecked(CharRange::new('h', 'k'), 4);

        // Overlapping in multiple segments, combining values accordingly.
        let refined_partitioning = part1.refine(&part2, |v1, v2| v1 * v2);

        let mut iter = refined_partitioning.iter();
        assert_eq!(iter.next(), Some((&CharRange::new('a', 'c'), &1))); // non-overlapping part from partitioning1
        assert_eq!(iter.next(), Some((&CharRange::new('d', 'e'), &2))); // overlap: 1 * 2 = 2
        assert_eq!(iter.next(), Some((&CharRange::new('f', 'g'), &6))); // overlap: 3 * 2 = 6
        assert_eq!(iter.next(), Some((&CharRange::new('h', 'j'), &12))); // overlap: 3 * 4 = 12
        assert_eq!(iter.next(), Some((&CharRange::new('k', 'k'), &4))); // non-overlapping part from partitioning2
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_refine_adjacent_ranges() {
        let mut partitioning1 = AlphabetPartitionMap::empty();
        partitioning1.insert_unchecked(CharRange::new('a', 'f'), 1);

        let mut partitioning2 = AlphabetPartitionMap::empty();
        partitioning2.insert_unchecked(CharRange::new('g', 'z'), 2);

        // Adjacent ranges, no overlap
        let refined_partitioning = partitioning1.refine(&partitioning2, |v1, v2| v1 + v2);

        let mut iter = refined_partitioning.iter();
        assert_eq!(iter.next(), Some((&CharRange::new('a', 'f'), &1))); // partitioning1
        assert_eq!(iter.next(), Some((&CharRange::new('g', 'z'), &2))); // partitioning2
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_refine_with_no_overlap() {
        let mut partitioning1 = AlphabetPartitionMap::empty();
        partitioning1.insert_unchecked(CharRange::new('a', 'c'), 1);

        let mut partitioning2 = AlphabetPartitionMap::empty();
        partitioning2.insert_unchecked(CharRange::new('x', 'z'), 2);

        // No overlap at all
        let refined_partitioning = partitioning1.refine(&partitioning2, |v1, v2| v1 + v2);

        let mut iter = refined_partitioning.iter();
        assert_eq!(iter.next(), Some((&CharRange::new('a', 'c'), &1))); // partitioning1
        assert_eq!(iter.next(), Some((&CharRange::new('x', 'z'), &2))); // partitioning2
        assert_eq!(iter.next(), None);
    }
}
