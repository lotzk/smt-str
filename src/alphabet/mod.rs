pub mod partition;

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
    /// use smt_str::alphabet::CharRange;
    /// use smt_str::SmtChar;
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
    /// use smt_str::alphabet::CharRange;
    /// use smt_str::SmtChar;
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
    /// use smt_str::alphabet::CharRange;
    /// use smt_str::SmtChar;
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
    /// # Example
    /// ```
    /// use smt_str::alphabet::CharRange;
    /// use smt_str::SmtChar;
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
    /// use smt_str::alphabet::CharRange;
    /// use smt_str::SmtChar;
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

    /// Checks if the given range is a subset of this range.
    /// A range is a subset of another range if all characters in the first range are also in the second range.
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
    /// assert!(r1.has_subset(&r2));
    /// assert!(!r1.has_subset(&r3));
    /// assert!(r1.has_subset(&r1));
    /// ```
    pub fn has_subset(&self, other: &Self) -> bool {
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
        if self.start > SmtChar::from(0) {
            result.push(Self::new(0, self.start.saturating_prev()));
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
    /// use smt_str::alphabet::{Alphabet, CharRange};
    /// use smt_str::SmtChar;
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
    /// use smt_str::alphabet::{Alphabet, CharRange};
    /// use smt_str::SmtChar;
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
        let mut last = SmtChar::from(0);
        for r in &self.ranges {
            if last < r.start {
                result.insert(CharRange::new(last, r.start.saturating_prev()));
            }
            last = match r.end.next() {
                Some(s) => s,
                None => return result, // We have reached the end of the SMT-LIB alphabet
            }
        }
        if last <= SmtChar::MAX {
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
        a.insert(CharRange::new(0, SmtChar::MAX));
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
    fn alphabet_union_comp_self(a: Alphabet) {
        let u = a.union(&a.complement());
        assert_eq!(u, Alphabet::full());
    }

    #[test]
    fn alphabet_union_comp_self_but_last() {
        let range = CharRange::new(0, SmtChar::MAX.saturating_prev());
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
}
