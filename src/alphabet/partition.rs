use std::{
    collections::{btree_map, BTreeMap},
    fmt::Display,
};

use super::CharRange;

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
    /// use smt_str::alphabet::{AlphabetPartition, CharRange};
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
    /// use smt_str::alphabet::{AlphabetPartition, CharRange};
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
    /// use smt_str::alphabet::{AlphabetPartition, CharRange};
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
    /// use smt_str::alphabet::{AlphabetPartition, CharRange};
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
    /// use smt_str::alphabet::{AlphabetPartition, CharRange};
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
    /// use smt_str::alphabet::{AlphabetPartition, CharRange};
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
    /// use smt_str::alphabet::{AlphabetPartitionMap, CharRange};
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
    /// use smt_str::alphabet::{AlphabetPartitionMap, CharRange};
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
    /// use smt_str::alphabet::{AlphabetPartitionMap, CharRange};
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
    /// use smt_str::alphabet::{AlphabetPartitionMap, CharRange};
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
    /// use smt_str::alphabet::{AlphabetPartitionMap, CharRange};
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
mod test {

    use super::*;

    #[test]
    fn refine_al_subsumed() {
        let r1 = CharRange::new(2u32, 5u32);
        let r2 = CharRange::new(3u32, 6u32);
        let r3 = CharRange::new(1u32, 4u32);
        let mut part = AlphabetPartition::empty();
        part = part.refine(&AlphabetPartition::singleton(r1.clone()));

        part = part.refine(&AlphabetPartition::singleton(r2.clone()));

        part = part.refine(&AlphabetPartition::singleton(r3.clone()));

        let mut iter = part.iter();
        assert_eq!(iter.next(), Some(&CharRange::new(1u32, 1u32)));
        assert_eq!(iter.next(), Some(&CharRange::new(2u32, 2u32)));
        assert_eq!(iter.next(), Some(&CharRange::new(3u32, 4u32)));
        assert_eq!(iter.next(), Some(&CharRange::new(5u32, 5u32)));
        assert_eq!(iter.next(), Some(&CharRange::new(6u32, 6u32)));
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
