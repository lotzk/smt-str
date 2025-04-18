//! This module defines types for partitioning the SMT-LIB alphabet.
//!
//! Unlike [`Alphabet`], ranges are not compacted and can be adjacent.
//!
//! ## Types
//!
//! - [`AlphabetPartition`] represents a finite partition of the SMT-LIB alphabet into non-overlapping [`CharRange`]s.
//! - [`AlphabetPartitionMap<T>`] extends this with values of type `T` associated to each range.
//!
//!
//! ## Refinement
//!
//! Both struct provide a partition refinement operation.
//! Given two partitions `P` and `Q`, the **refinement** of `P` w.r.t. `P` is a
//! partitioning `R` that is the set of all non-empty intersections
//!
//! - `p ∩ q` for all `p ∈ P` and `q ∈ Q`.  
//! - `p' ∩ q` for all `p' ∈ comp(P)` and `q ∈ Q`.  
//! - `p ∩ q'` for all `p ∈ P` and `q' ∈ comp(Q)`.
//!
//! where `comp(P)` and `comp(Q)` are the complements of `P` and `Q`, respectively.
//! The resulting ranges in `R` are disjoint, ordered, and their union is equal to the union of `P ∪ Q`.
//!
//! In other words, refinement splits ranges in `P` and `Q` as needed so that the resulting partition `R` respects the boundaries of both.
//!
//! For `AlphabetPartitionMap<T>`, the refinement operation also combines the values from both input partitions using a user-provided function `f: T × T -> T`, applied to the values associated with intersecting ranges.
use std::{
    collections::{btree_map, BTreeMap},
    fmt::Display,
};

use super::CharRange;

/// A partitioning of the SMT-LIB alphabet into disjoint character ranges without associated values.
///
/// This is a convenience wrapper around [`AlphabetPartitionMap<()>`] for value-less partitioning.
/// See the module-level documentation for details.
///
/// # Example
/// ```
/// use smt_str::alphabet::{partition::AlphabetPartition, CharRange};
///
/// let mut p = AlphabetPartition::default();
/// p.insert(CharRange::new('a', 'z')).unwrap();
///
/// assert_eq!(p.len(), 1);
/// ```
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
    /// use smt_str::alphabet::{partition::AlphabetPartition, CharRange};
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
    ///
    /// This method checks whether the given range overlaps with any existing partition.  
    /// If it does not, the range is inserted and `Ok(())` is returned.  
    /// If it overlaps with an existing partition, the insertion is rejected and the overlapping range is returned in `Err(...)`.
    ///
    /// This operation takes O(n) time, where `n` is the number of ranges in the partition.  
    /// If the caller can guarantee that the range does not overlap, use [`insert_unchecked`](Self::insert_unchecked) for improved performance.
    ///
    /// # Examples
    ///
    /// ```
    /// use smt_str::alphabet::{partition::AlphabetPartition, CharRange};
    ///
    /// let mut partitioning = AlphabetPartition::empty();
    ///
    /// // Insert a non-overlapping range
    /// let range = CharRange::new('a', 'z');
    /// assert_eq!(partitioning.insert(range.clone()), Ok(()));
    /// assert!(partitioning.contains(&range));
    ///
    /// // Insert an overlapping range
    /// let overlapping = CharRange::new('m', 'p');
    /// assert_eq!(partitioning.insert(overlapping), Err(CharRange::new('a', 'z')));
    /// ```
    pub fn insert(&mut self, range: CharRange) -> Result<(), CharRange> {
        self.map.insert(range, ())
    }

    /// Inserts the given character range into the partitioning, without checking if the partitioning is still valid.
    /// Takes O(log n) time, where n is the number of partitions in the partitioning.
    ///
    /// This method must be used with caution, as it can lead to an invalid partitioning if the range overlaps with an existing partition.
    ///
    /// # Examples
    ///
    /// ```
    /// use smt_str::alphabet::{partition::AlphabetPartition, CharRange};
    ///
    /// let mut partitioning = AlphabetPartition::empty();
    /// partitioning.insert_unchecked(CharRange::new('a','z'));
    /// assert!(partitioning.contains(&CharRange::new('a','z')));
    ///
    /// // This will lead to an invalid partitioning
    /// partitioning.insert_unchecked(CharRange::new('m','p'));
    /// assert!(partitioning.contains(&CharRange::new('m','p')));
    /// assert!(!partitioning.valid());
    /// ```
    pub fn insert_unchecked(&mut self, range: CharRange) {
        self.map.insert_unchecked(range, ());
    }

    /// Returns `true` if the given character range is explicitly contained in the partitioning.
    /// This method checks for the presence of the exact range, not subranges.
    ///
    /// # Examples
    ///
    /// ```
    /// use smt_str::alphabet::{partition::AlphabetPartition, CharRange};
    ///
    /// let range = CharRange::new('a', 'z');
    /// let mut partitioning = AlphabetPartition::empty();
    /// partitioning.insert_unchecked(range.clone());
    ///
    /// assert!(partitioning.contains(&range));
    ///
    /// // Subranges are not considered present
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
    ///
    /// Only removes the range if it exactly matches a partition in the set.  
    /// Returns `true` if the range was removed, and `false` if it was not present.
    ///
    ///
    /// # Examples
    ///
    /// ```
    /// use smt_str::alphabet::{partition::AlphabetPartition, CharRange};
    ///
    /// let mut partitioning = AlphabetPartition::empty();
    /// partitioning.insert_unchecked(CharRange::new('a', 'z'));
    ///
    /// assert!(partitioning.contains(&CharRange::new('a', 'z')));
    ///
    /// // Subranges are not considered matches
    /// assert!(!partitioning.remove(CharRange::new('a', 'm')));
    ///
    /// // Exact match is required
    /// assert!(partitioning.remove(CharRange::new('a', 'z')));
    /// assert!(!partitioning.contains(&CharRange::new('a', 'z')));
    /// ```
    pub fn remove(&mut self, range: CharRange) -> bool {
        self.map.remove(range).is_some()
    }

    /// Performs a partition refinement of this partitioning with the given partitioning.
    /// See module-level documentation and [`AlphabetPartitionMap::refine`] for details.
    ///
    /// # Examples
    ///
    /// ```
    /// use smt_str::alphabet::{partition::AlphabetPartition, CharRange};
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

    /// Returns an iterator over the character ranges in the partitioning.
    ///
    /// The iterator yields each [`CharRange`] in the partitioning, in sorted order.
    ///
    /// # Examples
    ///
    /// ```
    /// use smt_str::alphabet::{partition::AlphabetPartition, CharRange};
    ///
    /// let mut partitioning = AlphabetPartition::empty();
    /// partitioning.insert_unchecked(CharRange::new('a', 'b'));
    /// partitioning.insert_unchecked(CharRange::new('x', 'z'));
    ///
    /// let mut iter = partitioning.iter();
    /// assert_eq!(iter.next(), Some(&CharRange::new('a', 'b')));
    /// assert_eq!(iter.next(), Some(&CharRange::new('x', 'z')));
    /// assert_eq!(iter.next(), None);
    /// ```
    pub fn iter(&self) -> impl Iterator<Item = &CharRange> + '_ {
        self.map.iter().map(|(r, _)| r)
    }

    /// Returns `true` if the partitioning is valid, i.e., if no two character ranges overlap.
    /// This property holds as long as `insert_unchecked` is not used to insert overlapping ranges.
    ///
    /// # Example
    /// ```
    /// use smt_str::alphabet::{partition::AlphabetPartition, CharRange};
    /// let mut p = AlphabetPartition::empty();
    /// p.insert_unchecked(CharRange::new('a', 'f'));
    /// p.insert_unchecked(CharRange::new('g', 'z'));
    /// assert!(p.valid());
    /// p.insert_unchecked(CharRange::new('e', 'h'));
    /// assert!(!p.valid());
    ///
    pub fn valid(&self) -> bool {
        self.map.valid()
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

/// A partitioning of the SMT-LIB alphabet into disjoint character ranges, each associated with a value of type `T`.
///
/// See the module-level documentation for details.
///
/// /// # Example
/// ```
/// use smt_str::alphabet::{partition::AlphabetPartitionMap, CharRange};
///
/// let mut p = AlphabetPartitionMap::empty();
/// p.insert(CharRange::new('a', 'f'), 1).unwrap();
/// p.insert(CharRange::new('g', 'z'), 2).unwrap();
///
/// assert_eq!(p.len(), 2);
/// assert_eq!(p.get(&CharRange::new('a', 'f')), Some(&1));
/// ```
#[derive(Clone, Default, Debug)]
pub struct AlphabetPartitionMap<T: Clone> {
    /// The character ranges in the partitioning and the associated values.
    /// The partitions are ordered in a BTreeMap by the start and end of the character range.
    parts: BTreeMap<CharRange, T>,
}

impl<T: Clone> AlphabetPartitionMap<T> {
    /// Creates a new, empty partitioning.
    ///
    /// The resulting partitioning contains no character ranges.
    ///
    /// # Example
    /// ```
    /// use smt_str::alphabet::partition::AlphabetPartitionMap;
    /// let p: AlphabetPartitionMap<i32> = AlphabetPartitionMap::empty();
    /// assert!(p.is_empty());
    /// assert_eq!(p.len(), 0);
    /// ```
    pub fn empty() -> Self {
        Self {
            parts: BTreeMap::new(),
        }
    }

    /// Creates a partitioning map containing a single character range with the given associated value.
    ///
    /// # Example
    /// ```
    /// use smt_str::alphabet::{partition::AlphabetPartitionMap, CharRange};
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

    /// Attempts to insert a character range with an associated value into the partitioning.
    ///
    /// This method checks whether the given range overlaps with any existing range in the partitioning.
    /// If there is no overlap, the range is inserted and `Ok(())` is returned.
    /// If the range overlaps with an existing partition, insertion is rejected and the conflicting range is returned in `Err`.
    ///
    /// This operation runs in **O(n + log n)** time, where `n` is the number of partitions.
    /// If overlap checks are not necessary, use [`insert_unchecked`](Self::insert_unchecked) for faster insertion.
    ///
    /// # Example
    /// ```
    /// use smt_str::alphabet::{partition::AlphabetPartitionMap, CharRange};
    ///
    /// let mut partitioning = AlphabetPartitionMap::empty();
    ///
    /// let range = CharRange::new('a', 'z');
    /// assert_eq!(partitioning.insert(range.clone(), 1), Ok(()));
    /// assert_eq!(partitioning.get(&range), Some(&1));
    ///
    /// // Overlapping range cannot be inserted
    /// assert_eq!(
    ///     partitioning.insert(CharRange::new('m', 'p'), 1),
    ///     Err(CharRange::new('a', 'z'))
    /// );
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

    /// Inserts a character range with its associated value into the partitioning **without** checking for overlaps.
    ///
    /// This method assumes that the given range does not overlap with any existing partition.
    /// If this assumption is violated, the internal becomes invalid.
    ///
    /// Runs in **O(log n)** time, where `n` is the number of existing partitions.
    ///
    /// Use this method only when you are certain that the inserted range does not conflict with existing ones.
    /// For safe insertion with overlap checks, use [`insert`](Self::insert).
    ///
    /// # Example
    /// ```
    /// use smt_str::alphabet::{partition::AlphabetPartitionMap, CharRange};
    ///
    /// let mut partitioning = AlphabetPartitionMap::empty();
    /// partitioning.insert_unchecked(CharRange::new('a','z'), 0);
    /// assert_eq!(partitioning.get(&CharRange::new('a','z')), Some(&0));
    ///
    /// // Overlapping insertion is allowed, but the partitioning becomes invalid
    /// partitioning.insert_unchecked(CharRange::new('m','p'), 1);
    /// assert_eq!(partitioning.get(&CharRange::new('m','p')), Some(&1));
    /// assert!(!partitioning.valid());
    /// ```
    pub fn insert_unchecked(&mut self, range: CharRange, v: T) {
        self.parts.insert(range, v);
    }

    /// Returns a reference to the value associated with the given character range, if it exists.
    ///
    /// Only exact matches are returned.
    /// That is, the given range must match a range in the map exactly.
    ///
    /// Runs in **O(log n)** time, where `n` is the number of stored partitions.
    ///
    /// # Example
    /// ```
    /// use smt_str::alphabet::{partition::AlphabetPartitionMap, CharRange};
    ///
    /// let mut partitioning = AlphabetPartitionMap::empty();
    /// let range = CharRange::new('a', 'z');
    /// partitioning.insert_unchecked(range, 42);
    ///
    /// assert_eq!(partitioning.get(&CharRange::new('a', 'z')), Some(&42));
    /// assert_eq!(partitioning.get(&CharRange::new('a', 'm')), None); // no partial match
    /// ```
    pub fn get(&self, range: &CharRange) -> Option<&T> {
        self.parts.get(range)
    }

    /// Removes the given range from the partitioning.
    ///
    /// Only removes ranges that exactly match an existing partition. Subranges or overlapping ranges will not be removed.
    ///
    /// Returns the associated value if the range was present, or `None` otherwise.
    ///
    /// Runs in **O(log n)** time, where `n` is the number of partitions.
    ///
    /// # Examples
    /// ```
    /// use smt_str::alphabet::{partition::AlphabetPartitionMap, CharRange};
    ///
    /// let mut partitioning = AlphabetPartitionMap::empty();
    /// partitioning.insert_unchecked(CharRange::new('a', 'z'), 0);
    ///
    /// // Exact match can be removed
    /// assert_eq!(partitioning.remove(CharRange::new('a', 'z')), Some(0));
    ///
    /// // Subrange does not match exactly
    /// assert_eq!(partitioning.remove(CharRange::new('a', 'm')), None);
    /// ```
    pub fn remove(&mut self, range: CharRange) -> Option<T> {
        self.parts.remove(&range)
    }

    /// Returns the number of partitions in the partitioning.
    pub fn len(&self) -> usize {
        self.parts.len()
    }

    /// Returns `true`precisely if the partitioning is empty.
    pub fn is_empty(&self) -> bool {
        self.parts.is_empty()
    }

    /// Computes the coarsest common refinement of two partitionings.
    ///
    /// Given two partitionings `self: P` and `other: Q`, this returns a new partitioning `R` such that every range `r ∈ R` is the intersection of:
    ///
    /// - a range in `P` and a range in `Q`, or
    /// - a range in `P` and a range in the complement of `Q`, or
    /// - a range in `Q` and a range in the complement of `P`
    ///
    /// In other words, `R` is the coarsest partitioning that is a common refinement of `P` and `Q`.
    ///
    /// The associated value for each `r ∈ R` is determined as follows:
    ///
    /// - If `r` is contained in a range `p ∈ P` and disjoint from all ranges in `Q`, its value is `P(p)`.
    /// - If `r` is contained in a range `q ∈ Q` and disjoint from all ranges in `P`, its value is `Q(q)`.
    /// - If `r` is the intersection of `p ∈ P` and `q ∈ Q`, its value is `f(P(p), Q(q))`.
    ///
    /// # Arguments
    ///
    /// * `other` — the partitioning to refine with
    /// * `f` — a function used to merge values when ranges from both partitionings overlap
    ///
    /// # Example
    ///
    /// ```
    /// use smt_str::alphabet::{partition::AlphabetPartitionMap, CharRange};
    ///
    /// let mut p1 = AlphabetPartitionMap::empty();
    /// p1.insert_unchecked(CharRange::new('a', 'z'), 1);
    ///
    /// let mut p2 = AlphabetPartitionMap::empty();
    /// p2.insert_unchecked(CharRange::new('b', 'c'), 2);
    ///
    /// let refined = p1.refine(&p2, |a, b| a + b);
    ///
    /// let mut iter = refined.iter();
    /// assert_eq!(iter.next(), Some((&CharRange::new('a', 'a'), &1)));
    /// assert_eq!(iter.next(), Some((&CharRange::new('b', 'c'), &3)));
    /// assert_eq!(iter.next(), Some((&CharRange::new('d', 'z'), &1)));
    /// assert_eq!(iter.next(), None);
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

    /// Refines the partitioning with a single character range and associated value.
    ///
    /// This is a convenience method that creates a singleton partitioning and invokes [`refine`] with it.
    /// The result is equivalent to refining `self` with a partitioning containing only the given range.
    ///
    /// See [`refine`](Self::refine) for the semantics of refinement.
    ///
    /// # Arguments
    ///
    /// * `range` — the range to refine with
    /// * `val` — the value associated with the range
    /// * `f` — the merge function used when `range` overlaps with an existing range
    ///
    /// # Example
    ///
    /// ```
    /// use smt_str::alphabet::{partition::AlphabetPartitionMap, CharRange};
    ///
    /// let mut p = AlphabetPartitionMap::empty();
    /// p.insert_unchecked(CharRange::new('a', 'z'), 1);
    ///
    /// let refined = p.refine_single(CharRange::new('b', 'c'), 2, |a, b| a + b);
    ///
    /// let mut iter = refined.iter();
    /// assert_eq!(iter.next(), Some((&CharRange::new('a', 'a'), &1)));
    /// assert_eq!(iter.next(), Some((&CharRange::new('b', 'c'), &3)));
    /// assert_eq!(iter.next(), Some((&CharRange::new('d', 'z'), &1)));
    /// assert_eq!(iter.next(), None);
    /// ```
    pub fn refine_single<F>(&self, range: CharRange, val: T, f: F) -> Self
    where
        F: Fn(&T, &T) -> T,
    {
        let temp_part = AlphabetPartitionMap::singleton(range, val);
        self.refine(&temp_part, f)
    }

    /// Returns an iterator over the character ranges and associated values in the partitioning.
    ///
    /// The iterator yields pairs of [`CharRange`] and references to their associated values, in ascending order of ranges.
    ///
    /// # Example
    /// ```
    /// use smt_str::alphabet::{partition::AlphabetPartitionMap, CharRange};
    ///
    /// let mut p = AlphabetPartitionMap::empty();
    /// p.insert_unchecked(CharRange::new('a', 'c'), 1);
    /// p.insert_unchecked(CharRange::new('x', 'z'), 2);
    ///
    /// let mut iter = p.iter();
    /// assert_eq!(iter.next(), Some((&CharRange::new('a', 'c'), &1)));
    /// assert_eq!(iter.next(), Some((&CharRange::new('x', 'z'), &2)));
    /// assert_eq!(iter.next(), None);
    /// ```
    pub fn iter(&self) -> impl Iterator<Item = (&CharRange, &T)> + '_ {
        self.parts.iter()
    }

    /// Returns an iterator over the character ranges and mutable references to their associated values.
    ///
    /// The iterator yields pairs of [`CharRange`] and mutable references to their associated values,
    /// in ascending order of ranges. This allows modifying the values in-place.
    ///
    /// # Example
    /// ```
    /// use smt_str::alphabet::{partition::AlphabetPartitionMap, CharRange};
    ///
    /// let mut p = AlphabetPartitionMap::empty();
    /// p.insert_unchecked(CharRange::new('a', 'c'), 1);
    /// p.insert_unchecked(CharRange::new('x', 'z'), 2);
    ///
    /// for (_, value) in p.iter_mut() {
    ///     *value += 1;
    /// }
    ///
    /// let mut iter = p.iter();
    /// assert_eq!(iter.next(), Some((&CharRange::new('a', 'c'), &2)));
    /// assert_eq!(iter.next(), Some((&CharRange::new('x', 'z'), &3)));
    /// assert_eq!(iter.next(), None);
    /// ```
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&CharRange, &mut T)> + '_ {
        self.parts.iter_mut()
    }

    /// Checks whether the given character range overlaps with any existing partition in the map.
    ///
    /// Returns the first overlapping `(CharRange, value)` pair if any overlap exists; otherwise returns `None`.
    ///
    /// This method performs a linear scan over all ranges and runs in `O(n)` time, where `n` is the number of partitions.
    /// TODO: Could be optimized to `O(log n)` using binary search.
    ///
    /// # Arguments
    /// - `range`: The [`CharRange`] to test for overlap.
    ///
    /// # Example
    /// ```
    /// use smt_str::alphabet::{partition::AlphabetPartitionMap, CharRange};
    ///
    /// let mut p = AlphabetPartitionMap::empty();
    /// p.insert_unchecked(CharRange::new('a', 'z'), 1);
    ///
    /// assert!(p.overlaps(CharRange::new('m', 'p')).is_some());
    /// assert!(p.overlaps(CharRange::new('0', '9')).is_none());
    /// ```
    pub fn overlaps(&self, range: CharRange) -> Option<(&CharRange, &T)> {
        self.parts
            .iter()
            .find(|(r, _)| !r.intersect(&range).is_empty())
    }

    /// Returns `true` if the partitioning is valid, i.e., if no two character ranges overlap.
    /// This property holds as long as `insert_unchecked` is not used to insert overlapping ranges.
    ///
    /// This method checks that for every pair of consecutive ranges `(r1, r2)`, the end of `r1` is strictly less than the start of `r2`.
    /// This ensures that the partitioning forms a set of non-overlapping ranges.
    ///
    /// Runs in `O(n)` time, where `n` is the number of partitions.
    ///
    /// # Example
    /// ```
    /// use smt_str::alphabet::{partition::AlphabetPartitionMap, CharRange};
    ///
    /// let mut p = AlphabetPartitionMap::empty();
    /// p.insert_unchecked(CharRange::new('a', 'f'), 1);
    /// p.insert_unchecked(CharRange::new('g', 'z'), 2);
    /// assert!(p.valid());
    ///
    /// // Overlapping range leads to invalid partitioning
    /// p.insert_unchecked(CharRange::new('e', 'h'), 3);
    /// assert!(!p.valid());
    /// ```
    pub fn valid(&self) -> bool {
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
        part = part.refine(&AlphabetPartition::singleton(r1));

        part = part.refine(&AlphabetPartition::singleton(r2));

        part = part.refine(&AlphabetPartition::singleton(r3));

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
        let partitioning = AlphabetPartitionMap::singleton(range, 1);
        assert_eq!(partitioning.get(&range), Some(&1));
    }

    #[test]
    fn test_insert_non_overlapping() {
        let mut partitioning = AlphabetPartitionMap::empty();

        // Insert non-overlapping ranges
        let range1 = CharRange::new('a', 'f');
        let range2 = CharRange::new('g', 'z');

        assert_eq!(partitioning.insert(range1, 1), Ok(()));
        assert_eq!(partitioning.insert(range2, 2), Ok(()));
        assert_eq!(partitioning.get(&range1), Some(&1));
        assert_eq!(partitioning.get(&range2), Some(&2));
    }

    #[test]
    fn test_insert_overlapping() {
        let mut partitioning = AlphabetPartitionMap::empty();

        // Insert initial range
        let range1 = CharRange::new('a', 'm');
        let overlapping_range = CharRange::new('g', 'z');

        assert_eq!(partitioning.insert(range1, 1), Ok(()));

        // Insert overlapping range, expect an error
        assert_eq!(partitioning.insert(overlapping_range, 2), Err(range1));
    }

    #[test]
    fn test_remove_partition() {
        let mut partitioning = AlphabetPartitionMap::empty();

        let range = CharRange::new('a', 'z');
        partitioning.insert_unchecked(range, 1);
        assert_eq!(partitioning.get(&range), Some(&1));

        // Now remove the range and check if it's gone
        partitioning.remove(range);
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
