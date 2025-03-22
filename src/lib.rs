//! This crate provides types and utilities for working with SMT-LIB strings.
//! An SMT-LIB string is a sequence of characters with unicode code points in the range 0x0000 to 0x2FFFF (first two planes of Unicode).
//! The underlying semantics differ from those of Rust`s native `char` and `String` types:
//!
//! - **Character model**: Rust's `char` type represents any Unicode scalar value which is not a surrogate code point.
//!   In SMT-LIB, a character is a unicode code point in the range 0x0000 to 0x2FFFF, including surrogate code points (which are not valid Rust `char`s).
//! - **String length**: In Rust, the length of a string is counted in bytes whereas in In SMT-LIB, the length of a string is counted in characters.
//!   For example, if a character takes more than one byte to encode (such as 🦀), Rust's `String.len()` will return the number of bytes.
//!   In order to obtain the number of characters, one must count the number of `char`s in the string instead, which can easily lead to errors.
//! - **Escaping**: In SMT-LIB, the only escape sequences are of the form `\uXXXX` and `\u{X...}`.
//!   Especially, there are no escape sequences for control characters, such as `\n` or `\t`, that are present in Rust.
//!
//! This crate provides a convenient way to work with SMT-LIB strings through the [`SmtChar`] and [`SmtString`] types.
//! - [`SmtChar`] represents a Unicode code point in the range 0x0000 to 0x2FFFF (including surrogates).
//! - [`SmtString`] represents a sequence of `SmtChar` values and offers parsing, manipulation, and search utilities that conform to the SMT-LIB specification.

pub mod alphabet;
#[cfg(feature = "automata")]
pub mod automata;

#[cfg(feature = "regex")]
pub mod re;
#[cfg(feature = "sampling")]
pub mod sampling;

use std::{
    fmt::Display,
    ops::{self, Index},
};

use num_traits::{SaturatingAdd, SaturatingSub};
use quickcheck::Arbitrary;

/// The maximum unicode character.
pub const SMT_MAX_CODEPOINT: u32 = 0x2FFFF;

/// The minimum unicode character.
pub const SMT_MIN_CODEPOINT: u32 = 0x0000;

/// A Unicode character used in SMT-LIB strings.
///
/// In the SMT-LIB string theory, a character is any Unicode code point in the inclusive range
/// `0x0000` to `0x2FFFF`.
///
/// `SmtChar` is a wrapper around a `u32` and provides convenient methods to construct, inspect,
/// and manipulate SMT-LIB characters safely.
///
/// ## Examples
///
/// ```
/// use smt_str::SmtChar;
///
/// // Create a new `SmtChar` from a `u32` representing a Unicode code point.
/// let a = SmtChar::new(0x61); // 'a'
/// // Get the `u32` representation of the `SmtChar`.
/// assert_eq!(a.as_u32(), 0x61);
/// // Get the `char` representation of the `SmtChar`.
/// assert_eq!(a.as_char(), Some('a'));
///
/// // It is also possible to create an `SmtChar` from a `char`.
/// let b = SmtChar::from('b');
/// assert_eq!(b.as_u32(), 0x62);
/// assert_eq!(b.as_char(), Some('b'));
///
/// let surrogate = SmtChar::new(0xD800); // valid in SMT-LIB, invalid as Rust `char`
/// assert_eq!(surrogate.as_char(), None);
///
/// // Non-printable characters are escaped when displayed.
/// let newline = SmtChar::new(0x0A); // '\n'
/// assert_eq!(newline.to_string(), r#"\u{A}"#);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SmtChar(u32);

impl SmtChar {
    /// The maximum `SmtChar`.
    /// This is the unicode code point 0x2FFFF.
    pub const MAX: Self = Self(SMT_MAX_CODEPOINT);

    /// The minimum `SmtChar`.
    /// This is the unicode code point 0x0000.
    pub const MIN: Self = Self(SMT_MIN_CODEPOINT);

    /// Creates a new `SmtChar` from a `u32` code point.
    /// Panics if `c as u32 > 0x2FFFF`.
    pub fn new(c: u32) -> Self {
        assert!(c <= 0x2FFFF, "character out of range: {}", c);
        SmtChar(c)
    }

    /// Creates the `SmtChar` with the code point `1`.
    fn one() -> Self {
        SmtChar::new(1)
    }

    /// Get the `char` representation of this `SmtChar`, if it can be represented as a `char`.
    /// Returns `None` if this `SmtChar` is a surrogate code point.
    ///
    ///
    /// # Examples
    /// ```
    /// use smt_str::SmtChar;
    /// let c = SmtChar::from('a');
    /// assert_eq!(c.as_char(), Some('a'));
    /// // This is a surrogate code point and cannot be represented as a `char`.
    /// assert_eq!(SmtChar::new(55296).as_char(), None);
    ///```
    pub fn as_char(self) -> Option<char> {
        char::from_u32(self.0)
    }

    /// Get the `u32` representation of this `SmtChar`.
    /// The `u32` is the unicode code point of this `SmtChar`.
    ///
    /// # Examples
    /// ```
    /// use smt_str::SmtChar;
    /// assert_eq!(SmtChar::from('a').as_u32(), 97);
    /// assert_eq!(SmtChar::from('🦀').as_u32(), 129408);
    /// ```
    pub fn as_u32(self) -> u32 {
        self.0
    }

    /// Returns the next `SmtChar` in the range 0x0000 to 0x2FFFF.
    /// Panics if this `SmtChar` is the maximum `SmtChar`.
    ///
    /// # Examples
    /// ```
    /// use smt_str::SmtChar;
    /// let c = SmtChar::from('a');
    /// assert_eq!(c.next(), SmtChar::from('b'));
    /// ```
    ///
    /// Cannot get the next character after the maximum `SmtChar`:
    /// ```should_panic
    /// use smt_str::SmtChar;
    /// let c = SmtChar::MAX;
    /// let _ = c.next(); // panics
    /// ```
    pub fn next(self) -> Self {
        self + Self::one()
    }

    /// Returns the next `SmtChar` in the range 0x0000 to 0x2FFFF, if it exists.
    /// Returns `None` if this `SmtChar` is the maximum `SmtChar`.
    ///
    /// # Examples
    /// ```
    /// use smt_str::SmtChar;
    /// let c = SmtChar::from('a');
    /// assert_eq!(c.try_next(), Some(SmtChar::from('b')));
    /// assert_eq!(SmtChar::MAX.try_next(), None);
    /// ```
    pub fn try_next(self) -> Option<Self> {
        if self == Self::MAX {
            None
        } else {
            Some(self.next())
        }
    }

    /// Like `next`, but instead of panicking when this `SmtChar` is the maximum `SmtChar`, it returns the maximum `SmtChar`.
    ///
    /// # Examples
    /// ```
    /// use smt_str::SmtChar;
    /// let c = SmtChar::from('a');
    /// assert_eq!(c.saturating_next(), SmtChar::from('b'));
    /// assert_eq!(SmtChar::MAX.saturating_next(), SmtChar::MAX);
    /// ```
    pub fn saturating_next(self) -> Self {
        self.try_next().unwrap_or(Self::MAX)
    }

    /// Returns the previous `SmtChar` in the range 0x0000 to 0x2FFFF.
    /// Panics if this `SmtChar` is the minimum `SmtChar`.
    ///
    /// # Examples
    /// ```
    /// use smt_str::SmtChar;
    /// let c = SmtChar::from('b');
    /// assert_eq!(c.prev(), SmtChar::from('a'));
    /// ```
    ///
    /// Cannot get the previous character before the minimum `SmtChar`:
    ///
    /// ```should_panic
    /// use smt_str::SmtChar;
    /// let c = SmtChar::MIN;
    /// let _ = c.prev(); // panics
    /// ```
    pub fn prev(self) -> Self {
        self - Self::one()
    }

    /// Returns the previous `SmtChar` in the range 0x0000 to 0x2FFFF, if it exists.
    /// Returns `None` if this `SmtChar` is the minimum `SmtChar`.
    ///
    /// # Examples
    /// ```
    /// use smt_str::SmtChar;
    /// let c = SmtChar::from('b');
    /// assert_eq!(c.try_prev(), Some(SmtChar::from('a')));
    /// assert_eq!(SmtChar::MIN.try_prev(), None);
    /// ```
    pub fn try_prev(self) -> Option<Self> {
        if self == Self::MIN {
            None
        } else {
            Some(self.prev())
        }
    }

    /// Like `prev`, but instead of panicking when this `SmtChar` is the minimum `SmtChar`, it returns the minimum `SmtChar`.
    ///
    /// # Examples
    /// ```
    /// use smt_str::SmtChar;
    /// let c = SmtChar::from('b');
    /// assert_eq!(c.saturating_prev(), SmtChar::from('a'));
    /// assert_eq!(SmtChar::MIN.saturating_prev(), SmtChar::MIN);
    /// ```
    pub fn saturating_prev(self) -> Self {
        self.try_prev().unwrap_or(Self::MIN)
    }

    /// Returns `true` if this `SmtChar` is a printable ASCII character.
    /// Printable ASCII characters are in the range 0x00020 to 0x0007E.
    ///
    /// # Examples
    /// ```
    /// use smt_str::SmtChar;
    /// assert!(SmtChar::from('a').printable());
    /// assert!(!SmtChar::from('\n').printable());
    /// ```
    pub fn printable(self) -> bool {
        0x00020 <= self.0 && self.0 < 0x0007E
    }

    /// Escape this `SmtChar` as a  Unicode escape sequence.
    /// The escape sequence is of the form `\u{X}` where `X` is the hexadecimal representation of the unicode code point.
    /// The function always chooses the shortest escape sequence, i.e., it uses the smallest number of digits and does not pad with zeros.
    ///
    /// # Examples
    /// ```
    /// use smt_str::SmtChar;
    /// assert_eq!(SmtChar::from('a').escape(), r#"\u{61}"#);
    /// assert_eq!(SmtChar::from('\n').escape(), r#"\u{A}"#);
    /// assert_eq!(SmtChar::from('🦀').escape(), r#"\u{1F980}"#);
    /// assert_eq!(SmtChar::MAX.escape(), r#"\u{2FFFF}"#);
    /// assert_eq!(SmtChar::MIN.escape(), r#"\u{0}"#);
    ///
    /// ```
    pub fn escape(self) -> String {
        let mut escaped = String::with_capacity(6);
        escaped.push('\\');
        escaped.push('u');
        escaped.push('{');
        escaped.push_str(&format!("{:X}", self.0));
        escaped.push('}');
        escaped
    }

    /// Unescape a string that contains escaped characters.
    /// Escaped characters are of the following form:
    ///
    /// - `\uDDDD`
    /// - `\u{D}`,
    /// - `\u{DD}`,
    /// - `\u{DDD}`,
    /// - `\u{DDDD}`,
    /// - `\u{DDDDD}`
    ///
    /// where `D` is a hexadecimal digit. In the case `\u{DDDDD}`, the first digit must be in the range 0 to 2.
    /// The function returns `None` if the input string is not a valid escaped character.
    ///
    /// # Examples
    /// ```
    /// use smt_str::SmtChar;
    /// assert_eq!(SmtChar::unescape(r#"\u{61}"#), Some(SmtChar::from('a')));
    /// assert_eq!(SmtChar::unescape(r#"\u{A}"#), Some(SmtChar::from('\n')));
    /// assert_eq!(SmtChar::unescape(r#"\u{1F980}"#), Some(SmtChar::from('🦀')));
    /// assert_eq!(SmtChar::unescape(r#"\u{2FFFF}"#), Some(SmtChar::MAX));
    /// assert_eq!(SmtChar::unescape(r#"\u{0}"#), Some(SmtChar::MIN));
    ///
    /// // Invalid escape sequences
    ///
    /// assert_eq!(SmtChar::unescape(r#"\u{3000A}"#), None); // out of range
    /// assert_eq!(SmtChar::unescape(r#"\u{61"#), None); // missing closing brace
    /// assert_eq!(SmtChar::unescape(r#"\u{}"#), None); // empty digits
    /// ```
    pub fn unescape(escaped: &str) -> Option<Self> {
        let mut chars = escaped.chars();
        if chars.next()? != '\\' {
            return None;
        }
        if chars.next()? != 'u' {
            return None;
        }
        let mut digits = Vec::with_capacity(5);
        let mut lparen = false;
        let mut rparen = false;
        for c in chars {
            match c {
                '{' if !lparen => {
                    lparen = true;
                }
                '}' if lparen => {
                    rparen = true;
                }
                c if !rparen && c.is_ascii_hexdigit() => {
                    digits.push(c);
                }
                _ => {
                    return None;
                }
            }
        }
        if lparen && !rparen {
            return None;
        }
        if digits.is_empty() {
            return None;
        }
        // Convert the digits to a u32
        let mut code = 0;
        for c in digits {
            let digit = c.to_digit(16)?;
            code = code * 16 + digit;
        }
        if code > SMT_MAX_CODEPOINT {
            return None;
        }
        Some(SmtChar(code))
    }
}

/* Conversion from primitives */

impl From<u8> for SmtChar {
    fn from(c: u8) -> Self {
        SmtChar::new(c as u32)
    }
}

impl From<u16> for SmtChar {
    fn from(c: u16) -> Self {
        SmtChar::new(c as u32)
    }
}

impl From<u32> for SmtChar {
    fn from(c: u32) -> Self {
        SmtChar::new(c)
    }
}

impl From<i32> for SmtChar {
    fn from(c: i32) -> Self {
        if c < 0 {
            panic!("negative character: {}", c);
        }
        SmtChar::new(c as u32)
    }
}

impl From<char> for SmtChar {
    fn from(c: char) -> Self {
        SmtChar::new(c as u32)
    }
}

/* Operations */

impl ops::Add<SmtChar> for SmtChar {
    type Output = SmtChar;
    /// Adds another `SmtChar` to this `SmtChar`, shifting the unicode code point.
    /// The sum is the sum of the unicode code points of the two `SmtChar`s.
    /// Panics if the resulting code point is greater than [SMT_MAX_CODEPOINT] (= `0x2FFFF`).
    ///
    /// # Examples
    /// ```
    /// use smt_str::SmtChar;
    /// let c = SmtChar::from('a');
    /// assert_eq!(c + SmtChar::new(1), SmtChar::from('b'));
    /// assert_eq!(c + SmtChar::new(25), SmtChar::from('z'));
    /// ```
    ///
    /// Overflowing the maximum code point panics:
    ///
    /// ```should_panic
    /// use smt_str::SmtChar;
    /// let c = SmtChar::MAX;
    /// let _ = c + SmtChar::new(1); // panics
    /// ```
    fn add(self, rhs: SmtChar) -> Self::Output {
        SmtChar::new(self.0 + rhs.0)
    }
}

impl SaturatingAdd for SmtChar {
    /// Adds another `SmtChar` to this `SmtChar`, saturating at the maximum unicode code point.
    ///
    /// # Examples
    /// ```
    /// use smt_str::SmtChar;
    /// use num_traits::ops::saturating::SaturatingAdd;
    /// let c = SmtChar::from('a');
    /// assert_eq!(c.saturating_add(&SmtChar::new(1)), SmtChar::from('b'));
    /// assert_eq!(c.saturating_add(&SmtChar::new(25)), SmtChar::from('z'));
    /// let c = SmtChar::MAX;
    /// assert_eq!(c.saturating_add(&SmtChar::new(1)), SmtChar::MAX);
    /// ```
    fn saturating_add(&self, v: &Self) -> Self {
        let sum = (self.0 + v.0).min(SMT_MAX_CODEPOINT);
        SmtChar::new(sum)
    }
}

impl ops::Sub<SmtChar> for SmtChar {
    type Output = SmtChar;
    /// Subtracts another `SmtChar` from this `SmtChar`, shifting the unicode code point.
    /// The difference is the difference of the unicode code points of the two `SmtChar`s.
    /// Panics if the resulting code point is less than [SMT_MIN_CODEPOINT] (= `0`).
    ///
    /// # Examples
    /// ```
    /// use smt_str::SmtChar;
    /// let c = SmtChar::from('z');
    /// assert_eq!(c - SmtChar::new(1), SmtChar::from('y'));
    /// assert_eq!(c - SmtChar::new(25), SmtChar::from('a'));
    /// ```
    ///
    /// Underflowing the minimum code point panics:
    ///
    /// ```should_panic
    /// use smt_str::SmtChar;
    /// let c = SmtChar::MIN;
    /// let _ = c - SmtChar::new(1); // panics
    /// ```
    fn sub(self, rhs: SmtChar) -> Self::Output {
        SmtChar::new(self.0 - rhs.0)
    }
}

impl SaturatingSub for SmtChar {
    /// Subtracts another `SmtChar` from this `SmtChar`, saturating at the minimum unicode code point.
    ///
    /// # Examples
    /// ```
    /// use smt_str::SmtChar;
    /// let c = SmtChar::from('z');
    /// use num_traits::ops::saturating::SaturatingSub;
    /// assert_eq!(c.saturating_sub(&SmtChar::new(1)), SmtChar::from('y'));
    /// assert_eq!(c.saturating_sub(&SmtChar::new(25)), SmtChar::from('a'));
    /// let c = SmtChar::MIN;
    /// assert_eq!(c.saturating_sub(&SmtChar::new(1)), SmtChar::MIN);
    /// ```
    fn saturating_sub(&self, v: &Self) -> Self {
        let diff = self.0.saturating_sub(v.0);
        SmtChar::new(diff)
    }
}

impl Display for SmtChar {
    /// Display the `SmtChar` as a Unicode character if it is printable.
    /// Otherwise, display the character as a Unicode escape sequence (see [SmtChar::escape]).
    /// Additionally, backslashes and quotes are escaped.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.printable() {
            // printable ASCII character are always safe to unwrap
            let c = self.as_char().unwrap();
            // Although the character is printable, we still escape it if it is a backslash or a quote
            if c == '\\' || c == '"' {
                write!(f, "{}", self.escape())
            } else {
                write!(f, "{}", c)
            }
        } else {
            write!(f, "{}", self.escape())
        }
    }
}

/// An iterator over the characters in the range `start` to `end`.
#[derive(Debug, Clone)]
pub struct CharIterator {
    current: SmtChar,
    end: SmtChar,
}

impl CharIterator {
    /// Create a new iterator over the characters in the range `start` to `end` (both inclusively).
    pub fn new(start: SmtChar, end: SmtChar) -> Self {
        CharIterator {
            current: start,
            end,
        }
    }
}
impl Iterator for CharIterator {
    type Item = SmtChar;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current <= self.end {
            let c = self.current;
            self.current = self.current.try_next()?;
            Some(c)
        } else {
            None
        }
    }
}

/// An SMT-LIB string is a sequence of characters with unicode code points in the range 0x0000 to 0x2FFFF (first two planes of Unicode).
/// The characters are represented by the [`SmtChar`] type.
///
/// # Examples
/// ```
/// use smt_str::{SmtString, SmtChar};
///
/// // Create a new SmtString from a string literal
/// let s: SmtString = "foo".into();
///
/// // Obtain the length of the string
/// assert_eq!(s.len(), 3);
///
/// // SmtStrings have the correct length even for multi-byte characters
/// let s: SmtString = "🦀".into();
/// assert_eq!(s.len(), 1);
///
/// // In Rust, the length of a string is counted in bytes, not characters
/// assert_eq!("🦀".len(), 4);
/// assert_eq!("🦀".chars().count(), 1);
///
/// // SmtString can be parsed from a string with escape sequences
/// let s: SmtString = SmtString::parse(r#"foo\u{61}bar"#);
/// assert_eq!(s, SmtString::from("fooabar"));
///
/// // Printing the string escapes non-printable characters
/// let s: SmtString = SmtString::from("foo\nbar");
/// assert_eq!(s.to_string(), r#"foo\u{A}bar"#);
///
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SmtString(Vec<SmtChar>);

impl SmtString {
    /// The empty string.
    pub fn empty() -> Self {
        SmtString(Vec::new())
    }

    /// Create a new SmtString from a vector of SmtChar.
    pub fn new(chars: Vec<SmtChar>) -> Self {
        SmtString(chars)
    }

    /// Parse a string into an SmtString.
    ///
    /// The input string can contain escaped characters. The only valid escape sequences are:
    ///
    /// - `\uDDDD`
    /// - `\u{D}`,
    /// - `\u{DD}`,
    /// - `\u{DDD}`,
    /// - `\u{DDDD}`,
    /// - `\u{DDDDD}`
    ///
    /// where `D` is a hexadecimal digit such that the resulting code point is in the range 0x0000 to 0x2FFFF.
    /// If the string contains valid escape sequences, they are replaced with the corresponding `SmtChar`.
    /// If the string contains invalid escape sequences, they are treated as literals.
    /// For example, the string `"foo\u{61}bar"` is parsed as the string `"fooabar"`.
    /// But the string `"foo\u{61bar"` is parsed as the string `"foo\u{61bar"`.
    /// The holds for syntactically valid escape sequences that result in code points outside the valid range.
    /// For example, the string `"foo\u{3000A}bar"` is parsed as the string `"foo\u{3000A}bar"`, since the code point `0x3000A` is outside the valid range.
    ///
    /// # Examples
    /// ```
    /// use smt_str::{SmtString};
    /// let s: SmtString = SmtString::parse(r#"foo\u{61}bar"#);
    /// assert_eq!(s, SmtString::from("fooabar"));
    ///
    /// // Invalid escape sequence, treated as literal
    ///
    /// let s: SmtString = SmtString::parse(r#"foo\u{61bar"#);
    /// assert_eq!(s, SmtString::from(r#"foo\u{61bar"#));
    /// ```
    pub fn parse(input: &str) -> Self {
        let mut chars = input.chars().peekable();
        let mut result = Vec::new();
        let mut buffer = String::new();

        while let Some(&c) = chars.peek() {
            if c == '\\' {
                // Process escaped sequence
                buffer.clear();
                while let Some(&c) = chars.peek() {
                    buffer.push(c);
                    chars.next(); // Consume character
                    if c == '}' || (buffer.len() == 6 && !buffer.contains('{')) {
                        break;
                    }
                }
                if let Some(smt_char) = SmtChar::unescape(&buffer) {
                    result.push(smt_char);
                } else {
                    // Invalid escape sequence, treat as literal
                    for c in buffer.chars() {
                        result.push(c.into());
                    }
                }
            } else {
                // Process regular character
                result.push(SmtChar(c as u32));
                chars.next(); // Consume character
            }
        }

        SmtString(result)
    }

    /// Returns whether this string is empty.
    ///
    /// # Examples
    /// ```
    /// use smt_str::{SmtString};
    /// assert!(SmtString::empty().is_empty());
    /// let s: SmtString = "foo".into();
    /// assert!(!s.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Returns the length of this string.
    ///
    /// # Examples
    /// ```
    /// use smt_str::{SmtString};
    /// assert_eq!(SmtString::empty().len(), 0);
    /// let s: SmtString = "foo".into();
    /// assert_eq!(s.len(), 3);
    /// ```
    ///
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Empties this string, removing all characters.
    /// After calling this method, the string will be empty.
    ///
    /// # Examples
    /// ```
    /// use smt_str::{SmtString};
    /// let mut s: SmtString = "foo".into();
    /// s.clear();
    /// assert!(s.is_empty());
    /// ```
    pub fn clear(&mut self) {
        self.0.clear();
    }

    /// Appends the characters of `other` to this string.
    /// The characters are appended in order.
    ///
    /// # Examples
    /// ```
    /// use smt_str::{SmtString, SmtChar};
    /// let mut s: SmtString = "foo".into();
    /// let other: SmtString = "bar".into();
    /// s.append(&other);
    /// let mut iter = s.iter();
    /// assert_eq!(iter.next(), Some(&SmtChar::from('f')));
    /// assert_eq!(iter.next(), Some(&SmtChar::from('o')));
    /// assert_eq!(iter.next(), Some(&SmtChar::from('o')));
    /// assert_eq!(iter.next(), Some(&SmtChar::from('b')));
    /// assert_eq!(iter.next(), Some(&SmtChar::from('a')));
    /// assert_eq!(iter.next(), Some(&SmtChar::from('r')));
    /// assert_eq!(iter.next(), None);
    /// ```
    pub fn append(&mut self, other: &SmtString) {
        self.0.extend(other.0.iter().copied());
    }

    /// Pushes a character to the end of this string.
    ///
    /// # Examples
    /// ```
    /// use smt_str::{SmtString, SmtChar};
    /// let mut s = SmtString::empty();
    /// s.push(SmtChar::from('f'));
    /// s.push(SmtChar::from('o'));
    /// s.push(SmtChar::from('o'));
    /// assert_eq!(s, SmtString::from("foo"));  
    /// ```
    pub fn push(&mut self, c: impl Into<SmtChar>) {
        self.0.push(c.into());
    }

    /// Concatenates this string with `other` and returns the result.
    /// This is a convenience method that does not modify this string.
    /// The characters of `other` are appended to the characters of this string, see [append].
    ///
    /// # Examples
    /// ```
    /// use smt_str::{SmtString};
    /// let s1 = SmtString::from("foo");
    /// let s2 = SmtString::from("bar");
    /// let s3 = s1.concat(&s2);
    /// assert_eq!(s3, SmtString::from("foobar"));
    /// ```
    pub fn concat(&self, other: &SmtString) -> SmtString {
        let mut s = self.clone();
        s.append(other);
        s
    }

    /// Checks if this string contains a character.
    ///
    /// # Examples
    /// ```
    /// use smt_str::{SmtString, SmtChar};
    /// let s: SmtString = "foobar".into();
    /// assert!(s.contains_char('f'));
    /// assert!(s.contains_char('o'));
    /// assert!(s.contains_char('b'));
    /// assert!(!s.contains_char('z'));
    /// ```
    pub fn contains_char(&self, c: impl Into<SmtChar>) -> bool {
        self.0.contains(&c.into())
    }

    /// Return whether this string contains another string as a factor.
    /// This is a naive implementation that checks all possible factors of this string, leading to O(n^2) complexity.
    ///
    /// # Examples
    /// ```
    /// use smt_str::{SmtString};
    /// let s: SmtString = "foobar".into();
    /// assert!(s.contains(&SmtString::empty()));
    /// assert!(s.contains(&SmtString::from("foo")));
    /// assert!(s.contains(&SmtString::from("bar")));
    /// assert!(s.contains(&SmtString::from("oba")));
    /// assert!(!s.contains(&SmtString::from("baz")));
    ///
    /// // The empty string contains only has the empty string as a factor
    /// let empty: SmtString = SmtString::empty();
    /// assert!(empty.contains(&SmtString::empty()));
    /// assert!(!empty.contains(&SmtString::from("foo")));
    /// ```
    pub fn contains(&self, factor: &SmtString) -> bool {
        self.index_of(factor, 0).is_some()
    }

    /// Find the index of the first occurrence of a factor in the suffix of this string starting at `start`.
    /// Returns `None` if the factor is not found.
    /// The empty string is a factor of every string and will always return `Some(0)`.
    ///
    /// # Examples
    /// ```
    /// use smt_str::{SmtString};
    /// let s: SmtString = "foobar".into();
    /// assert_eq!(s.index_of(&SmtString::empty(),0), Some(0));
    /// assert_eq!(s.index_of(&SmtString::from("foo"),0), Some(0));
    /// assert_eq!(s.index_of(&SmtString::from("foo"),1), None);
    /// assert_eq!(s.index_of(&SmtString::from("bar"),0), Some(3));
    /// assert_eq!(s.index_of(&SmtString::from("oba"),0), Some(2));
    /// assert_eq!(s.index_of(&SmtString::from("baz"),0), None);
    ///
    /// // If the string is empty, the only factor is the empty string
    /// let empty: SmtString = SmtString::empty();
    /// assert_eq!(empty.index_of(&SmtString::empty(),0), Some(0));
    /// assert_eq!(empty.index_of(&SmtString::from("foo"),0), None);
    /// ```
    pub fn index_of(&self, factor: &SmtString, start: usize) -> Option<usize> {
        if self.is_empty() {
            return if factor.is_empty() { Some(0) } else { None };
        }

        (start..self.len()).find(|&i| self.drop(i).starts_with(factor))
    }

    /// Returns whether this string starts with a prefix.
    /// The empty string is a prefix of every string.
    ///
    /// # Examples
    /// ```
    /// use smt_str::{SmtString};
    /// let s: SmtString = "foobar".into();
    /// assert!(s.starts_with(&SmtString::empty()));
    /// assert!(s.starts_with(&SmtString::from("foo")));
    /// assert!(!s.starts_with(&SmtString::from("bar")));
    /// ```
    pub fn starts_with(&self, prefix: &SmtString) -> bool {
        self.0.starts_with(&prefix.0)
    }

    /// Returns whether this string ends with a suffix.
    /// The empty string is a suffix of every string.
    ///
    /// # Examples
    /// ```
    /// use smt_str::{SmtString};
    /// let s: SmtString = "foobar".into();
    /// assert!(s.ends_with(&SmtString::empty()));
    /// assert!(s.ends_with(&SmtString::from("bar")));
    /// assert!(!s.ends_with(&SmtString::from("foo")));
    /// ```
    pub fn ends_with(&self, suffix: &SmtString) -> bool {
        self.0.ends_with(&suffix.0)
    }

    /// Returns the first character of this string, if it is not empty.
    /// Returns `None` if this string is empty.
    ///
    /// # Examples
    /// ```
    /// use smt_str::{SmtString, SmtChar};
    /// let s: SmtString = "foo".into();
    /// assert_eq!(s.first(), Some('f'.into()));
    /// assert_eq!(SmtString::empty().first(), None);
    /// ```
    pub fn first(&self) -> Option<SmtChar> {
        self.0.first().copied()
    }

    /// Returns the last character of this string, if it is not empty.
    /// Returns `None` if this string is empty.
    ///
    /// # Examples
    /// ```
    /// use smt_str::{SmtString, SmtChar};
    /// let s: SmtString = "foo".into();
    /// assert_eq!(s.last(), Some('o'.into()));
    /// assert_eq!(SmtString::empty().last(), None);
    /// ```
    pub fn last(&self) -> Option<SmtChar> {
        self.0.last().copied()
    }

    /// Returns the prefix of length `n` of this string.
    /// If `n` is greater than the length of this string, the entire string is returned.
    /// If `n` is zero, the empty string is returned.
    ///
    /// # Examples
    /// ```
    /// use smt_str::{SmtString, SmtChar};
    /// let s: SmtString = "foo".into();
    ///
    /// assert_eq!( s.take(2), SmtString::from("fo"));
    /// assert!(s.take(10) == s);
    /// assert!(s.take(0) == SmtString::empty());
    /// ```
    ///
    pub fn take(&self, n: usize) -> SmtString {
        SmtString(self.0.iter().copied().take(n).collect())
    }

    /// Returns the suffix of this string after removing the first `n` characters.
    /// If `n` is greater than the length of this string, the empty string is returned.
    /// If `n` is zero, the entire string is returned.
    ///
    /// # Examples
    /// ```
    /// use smt_str::{SmtString, SmtChar};
    /// let s = SmtString::from("foo");
    /// assert_eq!(s.drop(2), SmtString::from("o"));
    /// assert_eq!(s.drop(10), SmtString::empty());
    /// assert_eq!(s.drop(0), s);
    /// ```
    pub fn drop(&self, n: usize) -> SmtString {
        SmtString(self.0.iter().copied().skip(n).collect())
    }

    /// Returns the `n`-th character of this string.
    /// Returns `None` if `n` is greater than or equal to the length of this string.
    ///
    /// # Examples
    /// ```
    /// use smt_str::{SmtString, SmtChar};
    /// let s: SmtString = "foo".into();
    /// assert_eq!(s.nth(0), Some(SmtChar::from('f')));
    /// assert_eq!(s.nth(1), Some(SmtChar::from('o')));
    /// assert_eq!(s.nth(2), Some(SmtChar::from('o')));
    /// assert_eq!(s.nth(3), None);
    /// ```
    pub fn nth(&self, n: usize) -> Option<SmtChar> {
        self.0.get(n).copied()
    }

    /// Returns the reverse of this string.
    ///
    /// # Examples
    /// ```
    /// use smt_str::{SmtString, SmtChar};
    /// let s: SmtString = "foo".into();
    /// let rev = s.reversed();
    /// let mut iter = rev.iter();
    /// assert_eq!(iter.next(), Some(&SmtChar::from('o')));
    /// assert_eq!(iter.next(), Some(&SmtChar::from('o')));
    /// assert_eq!(iter.next(), Some(&SmtChar::from('f')));
    /// assert_eq!(iter.next(), None);
    /// ```
    pub fn reversed(&self) -> Self {
        SmtString(self.0.iter().rev().copied().collect())
    }

    /// Repeat this string `n` times.
    /// If `n` is zero, the empty string is returned.
    /// If this string is empty, the empty string is returned.
    /// If `n` is one, this string is returned.
    /// Otherwise, the string is repeated `n` times.
    ///
    /// # Examples
    /// ```
    /// use smt_str::{SmtString, SmtChar};
    /// let s = SmtString::from("foo");
    /// assert_eq!(s.repeat(0), SmtString::empty());
    /// assert_eq!(s.repeat(1), s);
    /// assert_eq!(s.repeat(2), SmtString::from("foofoo"));
    /// ```
    pub fn repeat(&self, n: usize) -> Self {
        let mut result = Vec::with_capacity(self.len() * n);
        for _ in 0..n {
            result.extend(self.0.iter().copied());
        }
        SmtString(result)
    }

    /// Replaces the first occurrence of `from` in this string with `to`.
    /// If `from` is not found in this string, the string is returned unchanged.
    ///
    /// # Examples
    /// ```
    /// use smt_str::{SmtString};
    /// let s: SmtString = "barbar".into();
    /// let from: SmtString = "bar".into();
    /// let to: SmtString = "foo".into();
    /// assert_eq!(s.replace(&from, &to), SmtString::from("foobar"));
    /// ```
    pub fn replace(&self, from: &SmtString, to: &SmtString) -> SmtString {
        let mut result = SmtString::empty();
        if let Some(j) = self.index_of(from, 0) {
            result.append(&self.take(j));
            result.append(to);
            let i = j + from.len();
            result.append(&self.drop(i));
        } else {
            result = self.clone();
        }
        result
    }

    /// Replaces all occurrences of `from` in this string with `to`.
    /// If `from` is not found in this string, the string is returned unchanged.
    /// If `from` is the empty string, the string is returned unchanged.
    ///
    /// # Examples
    /// ```
    /// use smt_str::{SmtString};
    /// let s: SmtString = "barbar".into();
    /// let from: SmtString = "bar".into();
    /// let to: SmtString = "foo".into();
    /// assert_eq!(s.replace_all(&from, &to), SmtString::from("foofoo"));
    /// ```
    pub fn replace_all(&self, from: &SmtString, to: &SmtString) -> SmtString {
        if from.is_empty() || self.is_empty() {
            return self.clone(); // No changes needed if `from` is empty or `self` is empty
        }

        let mut result = SmtString::empty();
        let mut current = self.clone();

        while let Some(j) = current.index_of(from, 0) {
            result.append(&current.take(j));
            result.append(to);
            let i = j + from.len();
            current = current.drop(i);
        }

        result.append(&current);
        result
    }

    /// Returns an iterator over the characters of this string.
    pub fn iter(&self) -> std::slice::Iter<SmtChar> {
        self.0.iter()
    }
}

/* Conversions */

impl FromIterator<SmtChar> for SmtString {
    fn from_iter<I: IntoIterator<Item = SmtChar>>(iter: I) -> Self {
        SmtString(iter.into_iter().collect())
    }
}

impl FromIterator<SmtString> for SmtString {
    fn from_iter<I: IntoIterator<Item = SmtString>>(iter: I) -> Self {
        iter.into_iter()
            .fold(SmtString::empty(), |acc, s| acc.concat(&s))
    }
}

impl From<&str> for SmtString {
    fn from(s: &str) -> Self {
        SmtString(s.chars().map(SmtChar::from).collect())
    }
}

impl From<String> for SmtString {
    fn from(s: String) -> Self {
        SmtString::from(s.as_str())
    }
}

impl From<SmtChar> for SmtString {
    fn from(c: SmtChar) -> Self {
        SmtString(vec![c])
    }
}

impl Index<usize> for SmtString {
    type Output = SmtChar;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl Display for SmtString {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for c in &self.0 {
            write!(f, "{}", c)?;
        }
        Ok(())
    }
}

impl Arbitrary for SmtChar {
    fn arbitrary(g: &mut quickcheck::Gen) -> Self {
        let code = u32::arbitrary(g) % (SMT_MAX_CODEPOINT + 1);
        SmtChar(code)
    }
}

impl Arbitrary for SmtString {
    fn arbitrary(g: &mut quickcheck::Gen) -> Self {
        let len = usize::arbitrary(g) % 100;
        let chars = std::iter::repeat_with(|| SmtChar::arbitrary(g))
            .take(len)
            .collect();
        SmtString(chars)
    }
}

#[cfg(test)]
mod tests {

    use quickcheck::TestResult;
    use quickcheck_macros::quickcheck;

    use super::*;

    #[quickcheck]
    fn next_prev_inverse(s: SmtChar) -> TestResult {
        if s == SmtChar::MAX {
            return TestResult::discard();
        }
        let next = s.next();
        assert_eq!(next.prev(), s);
        TestResult::passed()
    }

    #[quickcheck]
    fn prev_next_inverse(s: SmtChar) -> TestResult {
        if s == SmtChar::MIN {
            return TestResult::discard();
        }
        let prev = s.prev();
        assert_eq!(prev.next(), s);
        TestResult::passed()
    }

    #[test]
    #[should_panic]
    fn next_past_max() {
        SmtChar::MAX.next();
    }

    #[test]
    #[should_panic]
    fn prev_past_min() {
        SmtChar::MIN.prev();
    }

    #[test]
    fn test_unescape_valid() {
        assert_eq!(SmtChar::unescape(r#"\u000A"#), Some(SmtChar(0x000A)));
        assert_eq!(SmtChar::unescape(r#"\u{A}"#), Some(SmtChar(0x000A)));
        assert_eq!(SmtChar::unescape(r#"\u{0A}"#), Some(SmtChar(0x000A)));
        assert_eq!(SmtChar::unescape(r#"\u{00A}"#), Some(SmtChar(0x000A)));
        assert_eq!(SmtChar::unescape(r#"\u{000A}"#), Some(SmtChar(0x000A)));
        assert_eq!(SmtChar::unescape(r#"\u{000A}"#), Some(SmtChar(0x000A)));
        assert_eq!(SmtChar::unescape(r#"\u{0000A}"#), Some(SmtChar(0x000A)));
    }

    #[test]
    fn test_unescape_empty() {
        assert_eq!(SmtChar::unescape(r#"\u{}"#), None);
    }

    #[test]
    fn test_unescape_too_large() {
        assert_eq!(SmtChar::unescape(r#"\u{3000A}"#), None);
        assert_eq!(SmtChar::unescape(r#"\u{4000A}"#), None);
        assert_eq!(SmtChar::unescape(r#"\u{5000A}"#), None);
        assert_eq!(SmtChar::unescape(r#"\u{F000A}"#), None);
    }

    #[test]
    fn test_unescape_not_hex() {
        assert_eq!(SmtChar::unescape(r#"\u{G}"#), None);
        assert_eq!(SmtChar::unescape(r#"\u000H"#), None);
        assert_eq!(SmtChar::unescape(r#"\u{39v}"#), None);
        assert_eq!(SmtChar::unescape(r#"\u{J}"#), None);
    }

    #[test]
    fn test_unescape_invalid_braces() {
        assert_eq!(SmtChar::unescape(r#"\u{000A"#), None);
        assert_eq!(SmtChar::unescape(r#"\u000A}"#), None);
        assert_eq!(SmtChar::unescape(r#"\u{0{00A}"#), None);
        assert_eq!(SmtChar::unescape(r#"\u{0}00A}"#), None);
    }

    #[test]
    fn test_unescape_invalid_prefix() {
        assert_eq!(SmtChar::unescape(r#"u{000A}"#), None);
        assert_eq!(SmtChar::unescape(r#"\{000A}"#), None);
        assert_eq!(SmtChar::unescape(r#"\u000A}"#), None);
    }

    #[quickcheck]
    fn test_escape_unescape_inverse(c: u32) -> TestResult {
        if c > SMT_MAX_CODEPOINT {
            return TestResult::discard();
        }
        let smt_char = SmtChar::new(c);
        assert_eq!(smt_char, SmtChar::unescape(&smt_char.escape()).unwrap());
        TestResult::passed()
    }

    #[quickcheck]
    fn append_empty_right_neutral(s: SmtString) {
        let mut s1 = s.clone();
        s1.append(&SmtString::empty());
        assert_eq!(s1, s);
    }

    #[quickcheck]
    fn append_empty_left_neutral(s: SmtString) {
        let mut s1 = SmtString::empty();
        s1.append(&s);
        assert_eq!(s1, s);
    }

    #[quickcheck]
    fn reverse_reverse_inverse(s: SmtString) {
        let rev = s.reversed();
        assert_eq!(rev.reversed(), s);
    }

    #[test]
    fn test_parse_valid_strings_without_escaped() {
        assert_eq!(
            SmtString::parse("foo"),
            SmtString(vec![
                SmtChar('f' as u32),
                SmtChar('o' as u32),
                SmtChar('o' as u32),
            ])
        );
        assert_eq!(
            SmtString::parse("123!@#"),
            SmtString(vec![
                SmtChar('1' as u32),
                SmtChar('2' as u32),
                SmtChar('3' as u32),
                SmtChar('!' as u32),
                SmtChar('@' as u32),
                SmtChar('#' as u32)
            ])
        );
    }

    #[test]
    fn test_parse_valid_string_with_one_escape() {
        assert_eq!(
            SmtString::parse(r#"a\u0042c"#),
            SmtString(vec![
                SmtChar('a' as u32),
                SmtChar('B' as u32), // Unicode for 'B'
                SmtChar('c' as u32)
            ])
        );

        assert_eq!(
            SmtString::parse(r#"x\u{41}y"#),
            SmtString(vec![
                SmtChar('x' as u32),
                SmtChar('A' as u32), // Unicode for 'A'
                SmtChar('y' as u32)
            ])
        );

        assert_eq!(
            SmtString::parse(r#"\u{1F600}"#), // Unicode for 😀
            SmtString(vec![SmtChar(0x1F600)])
        );
    }

    #[test]
    fn test_parse_valid_string_with_multiple_escape() {
        assert_eq!(
            SmtString::parse(r#"abc\u0044\u{45}f"#),
            SmtString(vec![
                SmtChar('a' as u32),
                SmtChar('b' as u32),
                SmtChar('c' as u32),
                SmtChar('D' as u32), // Unicode for 'D'
                SmtChar('E' as u32), // Unicode for 'E'
                SmtChar('f' as u32),
            ])
        );

        assert_eq!(
            SmtString::parse(r#"\u{1F604} smile \u{1F60A}"#), // 😄 smile 😊
            SmtString(vec![
                SmtChar(0x1F604), // 😄
                SmtChar(' ' as u32),
                SmtChar('s' as u32),
                SmtChar('m' as u32),
                SmtChar('i' as u32),
                SmtChar('l' as u32),
                SmtChar('e' as u32),
                SmtChar(' ' as u32),
                SmtChar(0x1F60A), // 😊
            ])
        );
    }

    #[test]
    fn test_parse_invalid_escape_sequences() {
        // Missing closing brace
        let s = r#"\u{123"#;
        let expected = SmtString::new(s.chars().map(SmtChar::from).collect());
        assert_eq!(SmtString::parse(s), expected);

        // Non-hex character in escape sequence
        let s = r#"\u{12G3}"#;
        let expected = SmtString::new(s.chars().map(SmtChar::from).collect());
        assert_eq!(SmtString::parse(s), expected);

        // Escape sequence too long
        let s = r#"\u{123456}"#;
        let expected = SmtString::new(s.chars().map(SmtChar::from).collect());
        assert_eq!(SmtString::parse(s), expected);

        // Escape sequence without digits
        let s = r#"\u{}"#;
        let expected = SmtString::new(s.chars().map(SmtChar::from).collect());
        assert_eq!(SmtString::parse(s), expected);

        // Invalid escape sequence (SMT 2.5 style)
        let s = r#"\x1234"#;
        let expected = SmtString::new(s.chars().map(SmtChar::from).collect());
        assert_eq!(SmtString::parse(s), expected);

        // Unicode above allowed SMT max
        let s = r#"\u{110000}"#;
        let expected = SmtString::new(s.chars().map(SmtChar::from).collect());
        assert_eq!(SmtString::parse(s), expected);
    }

    #[quickcheck]
    fn test_print_parse_inverse(s: SmtString) {
        let s1 = s.to_string();
        let s2 = SmtString::parse(&s1);
        assert_eq!(s, s2);
    }

    #[quickcheck]
    fn index_of_empty_is_always_zero(s: SmtString) {
        assert_eq!(s.index_of(&SmtString::empty(), 0), Some(0));
    }

    #[quickcheck]
    fn contains_empty(s: SmtString) {
        assert!(s.contains(&SmtString::empty()))
    }

    #[test]
    fn test_replace_at_start() {
        let s: SmtString = "foobar".into();
        let from: SmtString = "foo".into();
        let to: SmtString = "bar".into();
        assert_eq!(s.replace(&from, &to), "barbar".into());
    }

    #[test]
    fn test_replace_at_end() {
        let s: SmtString = "foobar".into();
        let from: SmtString = "bar".into();
        let to: SmtString = "foo".into();
        assert_eq!(s.replace(&from, &to), "foofoo".into());
    }

    #[test]
    fn test_replace_no_match() {
        let s: SmtString = "abcdef".into();
        let from: SmtString = "xyz".into();
        let to: SmtString = "123".into();
        assert_eq!(s.replace(&from, &to), s);
    }

    #[test]
    fn test_replace_empty_from() {
        let s: SmtString = "abcdef".into();
        let from: SmtString = "".into();
        let to: SmtString = "XYZ".into();
        assert_eq!(s.replace(&from, &to), "XYZabcdef".into()); // Empty string is inserted at the beginning
    }

    #[test]
    fn test_replace_empty_to() {
        let s: SmtString = "abcdef".into();
        let from: SmtString = "cd".into();
        let to: SmtString = "".into();
        assert_eq!(s.replace(&from, &to), "abef".into()); // `cd` is removed
    }

    #[test]
    fn test_replace_full_string() {
        let s: SmtString = "abcdef".into();
        let from: SmtString = "abcdef".into();
        let to: SmtString = "xyz".into();
        assert_eq!(s.replace(&from, &to), "xyz".into());
    }

    #[test]
    fn test_replace_repeated_pattern() {
        let s: SmtString = "abcabcabc".into();
        let from: SmtString = "abc".into();
        let to: SmtString = "x".into();
        assert_eq!(s.replace(&from, &to), "xabcabc".into()); // Only first occurrence is replaced
    }

    #[test]
    fn test_replace_single_character() {
        let s: SmtString = "banana".into();
        let from: SmtString = "a".into();
        let to: SmtString = "o".into();
        assert_eq!(s.replace(&from, &to), "bonana".into()); // Only first 'a' is replaced
    }

    #[test]
    fn test_replace_all_basic() {
        let s: SmtString = "foobarbar".into();
        let from: SmtString = "bar".into();
        let to: SmtString = "foo".into();
        assert_eq!(s.replace_all(&from, &to), "foofoofoo".into());
    }

    #[test]
    fn test_replace_all_complete() {
        let s: SmtString = "abcabcabc".into();
        let from: SmtString = "abc".into();
        let to: SmtString = "xyz".into();
        assert_eq!(s.replace_all(&from, &to), "xyzxyzxyz".into());
    }

    #[test]
    fn test_replace_all_no_match() {
        let s: SmtString = "abcdef".into();
        let from: SmtString = "xyz".into();
        let to: SmtString = "123".into();
        assert_eq!(s.replace_all(&from, &to), "abcdef".into());
    }

    #[test]
    fn test_replace_all_empty_from() {
        let s: SmtString = "abcdef".into();
        let from: SmtString = "".into();
        let to: SmtString = "XYZ".into();
        assert_eq!(s.replace_all(&from, &to), s); // No-op
    }

    #[test]
    fn test_replace_all_empty_to() {
        let s: SmtString = "banana".into();
        let from: SmtString = "a".into();
        let to: SmtString = "".into();
        assert_eq!(s.replace_all(&from, &to), "bnn".into()); // All 'a's are removed
    }

    #[test]
    fn test_replace_all_full_string() {
        let s: SmtString = "abcdef".into();
        let from: SmtString = "abcdef".into();
        let to: SmtString = "xyz".into();
        assert_eq!(s.replace_all(&from, &to), "xyz".into());
    }

    #[test]
    fn test_replace_all_overlapping_occurrences() {
        let s: SmtString = "aaaa".into();
        let from: SmtString = "aa".into();
        let to: SmtString = "b".into();
        assert_eq!(s.replace_all(&from, &to), "bb".into()); // "aa" -> "b", then another "aa" -> "b"
    }

    #[test]
    fn test_replace_all_overlapping_occurrences_2() {
        let s: SmtString = "aaa".into();
        let from: SmtString = "aa".into();
        let to: SmtString = "b".into();
        assert_eq!(s.replace_all(&from, &to), "ba".into()); // "aa" -> "b", then another "aa" -> "b"
    }

    #[test]
    fn test_replace_all_overlapping_occurrences_3() {
        let s: SmtString = "aaa".into();
        let from: SmtString = "aa".into();
        let to: SmtString = "aaa".into();
        assert_eq!(s.replace_all(&from, &to), "aaaa".into());
    }
}
