pub mod alphabet;
#[cfg(feature = "automata")]
pub mod automata;
pub mod re;

use std::{fmt::Display, ops::Index};

/// A unicode character in the range 0x0000 to 0x2FFFF.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SmtChar(u32);

/// The maximum unicode character.
pub const SMT_MAX_CODEPOINT: u32 = 0x2FFFF;

/// The minimum unicode character.
pub const SMT_MIN_CODEPOINT: u32 = 0x0000;

impl SmtChar {
    /// The maximum `SmtChar`.
    pub const MAX: Self = Self(SMT_MAX_CODEPOINT);

    /// The minimum `SmtChar`.
    pub const MIN: Self = Self(SMT_MIN_CODEPOINT);

    /// Create a new `SmtChar` from a `char`.
    /// Panics if the `char` is not in the range 0x0000 to 0x2FFFF.
    pub fn new(c: char) -> Self {
        let code = c as u32;
        assert!(code <= 0x2FFFF, "character out of range: {}", c);
        SmtChar(code)
    }

    /// Get the `char` representation of this `SmtChar`.
    ///
    /// # Examples
    /// ```
    /// use smt_strings::SmtChar;
    /// let c = SmtChar::new('a');
    /// assert_eq!(c.as_char(), 'a');
    ///
    pub fn as_char(self) -> char {
        std::char::from_u32(self.0).unwrap()
    }

    /// Returns the next `SmtChar` in the range 0x0000 to 0x2FFFF.
    /// Returns `None` if this `SmtChar` is the maximum `SmtChar`.
    ///
    /// # Examples
    /// ```
    /// use smt_strings::SmtChar;
    /// let c = SmtChar::new('a');
    /// assert_eq!(c.next(), Some(SmtChar::new('b')));
    /// assert_eq!(SmtChar::MAX.next(), None);
    /// ```
    pub fn next(self) -> Option<Self> {
        if self.0 == SMT_MAX_CODEPOINT {
            None
        } else {
            Some(SmtChar(self.0 + 1))
        }
    }

    /// Like `next`, but instead of returning `None` when this `SmtChar` is the maximum `SmtChar`, it returns the maximum `SmtChar`.
    ///
    /// # Examples
    /// ```
    /// use smt_strings::SmtChar;
    /// let c = SmtChar::new('a');
    /// assert_eq!(c.saturating_next(), SmtChar::new('b'));
    /// assert_eq!(SmtChar::MAX.saturating_next(), SmtChar::MAX);
    /// ```
    pub fn saturating_next(self) -> Self {
        if self.0 == SMT_MAX_CODEPOINT {
            SmtChar::MAX
        } else {
            SmtChar(self.0 + 1)
        }
    }

    /// Returns the previous `SmtChar` in the range 0x0000 to 0x2FFFF.
    /// Returns `None` if this `SmtChar` is the minimum `SmtChar`.
    ///
    /// # Examples
    /// ```
    /// use smt_strings::SmtChar;
    /// let c = SmtChar::new('b');
    /// assert_eq!(c.prev(), Some(SmtChar::new('a')));
    /// assert_eq!(SmtChar::MIN.prev(), None);
    /// ```
    pub fn prev(self) -> Option<Self> {
        if self.0 == SMT_MIN_CODEPOINT {
            None
        } else {
            Some(SmtChar(self.0 - 1))
        }
    }

    /// Like `prev`, but instead of returning `None` when this `SmtChar` is the minimum `SmtChar`, it returns the minimum `SmtChar`.
    ///
    /// # Examples
    /// ```
    /// use smt_strings::SmtChar;
    /// let c = SmtChar::new('b');
    /// assert_eq!(c.saturating_prev(), SmtChar::new('a'));
    /// assert_eq!(SmtChar::MIN.saturating_prev(), SmtChar::MIN);
    /// ```
    pub fn saturating_prev(self) -> Self {
        if self.0 == SMT_MIN_CODEPOINT {
            SmtChar::MIN
        } else {
            SmtChar(self.0 - 1)
        }
    }

    /// Returns `true` if this `SmtChar` is a printable ASCII character.
    /// Printable ASCII characters are in the range 0x00020 to 0x0007E.
    ///
    /// # Examples
    /// ```
    /// use smt_strings::SmtChar;
    /// assert!(SmtChar::new('a').printable());
    /// assert!(!SmtChar::new('\n').printable());
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
    /// use smt_strings::SmtChar;
    /// assert_eq!(SmtChar::new('a').escape(), r#"\u{61}"#);
    /// assert_eq!(SmtChar::new('\n').escape(), r#"\u{A}"#);
    /// assert_eq!(SmtChar::new('🦀').escape(), r#"\u{1F980}"#);
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
    /// - `\u{D'DDDD}`
    ///
    /// where `D` is a hexadecimal digit and `D'` is a hexadecimal digit in the range 0 to 2.
    /// The function returns `None` if the input string is not a valid escaped character.
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

impl From<char> for SmtChar {
    fn from(c: char) -> Self {
        SmtChar::new(c)
    }
}

impl From<u32> for SmtChar {
    fn from(c: u32) -> Self {
        assert!(c <= SMT_MAX_CODEPOINT, "character out of range: {}", c);
        SmtChar(c)
    }
}

impl Display for SmtChar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.printable() {
            write!(f, "{}", self.as_char())
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
            self.current = self.current.next()?;
            Some(c)
        } else {
            None
        }
    }
}

/// An SMT-LIB string.
/// An SMT-LIB string is a sequence of [SmtChar] characters.
/// The empty string is represented by an empty vector.
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

    pub fn parse(input: &str) -> Option<Self> {
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
                    return None; // Invalid escape sequence
                }
            } else {
                // Process regular character
                result.push(SmtChar(c as u32));
                chars.next(); // Consume character
            }
        }

        Some(SmtString(result))
    }

    /// Returns whether this string is empty.
    ///
    /// # Examples
    /// ```
    /// use smt_strings::{SmtString};
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
    /// use smt_strings::{SmtString};
    /// assert_eq!(SmtString::empty().len(), 0);
    /// let s: SmtString = "foo".into();
    /// assert_eq!(s.len(), 3);
    /// ```
    ///
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Appends the characters of `other` to this string.
    /// The characters are appended in order.
    ///
    /// # Examples
    /// ```
    /// use smt_strings::{SmtString, SmtChar};
    /// let mut s: SmtString = "foo".into();
    /// let other: SmtString = "bar".into();
    /// s.append(&other);
    /// let mut iter = s.iter();
    /// assert_eq!(iter.next(), Some(&SmtChar::new('f')));
    /// assert_eq!(iter.next(), Some(&SmtChar::new('o')));
    /// assert_eq!(iter.next(), Some(&SmtChar::new('o')));
    /// assert_eq!(iter.next(), Some(&SmtChar::new('b')));
    /// assert_eq!(iter.next(), Some(&SmtChar::new('a')));
    /// assert_eq!(iter.next(), Some(&SmtChar::new('r')));
    /// assert_eq!(iter.next(), None);
    /// ```
    pub fn append(&mut self, other: &SmtString) {
        self.0.extend(other.0.iter().copied());
    }

    /// Pushes a character to the end of this string.
    ///
    /// # Examples
    /// ```
    /// use smt_strings::{SmtString, SmtChar};
    /// let mut s = SmtString::empty();
    /// s.push(SmtChar::new('f'));
    /// s.push(SmtChar::new('o'));
    /// s.push(SmtChar::new('o'));
    /// assert_eq!(s, SmtString::from("foo"));  
    /// ```
    pub fn push(&mut self, c: SmtChar) {
        self.0.push(c);
    }

    /// Concatenates this string with `other` and returns the result.
    /// This is a convenience method that does not modify this string.
    /// The characters of `other` are appended to the characters of this string, see [append].
    ///
    /// # Examples
    /// ```
    /// use smt_strings::{SmtString};
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

    /// Returns the first character of this string, if it is not empty.
    /// Returns `None` if this string is empty.
    ///
    /// # Examples
    /// ```
    /// use smt_strings::{SmtString, SmtChar};
    /// let s: SmtString = "foo".into();
    /// assert_eq!(s.first(), Some(SmtChar::new('f')));
    /// assert_eq!(SmtString::empty().first(), None);
    /// ```
    pub fn first(&self) -> Option<SmtChar> {
        self.0.first().copied()
    }

    /// Returns the prefix of length `n` of this string.
    /// If `n` is greater than the length of this string, the entire string is returned.
    /// If `n` is zero, the empty string is returned.
    ///
    /// # Examples
    /// ```
    /// use smt_strings::{SmtString, SmtChar};
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
    /// use smt_strings::{SmtString, SmtChar};
    /// let s = SmtString::from("foo");
    /// assert_eq!(s.drop(2), SmtString::from("o"));
    /// assert_eq!(s.drop(10), SmtString::empty());
    /// assert_eq!(s.drop(0), s);
    /// ```
    pub fn drop(&self, n: usize) -> SmtString {
        SmtString(self.0.iter().copied().skip(n).collect())
    }

    /// Returns the reverse of this string.
    ///
    /// # Examples
    /// ```
    /// use smt_strings::{SmtString, SmtChar};
    /// let s: SmtString = "foo".into();
    /// let rev = s.reversed();
    /// let mut iter = rev.iter();
    /// assert_eq!(iter.next(), Some(&SmtChar::new('o')));
    /// assert_eq!(iter.next(), Some(&SmtChar::new('o')));
    /// assert_eq!(iter.next(), Some(&SmtChar::new('f')));
    /// assert_eq!(iter.next(), None);
    /// ```
    pub fn reversed(&self) -> Self {
        SmtString(self.0.iter().rev().copied().collect())
    }

    /// Returns an iterator over the characters of this string.
    pub fn iter(&self) -> std::slice::Iter<SmtChar> {
        self.0.iter()
    }
}

impl FromIterator<SmtChar> for SmtString {
    fn from_iter<I: IntoIterator<Item = SmtChar>>(iter: I) -> Self {
        SmtString(iter.into_iter().collect())
    }
}

impl From<&str> for SmtString {
    fn from(s: &str) -> Self {
        SmtString(s.chars().map(SmtChar::new).collect())
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

#[cfg(test)]
mod tests {

    use quickcheck::{Arbitrary, TestResult};
    use quickcheck_macros::quickcheck;

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

    use super::*;

    #[quickcheck]
    fn next_prev_inverse(s: SmtChar) -> TestResult {
        if s == SmtChar::MAX {
            return TestResult::discard();
        }
        let next = s.next().unwrap();
        assert_eq!(next.prev(), Some(s));
        TestResult::passed()
    }

    #[quickcheck]
    fn prev_next_inverse(s: SmtChar) -> TestResult {
        if s == SmtChar::MIN {
            return TestResult::discard();
        }
        let prev = s.prev().unwrap();
        assert_eq!(prev.next(), Some(s));
        TestResult::passed()
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
    fn test_escape_unescape_inverse(c: char) -> TestResult {
        if c as u32 > SMT_MAX_CODEPOINT {
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
            Some(SmtString(vec![
                SmtChar('f' as u32),
                SmtChar('o' as u32),
                SmtChar('o' as u32),
            ]))
        );
        assert_eq!(
            SmtString::parse("123!@#"),
            Some(SmtString(vec![
                SmtChar('1' as u32),
                SmtChar('2' as u32),
                SmtChar('3' as u32),
                SmtChar('!' as u32),
                SmtChar('@' as u32),
                SmtChar('#' as u32)
            ]))
        );
    }

    #[test]
    fn test_parse_valid_string_with_one_escape() {
        assert_eq!(
            SmtString::parse(r#"a\u0042c"#),
            Some(SmtString(vec![
                SmtChar('a' as u32),
                SmtChar('B' as u32), // Unicode for 'B'
                SmtChar('c' as u32)
            ]))
        );

        assert_eq!(
            SmtString::parse(r#"x\u{41}y"#),
            Some(SmtString(vec![
                SmtChar('x' as u32),
                SmtChar('A' as u32), // Unicode for 'A'
                SmtChar('y' as u32)
            ]))
        );

        assert_eq!(
            SmtString::parse(r#"\u{1F600}"#), // Unicode for 😀
            Some(SmtString(vec![SmtChar(0x1F600)]))
        );
    }

    #[test]
    fn test_parse_valid_string_with_multiple_escape() {
        assert_eq!(
            SmtString::parse(r#"abc\u0044\u{45}f"#),
            Some(SmtString(vec![
                SmtChar('a' as u32),
                SmtChar('b' as u32),
                SmtChar('c' as u32),
                SmtChar('D' as u32), // Unicode for 'D'
                SmtChar('E' as u32), // Unicode for 'E'
                SmtChar('f' as u32),
            ]))
        );

        assert_eq!(
            SmtString::parse(r#"\u{1F604} smile \u{1F60A}"#), // 😄 smile 😊
            Some(SmtString(vec![
                SmtChar(0x1F604), // 😄
                SmtChar(' ' as u32),
                SmtChar('s' as u32),
                SmtChar('m' as u32),
                SmtChar('i' as u32),
                SmtChar('l' as u32),
                SmtChar('e' as u32),
                SmtChar(' ' as u32),
                SmtChar(0x1F60A), // 😊
            ]))
        );
    }

    #[test]
    fn test_parse_invalid_escape_sequences() {
        // Missing closing brace
        assert_eq!(SmtString::parse(r#"\u{123"#), None);

        // Non-hex character in escape sequence
        assert_eq!(SmtString::parse(r#"\u{12G3}"#), None);

        // Escape sequence too long
        assert_eq!(SmtString::parse(r#"\u{123456}"#), None);

        // Escape sequence without digits
        assert_eq!(SmtString::parse(r#"\u{}"#), None);

        // Invalid escape sequence (SMT 2.5 style)
        assert_eq!(SmtString::parse(r#"\x1234"#), None);

        // Unicode above allowed SMT max
        assert_eq!(SmtString::parse(r#"\u{110000}"#), None); // Invalid Unicode point
    }

    #[test]
    fn test_parse_invalid_escape_sequence_with_longer_string() {
        // Mix of valid and invalid escape sequences in a longer string
        assert_eq!(SmtString::parse("Hello \\u{1F60G} World"), None); // Invalid hex digit 'G'
        assert_eq!(SmtString::parse("Test \\u{} fail"), None); // Empty escape sequence
        assert_eq!(SmtString::parse("Mix \\u{41} and \\u{XYZ}"), None); // Invalid escape 'XYZ'
    }

    #[quickcheck]
    fn test_print_parse_inverse(s: SmtString) {
        let s1 = s.to_string();
        let s2 = SmtString::parse(&s1).unwrap();
        assert_eq!(s, s2);
    }
}
