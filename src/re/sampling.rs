use super::{deriv::Deriver, ReBuilder, Regex, SmtString};

/// Tries to sample a word that is accepted or not accepted by the regex.
/// The function will only return a word if it can be generated within the given depth.
pub fn try_sample(regex: &Regex, builder: &mut ReBuilder, max_depth: usize) -> Option<SmtString> {
    let mut w = SmtString::empty();
    let mut deriver = Deriver::default();

    let mut i = 0;
    let mut re = regex.clone();

    if re.nullable() {
        return Some(w);
    }

    while !re.nullable() && i < max_depth {
        let next = re.first().iter().filter_map(|c| c.choose()).next()?;
        w.push(next);
        re = deriver.deriv(&re, next, builder);
        i += 1;
    }

    if re.nullable() {
        Some(w)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {

    use quickcheck_macros::quickcheck;
    use smallvec::smallvec;

    use crate::alphabet::CharRange;

    use super::*;

    #[test]
    fn sample_const() {
        let mut builder = ReBuilder::default();
        let regex = builder.to_re("foo".into());

        assert_eq!(try_sample(&regex, &mut builder, 3), Some("foo".into()));
        assert_eq!(try_sample(&regex, &mut builder, 10), Some("foo".into()));
        assert_eq!(try_sample(&regex, &mut builder, 2), None);
    }

    #[test]
    fn sample_with_optional_characters() {
        let mut builder = ReBuilder::default();

        let o = builder.to_re("o".into());
        let fo = builder.to_re("fo".into());
        let bar = builder.to_re("bar".into());
        let o_or_bar = builder.union(smallvec![o, bar]);
        let regex = builder.concat(smallvec![fo, o_or_bar]);

        // Test matching "foo"
        assert!(try_sample(&regex, &mut builder, 3).is_some());
    }

    #[quickcheck]
    fn sample_with_character_range(range: CharRange) {
        let mut builder = ReBuilder::default();
        let regex = builder.range(range);

        assert!(try_sample(&regex, &mut builder, 1).is_some());
        // Test matching word within the class
        assert!(try_sample(&regex, &mut builder, 3).is_some());
    }

    #[quickcheck]
    fn sample_character_range(range: CharRange) {
        let mut builder = ReBuilder::default();
        let regex = builder.range(range);

        assert!(try_sample(&regex, &mut builder, 1).is_some());
        // Test matching word within the class
        assert!(try_sample(&regex, &mut builder, 3).is_some());
    }

    #[quickcheck]
    fn sample_character_range_pow(range: CharRange, n: u32) {
        let n = n % 100;
        let mut builder = ReBuilder::default();
        let regex = builder.range(range);
        let regex = builder.pow(regex, n as u32);

        for i in 0..n {
            assert!(try_sample(&regex, &mut builder, i as usize).is_none());
        }
        assert!(try_sample(&regex, &mut builder, n as usize).is_some());
    }

    #[quickcheck]
    fn sample_alternatives(rs: Vec<CharRange>) {
        let n = rs.len();
        let mut builder = ReBuilder::default();
        let rs = rs.into_iter().map(|r| builder.range(r)).collect();
        let regex = builder.union(rs);

        if n > 0 {
            assert!(try_sample(&regex, &mut builder, 1).is_some());
        } else {
            assert!(try_sample(&regex, &mut builder, 10).is_none());
        }
    }

    #[test]
    fn sampling_alternatives_bug() {
        let rs = vec![
            //CharRange::new(76887, 179877),
            //CharRange::new(142686, 186533),
            //CharRange::new(51684, 146039),
            CharRange::new(2, 5),
            CharRange::new(3, 6),
            CharRange::new(1, 4),
        ];

        //  CharRange  CharRange { start: SmtChar(51684), end: SmtChar(146039) }])]
        let n = rs.len();
        let mut builder = ReBuilder::default();
        let rs = rs.into_iter().map(|r| builder.range(r)).collect();
        let regex = builder.union(rs);

        if n > 0 {
            assert!(try_sample(&regex, &mut builder, 1).is_some());
        } else {
            assert!(try_sample(&regex, &mut builder, 10).is_none());
        }
    }

    #[quickcheck]
    fn sample_opt(r: CharRange) {
        let mut builder = ReBuilder::default();
        let r = builder.range(r);
        let regex = builder.opt(r);

        assert!(try_sample(&regex, &mut builder, 0).is_some());
        assert!(try_sample(&regex, &mut builder, 1).is_some());
    }

    #[test]
    fn sample_empty_string() {
        let mut builder = ReBuilder::default();
        let regex = builder.epsilon();

        assert!(try_sample(&regex, &mut builder, 0).is_some());
    }

    #[test]
    fn sample_empty_regex() {
        let mut builder = ReBuilder::default();
        let regex = builder.none();

        assert!(try_sample(&regex, &mut builder, 0).is_none());
        assert!(try_sample(&regex, &mut builder, 20).is_none());
    }

    #[test]
    fn sample_all() {
        let mut builder = ReBuilder::default();
        let regex = builder.all();

        assert!(try_sample(&regex, &mut builder, 0).is_some());
        assert!(try_sample(&regex, &mut builder, 20).is_some());
    }

    #[test]
    fn sample_any() {
        let mut builder = ReBuilder::default();
        let regex = builder.any_char();

        assert!(try_sample(&regex, &mut builder, 0).is_none());
        assert!(try_sample(&regex, &mut builder, 20).is_some());
    }
}
