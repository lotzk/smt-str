# `smt-str`

[![CI](https://github.com/lotzk/smt-str/actions/workflows/build.yml/badge.svg)](https://github.com/lotzk/smt-str/actions/workflows/build.yml)
[![Crates.io](https://img.shields.io/crates/v/smt-str)](https://crates.io/crates/smt-str)
[![Docs.rs](https://docs.rs/smt-str/badge.svg)](https://docs.rs/smt-str)

**A Rust crate for working with SMT-LIB strings, regular expressions, and automata.**

`smt-str` provides data structures and utilities to parse, manipulate, and reason about strings and regular expressions following the semantics defined in the [SMT-LIB theory of strings](https://smtlib.cs.uiowa.edu/theories-UnicodeStrings.shtml). It also includes tools for compiling SMT-LIB regular expressions into NFAs.

## Installation

Install with cargo add:

```bash
cargo add smt-str --features=<features>
```

By default, no addtional features are enabled.
In that case, the crate only provides basic SMT string handling.
See below for available features.

### Feature Flags

This crate provides several feature flags to enable or disable specific functionality:

Feature | Enables
--- | ---
`regex`     | Regex construction
`automata`  | NFAs, regex-to-NFA compilation, visualization
`sampling`  | Random generation of strings from regular expressions or NFAs
`full`      | Enables all of the above

## Status

This crate is in early development.

## Related

- [SMT-LIB String Theory Spec](https://smtlib.cs.uiowa.edu/theories-UnicodeStrings.shtml)
