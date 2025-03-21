#!/bin/bash
set -e

echo "Checking formatting..."
cargo fmt --all -- --check --verbose

echo "Running Clippy..."
cargo clippy --features=full -- -D warnings --verbose

echo "Building..."
cargo build --features=full --verbose

echo "Testing..."
cargo test --features=full --verbose

echo "Cleaning..."
cargo clean