#!/usr/bin/env bash

# Cause the script to exit if a single command fails
set -eo pipefail -v

# this stops git rev-parse from failing if we run this from the .git directory
builtin cd "$(dirname "${BASH_SOURCE:-$0}")"

ROOT="$(git rev-parse --show-toplevel)"
builtin cd "$ROOT" || exit 1


YAPF_FLAGS=(
    '--style' "$ROOT/setup.cfg"
    '--recursive'
    '--parallel'
)

YAPF_EXCLUDES=(
    '--exclude' 'docker/*'
)

# Format specified files
format() {
    yapf --in-place "${YAPF_FLAGS[@]}" -- "$@"
}

# Format all files, and print the diff to stdout for travis.
format_all() {
    yapf --diff "${YAPF_FLAGS[@]}" "${YAPF_EXCLUDES[@]}" ./**/*.py
}

format_all_in_place() {
    yapf --in-place "${YAPF_FLAGS[@]}" "${YAPF_EXCLUDES[@]}" ./**/*.py
}

# This flag formats individual files. --files *must* be the first command line
# arg to use this option.
if [[ "$1" == '--files' ]]; then
    format "${@:2}"
    # If `--all` is passed, then any further arguments are ignored and the
    # entire python directory is formatted.
elif [[ "$1" == '--all' ]]; then
    format_all
elif [[ "$1" == '--all-in-place' ]]; then
    format_all_in_place
else
    # Format only the files that changed in last commit.
    exit 1
fi
