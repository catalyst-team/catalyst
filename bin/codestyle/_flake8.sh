#!/usr/bin/env bash

# Cause the script to exit if a single command fails
set -eo pipefail -v

# this stops git rev-parse from failing if we run this from the .git directory
builtin cd "$(dirname "${BASH_SOURCE:-$0}")"

ROOT="$(git rev-parse --show-toplevel)"
builtin cd "$ROOT" || exit 1


flake8 "$@"
