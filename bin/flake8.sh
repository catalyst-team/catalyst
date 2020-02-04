#!/usr/bin/env bash

# Cause the script to exit if a single command fails
set -eo pipefail -v

# this stops git rev-parse from failing if we run this from the .git directory
builtin cd "$(dirname "${BASH_SOURCE:-$0}")"

ROOT="$(git rev-parse --show-toplevel)"
builtin cd "$ROOT" || exit 1

git remote add 'upstream' 'https://github.com/catalyst-team/catalyst' || true

# Only fetch master since that's the branch we're diffing against.
git fetch upstream master

MERGEBASE="$(git merge-base upstream/master HEAD)"
if ! git diff --diff-filter=ACM --quiet --exit-code "$MERGEBASE" -- '*.py' &>/dev/null; then
    flake8 "$@" $(git diff --name-only --diff-filter=AM "$MERGEBASE" -- '*.py')
fi
