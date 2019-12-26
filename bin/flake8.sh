#!/usr/bin/env bash

# Cause the script to exit if a single command fails
set -eo pipefail

# this stops git rev-parse from failing if we run this from the .git directory
builtin cd "$(dirname "${BASH_SOURCE:-$0}")"

ROOT="$(git rev-parse --show-toplevel)"
builtin cd "$ROOT" || exit 1

# Only fetch master since that's the branch we're diffing against.
git fetch upstream master

MERGEBASE="$(git merge-base upstream/master HEAD)"
if ! git diff --diff-filter=ACM --quiet --exit-code "$MERGEBASE" -- '*.py' &>/dev/null; then
    flake8 "$@" --filename=$(git diff --name-only --diff-filter=AM "$MERGEBASE" -- '*.py')
fi

if ! git diff --quiet &>/dev/null; then
    echo 'Reformatted changed files. Please review and stage the changes.' 1>&2
    echo 'Files updated:' 1>&2
    echo 1>&2

    git --no-pager diff --name-only 1>&2

    exit 1
fi
