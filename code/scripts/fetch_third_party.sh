#!/usr/bin/env bash
# Fetch the two external dependencies at the commits every number in data/ was
# produced against. They are not vendored here: one is someone else's executor
# and the other is a benchmark, and both carry their own licences.
#
#   bash code/scripts/fetch_third_party.sh [destination]
#
# Default destination is third_party/ beside this repository's code/ directory,
# which is where the scripts look unless VJB_THIRD_PARTY says otherwise.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"     # .../code
DEST="${1:-${VJB_THIRD_PARTY:-$HERE/third_party}}"

JEV_COMMIT=1231850a0bf1a0c0341fe408ef1668dbbfdfac46
MINIWOB_COMMIT=33c3b4d

mkdir -p "$DEST"

if [ ! -d "$DEST/jev-ultrafast/.git" ]; then
  echo "fetching jev-ultrafast @ ${JEV_COMMIT:0:8}"
  git clone -q https://github.com/browser-use/jev-ultrafast.git "$DEST/jev-ultrafast"
fi
git -C "$DEST/jev-ultrafast" fetch -q origin "$JEV_COMMIT" 2>/dev/null || git -C "$DEST/jev-ultrafast" fetch -q
git -C "$DEST/jev-ultrafast" checkout -q "$JEV_COMMIT"

if [ ! -d "$DEST/miniwob-plusplus/.git" ]; then
  echo "fetching miniwob-plusplus @ $MINIWOB_COMMIT"
  git clone -q https://github.com/Farama-Foundation/miniwob-plusplus.git "$DEST/miniwob-plusplus"
fi
git -C "$DEST/miniwob-plusplus" fetch -q origin 2>/dev/null || true
git -C "$DEST/miniwob-plusplus" checkout -q "$MINIWOB_COMMIT"

# The runners want the task pages at one path, not nested three deep.
rm -rf "$DEST/miniwob-html"
cp -r "$DEST/miniwob-plusplus/miniwob/html" "$DEST/miniwob-html"

echo
echo "third_party ready at $DEST"
git -C "$DEST/jev-ultrafast"     log -1 --format='  jev-ultrafast     %h  %s'
git -C "$DEST/miniwob-plusplus"  log -1 --format='  miniwob-plusplus  %h  %s'
echo "  miniwob-html      $(ls "$DEST/miniwob-html/miniwob"/*.html | wc -l | tr -d ' ') task templates"
echo
echo "If you put it elsewhere, export VJB_THIRD_PARTY=$DEST before running the scripts."
