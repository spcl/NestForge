#!/usr/bin/env bash
# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
# Patch release vX.Y.Z -> vX.Y.(Z+1) of this repository's PyPI package, from a clean main equal to origin/main.
#   helpers/release_patch.sh                # wait for CI on HEAD, then release (PyPI and GitHub, from one build)
#   helpers/release_patch.sh --dry-run      # print the plan and build + check the tree; change nothing
#   helpers/release_patch.sh --github-only  # tag and GitHub release with the dists, no PyPI upload; the next plain
#                                           # run uploads exactly those dists first
# Steps: our packages' pins move to their newest PyPI releases (if any moved: a pins/* PR, stop); wait until every CI
# run on HEAD is green; bump the version, add a CHANGELOG section of the commits since the last tag (unless one is
# there), commit, tag, build, push, upload, create the GitHub release.
# Needs gh (authenticated), uv, jq, and for PyPI TWINE_USER and TWINE_TOKEN in the environment (never in a file).
# The same script lives in dace-relayout, dace-alloy, dace-fortran, tactics and fp-arena; keep the copies identical.
set -euo pipefail
ulimit -c 0
die() { echo "release_patch.sh: $*" >&2; exit 1; }
cd "$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
dry=0
github_only=0
case "${1:-}" in
    --dry-run) dry=1 ;;
    --github-only) github_only=1 ;;
    "") ;;
    *) echo "usage: $0 [--dry-run | --github-only]" >&2; exit 2 ;;
esac
twine() {
    [[ -n "${TWINE_USER:-}" && -n "${TWINE_TOKEN:-}" ]] || die "TWINE_USER and TWINE_TOKEN must be set (or pass --github-only)"
    TWINE_USERNAME="${TWINE_USER}" TWINE_PASSWORD="${TWINE_TOKEN}" uvx twine upload --non-interactive "$@"
}
on_pypi() { [[ "$(curl -s -o /dev/null -w '%{http_code}' "https://pypi.org/pypi/$1/$2/json")" != 404 ]]; }
project() { python3 -c 'import sys, tomllib; print(tomllib.load(open("pyproject.toml", "rb"))["project"][sys.argv[1]])' "$1"; }

[[ "$(git branch --show-current)" == main ]] || die "not on main"
[[ -z "$(git status --porcelain)" ]] || die "the working tree has uncommitted changes"
git fetch -q origin main --tags
head="$(git rev-parse HEAD)"
[[ "${head}" == "$(git rev-parse origin/main)" ]] || die "main is not origin/main; pull or push first"
slug="$(gh repo view --json nameWithOwner -q .nameWithOwner)"
pins_pr="$(gh pr list -R "${slug}" --state open --json headRefName,url -q '.[] | select(.headRefName | startswith("pins/")) | .url')"
[[ -z "${pins_pr}" ]] || die "a pin-bump PR is open (${pins_pr}): merge it once its CI is green, then rerun"

# Our packages pin released versions only: each `>=` bound rises to the newest PyPI version and a [tool.uv.sources] git
# pin of one of them is dropped. A bump is tested where it lands: it goes to a pins/* PR and the script stops.
bumps="$(python3 - "${dry}" <<'EOF'
import json
import pathlib
import re
import sys
import time
import tomllib
import urllib.error
import urllib.request

OURS = ["dace-extended", "dace-fortran", "fp-arena", "dace-relayout", "dace-alloy", "tactics", "hpcagent-bench"]
path = pathlib.Path("pyproject.toml")
text = path.read_text()
own = tomllib.loads(text)["project"]["name"]


def newest(name: str) -> str:
    """The newest PyPI version of ``name``, or "" while it has none (the query string skips the CDN's stale copy)."""
    try:
        with urllib.request.urlopen(f"https://pypi.org/pypi/{name}/json?fresh={time.time_ns()}") as reply:
            return json.load(reply)["info"]["version"]
    except urllib.error.HTTPError:
        return ""


for name in OURS:
    bound = re.compile(rf'(["\']{re.escape(name)}(?:\[[^\]]*\])?>=)([0-9][0-9.]*)')
    source = re.compile(rf'^{re.escape(name)} = \{{ git = "[^"]+", (?:rev|tag|branch) = "([^"]+)" \}}\n', re.MULTILINE)
    if name == own or not (bound.search(text) or source.search(text)) or not (version := newest(name)):
        continue
    for match in bound.finditer(text):
        if match.group(2) != version:
            print(f"{name}>={match.group(2)} -> >={version}")
    text = bound.sub(rf"\g<1>{version}", text)
    if match := source.search(text):
        print(f"{name} source git {match.group(1)[:12]} -> PyPI {version}")
        text = text[: match.start()] + text[match.end() :]
# A sources table left empty goes too, with the comment lines right above it
text = re.sub(r"(?:^#[^\n]*\n)*^\[tool\.uv\.sources\]\n(?=\n|\[|\Z)", "", text, flags=re.MULTILINE)
if sys.argv[1] != "1":
    path.write_text(text)
EOF
)"
if [[ -n "${bumps}" ]]; then
    printf '=== our packages: newest releases ===\n%s\n' "${bumps}"
    ((dry)) && { echo "dry run: a release first opens a PR with these bumps and stops for its CI"; exit 0; }
    [[ ! -f uv.lock ]] || uv lock -q --refresh
    branch="pins/$(date +%Y%m%d-%H%M%S)"
    git switch -q -c "${branch}"
    git commit -qam "Pin the newest releases of our packages" -m "${bumps}"
    git push -q origin "${branch}"
    pr="$(gh pr create -R "${slug}" --base main --head "${branch}" --title "Pin the newest releases of our packages" \
        --body "${bumps}")"
    git switch -q main
    echo "opened ${pr}; merge it once its CI is green, then rerun $0"
    exit 0
fi

name="$(project name)"
old="$(project version)"
[[ "${old}" =~ ^([0-9]+)\.([0-9]+)\.([0-9]+)$ ]] || die "version '${old}' is not X.Y.Z"
new="${BASH_REMATCH[1]}.${BASH_REMATCH[2]}.$((BASH_REMATCH[3] + 1))"
tag="v${new}"
work="$(mktemp -d "${TMPDIR:-/tmp}/release-patch.XXXXXX")"
trap 'rm -rf "${work}"' EXIT
# A --github-only release reaches PyPI on the next plain run: exactly the dists its GitHub release carries.
if ! on_pypi "${name}" "${old}" && gh release view "v${old}" -R "${slug}" >/dev/null 2>&1 && ! ((dry || github_only)); then
    mkdir "${work}/pending"
    gh release download "v${old}" -R "${slug}" -D "${work}/pending" -p '*.whl' -p '*.tar.gz'
    twine --skip-existing "${work}"/pending/*
    echo "uploaded the GitHub-only v${old}: https://pypi.org/project/${name}/${old}/"
fi
git rev-parse -q --verify "refs/tags/${tag}" >/dev/null && die "tag ${tag} exists already"
on_pypi "${name}" "${new}" && die "${name} ${new} is on PyPI already"
git rev-parse -q --verify "refs/tags/v${old}" >/dev/null || die "no tag v${old} for the current version"
commits="$(git log --no-merges --format='- %s (%h)' "v${old}..HEAD")"
[[ -n "${commits}" ]] || { echo "${name}: no commits since v${old}, nothing to release"; exit 0; }
printf '=== %s %s -> %s (%s, main %s) ===\n%s\n' "${name}" "${old}" "${new}" "${slug}" "${head:0:9}" "${commits}"
if ((dry)); then
    uv build -q -o "${work}/dist"
    uvx twine check --strict "${work}"/dist/*
    echo "dry run: nothing changed; rerun without --dry-run to release ${tag}"
    exit 0
fi
((github_only)) || [[ -n "${TWINE_USER:-}" && -n "${TWINE_TOKEN:-}" ]] || die "TWINE_USER and TWINE_TOKEN must be set"

# Only a commit every workflow passed on is released: wait while CI on HEAD runs, stop on the first failure.
while :; do
    runs="$(gh run list -R "${slug}" -c "${head}" -L 100 --json workflowName,status,conclusion)"
    bad="$(jq -r '.[] | select(.status == "completed" and .conclusion != "success" and .conclusion != "skipped")
        | "  \(.workflowName): \(.conclusion)"' <<<"${runs}")"
    [[ -z "${bad}" ]] || die "CI on ${head:0:9} failed:"$'\n'"${bad}"
    [[ "$(jq length <<<"${runs}")" -gt 0 && "$(jq '[.[] | select(.status != "completed")] | length' <<<"${runs}")" == 0 ]] &&
        break
    echo "waiting for CI on ${head:0:9} ($(date +%H:%M))"
    sleep 60
done

sed -i "0,/^version = \"${old}\"$/s//version = \"${new}\"/" pyproject.toml
[[ "$(project version)" == "${new}" ]] || die "could not set version ${new} in pyproject.toml"
[[ ! -f uv.lock ]] || uv lock -q
if ! grep -q "^## ${new} " CHANGELOG.md; then
    # The new section goes above the first existing "## " section
    SECTION="## ${new} ($(date +%F))"$'\n\n'"${commits}"$'\n' awk \
        '!done && /^## / { print ENVIRON["SECTION"]; done = 1 } { print } END { if (!done) print "\n" ENVIRON["SECTION"] }' \
        CHANGELOG.md >"${work}/cl"
    mv "${work}/cl" CHANGELOG.md
fi
notes="$(awk -v v="## ${new} " 'index($0, v) == 1 { on = 1; next } on && /^## / { exit } on' CHANGELOG.md)"
git commit -qam "Release ${tag}"
git tag -a "${tag}" -m "${name} ${new}"
uv build -q -o "${work}/dist"
uvx twine check --strict "${work}"/dist/*
git push -q origin main "${tag}"
((github_only)) || twine "${work}"/dist/*
gh release create "${tag}" -R "${slug}" --verify-tag --title "${name} ${new}" --notes "${notes}" --generate-notes \
    "${work}"/dist/*
echo "released ${name} ${new}: https://github.com/${slug}/releases/tag/${tag}"
((github_only)) && echo "GitHub only; rerun $0 to upload it to PyPI" || echo "https://pypi.org/project/${name}/${new}/"
