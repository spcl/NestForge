---
name: unit-testing
description: >-
  How to write and run unit tests in this codebase. Use whenever adding, changing,
  or reviewing a test: "write a test for this", "add unit tests", "cover this with
  tests", "why is this test failing", "run the tests", "review these tests". Covers
  what a test must assert, how it is named and documented, the hard rules about
  never weakening a test to make it pass, and how to run the suite so it does not
  lie to you.
---

# Unit testing

A test states a PROPERTY the code must have. It is read by whoever breaks it six months from
now, so the name and the docstring have to say what broke, not what the code does.

## The hard rules

These override convenience, a deadline, and a red suite.

- **NEVER weaken a test to make it pass.** Not a loosened tolerance, not a narrowed input, not a
  removed assertion, not an `xfail`. A red test is information; a green one that no longer checks
  anything is a lie that survives you.
- **NEVER delete or skip a test to make a suite green.** If a test is genuinely obsolete, say so
  explicitly and explain WHY the property no longer exists, then delete it in its own change with
  that reasoning in the commit message. "It was failing" is not a reason.
- **A deliberately removed BEHAVIOUR takes its test with it -- and gets a new test for the new
  behaviour.** Deleting the old test alone is how a regression ships.
- **Warnings are errors.** A new warning during a test run is a failure, not noise.
- **Tests are consumers, not dead code.** A helper no test exercises is unverified. When you add
  a function, add the test that uses it the way production does.
- **Never use a norm-style aggregate error** (`norm(a - b)`) as a correctness gate: it hides a
  single catastrophically wrong element in a large array. Compare elementwise with an explicit
  tolerance.
- **Fixtures must match the real shapes** the frontend/production path produces. A fixture with a
  convenient shape tests a program that does not exist.
- **Match the INTERFACE, not the bug.** If the code under test is wrong, fix the code; do not
  encode its current wrong output as the expectation.
- **`monkeypatch` is fine in a unit test; it is never fine in the code under test.** If production
  code patches another module at runtime to make something work, fix the design (a parameter, a
  constructor argument, a hook the owner exposes) instead of testing around the patch.

## Writing one

**Name it after the property, as a sentence.** The name is what a failure report shows:

    def test_the_arm_language_is_the_identity_not_the_bodys_claim(...)
    def test_a_row_that_already_carries_an_identity_is_copied_untouched(...)
    def test_an_unregistered_packet_is_stable_and_warns(...)

not `test_language`, `test_copy_2`, `test_edge_case`.

**Docstring says WHY the property matters**, in one or two lines, and only when the name cannot
carry it. State the failure the test prevents, not the steps it performs:

    """The request body names its own language and an agent may put anything there, so the column
    an experiment groups by has to come from the arm."""

Never restate the assertions in prose. Never write "Tests that ...".

**One property per test.** A test that asserts four unrelated things reports one failure and
hides three.

**Tables go through `@pytest.mark.parametrize`**, one case per row, so each case fails by name.
Put the expected value in the table, not computed in the test body -- a test that recomputes the
implementation checks nothing.

**Assert on the observable contract**: the returned value, the written row, the raised exception
type and its message, the warning. Not on private helpers, call counts, or internal ordering the
contract does not promise.

**Test the boundary that actually bites**: the empty input, the single element, the value that
sorts first, the unregistered name, the second call (idempotence), the concurrent writer. A test
of the happy path alone is a smoke test.

**A failure message must locate the fault.** Prefer `assert got == want, got` over a bare assert
when the value is not in the expression.

## Running them

- Run the suite through the repo's wrapper (e.g. `tools/run_tests.sh`) rather than
  `python -m pytest` by path. A bare interpreter invocation misses PATH, `PKG_CONFIG_PATH` and
  `CPATH`, and the run then reports dozens of RED environment failures that are not the tree --
  a false verdict that costs hours.
- `--maxfail=20`: enough to see a pattern, not so many that the log is unreadable.
- A FULL suite belongs in a batch job on a compute node. A login node is for a targeted selection.
- Before believing a failure is yours, check whether it pre-exists your change (compare against
  `git show HEAD:<file>`); say explicitly which failures are pre-existing rather than fixing
  unrelated things inside your change.

## Reviewing one

Ask, in order: what property does this state? would it fail if the bug it names were
reintroduced? does it assert on the contract or on the implementation? is the fixture the shape
production produces? does a failure message say what is wrong? If a test cannot fail, it is
documentation, and worse documentation than a comment.
