# Agent instructions

## Seer fix pull requests

For every fix PR that Seer creates or updates:

- Trace the failing request through the relevant code to the root cause. Name the key files
  and functions; do not infer the design from the error message alone.
- List the viable alternative designs, then explain why the chosen design is the smallest
  complete fix. State the main tradeoff; do not invent alternatives just to fill a list.
- Include short **Before** and **After** system diagrams in the PR description. Use fenced
  Mermaid flowcharts or sequence diagrams, or text diagrams if clearer. Show the failing and
  corrected paths, with the same component names and only the affected components and flows.
- Run a quick, focused local test of the changed behavior. Give the exact command, actual
  result, and what it verifies. If a test cannot run, say why; never claim an unrun test passed.
- Keep the PR description short: a few bullets for the code trace, alternatives, design choice,
  and test evidence, plus the two diagrams. Avoid repeated context, logs, and long narratives.
  Keep the description and diagrams current when the fix changes.
- Keep the PR as a draft for human review. Do not enable auto-merge or merge the PR.
