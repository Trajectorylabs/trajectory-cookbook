# Agent instructions

## Seer fix pull requests

For every fix PR that Seer creates or updates:

- Include **Before** and **After** system diagrams in the PR description. Use fenced Mermaid
  flowcharts or sequence diagrams. Use text diagrams if Mermaid cannot show the flow clearly.
- Show the affected components, interfaces, and data or control flow. Show the failing path
  before the fix and the corrected path after it. Use the same component names in both diagrams.
- Base the diagrams on the code. Name the relevant files or functions and explain the changed
  behavior. Keep the scope small; do not invent components or unrelated architecture changes.
- Keep the diagrams current when the fix changes. Include the test results and any limits that
  a reviewer must check.
- Keep the PR as a draft for human review. Do not enable auto-merge or merge the PR.
