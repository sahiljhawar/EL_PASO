<!--
SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
SPDX-FileContributor: Sahil Jhawar

SPDX-License-Identifier: Apache-2.0
-->

## Summary

<!-- Dependency bump, CI/workflow change, build system tweak, or other non-functional chore. -->

- What changed:
- Why (security fix, new feature needed elsewhere, upstream deprecation, etc.):

## Checklist

- [ ] `pyproject.toml` version constraint updated consistently.
- [ ] `pytest -m basic` passes locally against the new dependency/tooling version.
- [ ] IRBEM build still succeeds if the change touches build hooks or Fortran toolchain.
- [ ] CI workflows (`.github/workflows/*.yml`) still pass; any workflow file changes tested via `workflow_dispatch` or a draft PR run.
- [ ] Changelog/release notes updated if this affects the published package.

## Notes for reviewers

<!-- Breaking changes in the dependency, migration steps, or anything that needs a version bump (`bumpver`). -->
