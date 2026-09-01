<!--
SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
SPDX-FileContributor: Sahil Jhawar

SPDX-License-Identifier: Apache-2.0
-->

## Summary

<!-- Which satellite/instrument does this add or extend? What does the recipe produce (quantities, coordinate systems, cadence)? -->

- Mission/instrument:
- Data product(s):
- Related issue:

## Checklist

- [ ] Recipe lives under `el_paso/recipes/<mission>/` and follows the structure of an existing recipe (input handling, metadata, output to PRBEM-style standard names).
- [ ] `el_paso/recipes/<mission>/__init__.py` updated if a new module was added.
- [ ] `README.md` "Available processing scripts" list updated to include this recipe.
- [ ] Type hints added on new/changed code; `ty check` passes.
- [ ] A test covers the new recipe (marked `basic` or `visual` as appropriate), and `pytest` passes locally.
- [ ] If the recipe needs new reference/system test data, it's been added to the Zenodo dataset (or flagged in this PR for a maintainer to do so).
- [ ] If the recipe needs credentials/secrets to download data (e.g. `ESA_CLIENT_ID`), that's called out below.

## Notes for reviewers

<!-- Anything unusual about the data source, quirks in the raw files, units, or assumptions worth flagging. -->
