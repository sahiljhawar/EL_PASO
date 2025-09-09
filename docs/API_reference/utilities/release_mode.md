<!--
SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
SPDX-FileContributor: Bernhard Haas

SPDX-License-Identifier: Apache 2.0
-->

For publishing a data product, a release mode ensures reproducibility by associating the product with a specific Git commit hash. This commit hash serves as a unique identifier for the exact version of the EL-PASO code used to generate the data.

By linking the data to a precise point in the Git history, anyone can retrieve the identical code version and replicate the results, guaranteeing that the data product is traceable and verifiable.

Before release mode can be activated, the EL-PASO repository must not contain any changes which are not commited to the repository.

Under release mode, additional metadata of the author who processed the data is also storred as metadata.

::: el_paso.release_mode
