<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) lapoule.dev -->

# Aerial orthophoto fixtures

Six 256² tiles of aerial orthophotography, as the server sends them (JPEG,
not re-encoded), for the encoders' quality measurements
(`tests/support/quality.rs`, `examples/quality-report.rs`) and the oracle
tests (`tests/gpu_oracle.rs`). They are the content a map viewer drapes at
mid altitudes, at Web Mercator zoom levels 12 to 16.

Source: IGN, *BD ORTHO®* (layer `ORTHOIMAGERY.ORTHOPHOTOS`), served by the
Géoplateforme WMTS `https://data.geopf.fr/wmts`, tile matrix set `PM`,
fetched on 23 Nov. 2026. Licence: **Licence Ouverte / Open Licence 2.0**
(Etalab), which allows reuse, redistribution and modification, commercial
included, with attribution: « IGN – BD ORTHO® ».

| File | Zoom | Column / row | What it exercises |
|---|---|---|---|
| `paris-urban.jpeg` | 16 | 33193 / 22545 | roofs, shadows, the Seine: hard edges at full resolution |
| `beauce-fields.jpeg` | 13 | 4135 / 2837 | field parcels: flat colours meeting along straight edges |
| `vosges-forest.jpeg` | 14 | 8508 / 5691 | forest canopy: fine dark texture, low contrast |
| `mont-blanc.jpeg` | 13 | 4252 / 2919 | snow and rock: near-white, near-grey blocks with sharp shadows |
| `brittany-coast.jpeg` | 14 | 8099 / 5652 | water, sand and houses: smooth gradients beside detail |
| `lyon-suburbs.jpeg` | 12 | 2103 / 1461 | a city from about 20 km: dense multicoloured detail in every block |
