<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) lapoule.dev -->

# Imagery fixtures

Real imagery tiles, as the source serves them (not re-encoded), for the block
encoder's quality gate (`tests/texcomp_quality.rs`). Each is 512×512 and is cut
into four 256² tiles, the viewer's tile size.

All of them are NASA imagery served by NASA EOSDIS GIBS, which is in the public
domain (NASA's media usage guidelines). They were fetched on 21 Oct. 2026 from
the WMTS endpoint `https://gibs.earthdata.nasa.gov/wmts/epsg4326/best/`.

| File | Layer | Date | Tile matrix | Row / col | What it exercises |
|---|---|---|---|---|---|
| `alps.jpeg` | `BlueMarble_NextGeneration` | 2004-08-01 | `500m/7` | 19 / 83 | Alpine relief, snow edges |
| `nile.jpeg` | `BlueMarble_NextGeneration` | 2004-08-01 | `500m/7` | 26 / 93 | desert, the Nile's green ribbon, coast |
| `coast.jpeg` | `BlueMarble_ShadedRelief_Bathymetry` | 2004-08-01 | `500m/7` | 19 / 78 | long smooth gradients (bathymetry) |
| `modis.jpeg` | `MODIS_Terra_CorrectedReflectance_TrueColor` | 2024-07-15 | `250m/8` | 26 / 186 | clouds over land: high contrast, fine detail |
| `paris-night.png` | `VIIRS_Black_Marble` | 2016-01-01 | `500m/7` | 18 / 80 | night lights: bright points on black, the PNG source path |

Credit: NASA Earth Observatory (Blue Marble, Black Marble); NASA EOSDIS
LANCE/GIBS (MODIS).
