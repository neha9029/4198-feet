# 4,198 Feet

*4,198 Feet* is a generative tessellation of the Great Salt Lake in which the water lost below the healthy elevation is hidden in daylight. The hidden cells are spectral metamers of the exposed lakebed under CIE D65 and diverge coherently under tungsten light. The fixed seed is **1931**.

## Data and method

The Phase 1 shorelines are reconstructed from Landsat water-occurrence frequency calibrated to Casey–Root elevation–area relationships; they are **not surveyed elevation contours**. The water-frequency source is [JRC Global Surface Water v1.5 (1984–2024)](https://global-surface-water.appspot.com/download), tile `120W_50N`, distributed under the Copernicus terms with map attribution “Source: EC JRC/Google.” The calibration table is the public [`bathymetry_casey_root_ngvd29.csv`](https://assets.super.so/936849e5-ca7c-48b3-b907-91fc48c4b0f6/files/136f9cab-844d-45f7-bcfa-d28d500ffed3/bathymetry_casey_root_ngvd29.csv) published on the [Great Salt Lake Accounting datasets page](https://gslaccounting.org/great-salt-lake-accounting-data-datasets); both the table and the gauge levels used here are in **NGVD29**.

Occurrence is kept at its native 0.00025-degree grid in `data/gsw_occurrence_gsl.tif`, then reprojected without downsampling to a 30 m UTM Zone 12N measurement grid. Bilinear reprojection treats occurrence as a continuous frequency, while the diagnostic extent and transition classes use nearest-neighbour reprojection. A four-neighbour connected mask seeded in both lake arms is thresholded until its pixel area matches the table’s basin-plan area at 4,198.0 ft and 4,189.6 ft. The band between those masks is the lost-water class; its occurrence ranks are mapped monotonically to 0–8.4 ft of pseudo-depth.

## Limitations

The accepted masks remain an occurrence-based reconstruction, not a surveyed contour. The approved C2-C7 regions are excluded as the US Magnesium Stansbury Basin solar-evaporation system documented by [Utah Geological Survey RI-171](https://ugspub.nr.utah.gov/publications/reports_of_investigations/RI-171.pdf). C1 is excluded as the probable Compass Minerals Clyman Bay west-pond complex after all 2.615 km² of its detected region detached from open north-arm water when one-pixel (30 m) dike gaps were removed. Compass pond locations and operating context are documented in [USACE public notice SPK-2007-00121](https://www.spk.usace.army.mil/Portals/12/documents/regulatory/eis/200700121-eis/200700121-PN-posted-07-09-2013.pdf); the whole-complex construction is checked against Compass Minerals' [SEC technical report figures 13-1 through 13-3](https://www.sec.gov/Archives/edgar/data/1227654/000122765421000300/cmp-20210930xxex961ogdenpo.htm).

Diked or managed water is excluded regardless of operator because its water level does not track the Great Salt Lake. The same reproducible rule was first applied to every loose-screen candidate: remove one-pixel (30 m) connections to seeded open-lake water in both arms and classify the candidate as managed when at least 90% of its area detaches and its boundary is rectilinear or its occurrence is near-constant. A second whole-complex step then assigns every accepted-water pixel in the Compass east, Compass west, and official TIGERweb Willard Reservoir AOIs to `managed`, repeating calibration until no water can re-enter an AOI. The resulting persistent `managed` class is 383.482 km². Managed pixels use the land spectral family and are always painted—never water and never hidden lakebed.

Thresholds were recalibrated with the managed class forbidden but with the original four-neighbour lake connectivity retained; the 60 m erosion test is diagnostic only and is not applied globally. The supplied West Desert reference point (NAD83 / UTM 12N 332,885 E, 4,578,003 N) is inside the existing crop, so no westward extension was necessary.

The reprojected crop contains 580.085 km² of source no-data along its outer edge. None is enclosed by the accepted 4,198 ft mask or lies within 1 km of either accepted shoreline. Both arms remain fitted jointly. Separate-arm diagnostics would change 15.296% of the healthy-mask union and 9.401% of the current-mask union; these alternate masks are not adopted.

The diagnostic criteria, candidate coordinates, C1 test, uniform N-candidate decisions, whole-complex AOIs, and recalibrations are recorded in `out/phase1/phase1_checks_report.json`, `out/phase1/phase1_managed_report.json`, `out/phase1/managed_decision_report.json`, and `out/phase1/managed_aoi_report.json`.

## Phase 2 method

Phase 2 uses a 25,000-point, 49,866-triangle Delaunay texture mesh with two Lloyd relaxation passes and 132 unique frame-boundary points. Site density is uniform and independent of water class; the hidden 4,198 ft shoreline has no anchor or density boost. Trials at 15k, 25k, and 30k points all pass the density-camouflage test. At the selected 25k density, lost-water and land triangle densities within 5 km of the healthy shoreline differ by 3.06%, inside the ±10% requirement. Each triangle owns a bounded 41-sample reflectance spectrum rather than a display colour: land, managed areas, and lost-water triangles share one muted earth/sage `smooth_field` family with deterministic per-cell variation. With relief enabled, adjacent land-family facets have median D65 ΔE76 4.405. The lost-to-land and land-to-land adjacency distributions remain indistinguishable (two-sample KS p = 0.624; medians differ by 3.73%), so the reconstructed healthy shoreline is not traceable in daylight.

The arms are separated by the actual Lucin Cutoff railroad geometry, not a latitude. The 15-vertex working polyline was simplified at 20 m from current [OpenStreetMap railway data](https://api.openstreetmap.org/api/0.6/map.json?bbox=-113.25,41.15,-112.30,41.30), cross-referenced against the [OpenStreetMap Utah railway inventory](https://wiki.openstreetmap.org/wiki/Utah/Railroads), and is used under the ODbL with [OpenStreetMap attribution](https://www.openstreetmap.org/copyright). Pixels north of the causeway and west of Promontory Point are Gunnison Bay; the water east of Promontory Point—including Bear River Bay and Farmington Bay—is connected to the south arm, consistent with the [Utah Geological Survey's causeway description](https://geology.utah.gov/popular/great-salt-lake/a-lake-divided/). The resulting lost-water allocation is 437.621 km² / 833 triangles north and 827.077 km² / 1,526 triangles south. A regression test asserts that cells around the Bear River Bay reference centroid remain south-arm cells.

For every lost-water triangle, SLSQP searches a smooth projected spectral basis whose perturbations are in the exact D65 XYZ nullspace of that triangle's own base spectrum. Under Illuminant A, the north arm sweeps a*b* angles 0°, +15°, and −15° while the south sweeps −180°, −175°, −170°, −165°, and −150°; each angle is tested with second-difference weights 20, 50, and 200. The objective is `−10·depth_strength·projection + 2·||orthogonal||² + s·||D2 r||²`, with zero L* weight and five seeded restarts projected from `clip(r0 + N(0, 0.15))`. Depth strength is `0.6 + 0.4 × normalised depth` and weights the projection rather than capping it. The fixed seed is 1931, spectra remain in [0.015, 0.96], and the final D65 image is pixel-identical to the unmodified baseline.

Variant B supplies fixed, non-metameric current-water spectra at L* about 45 with per-cell variation: adjacent same-arm water facets have median D65 ΔE76 1.989. North water is reduced to 75% of its former chroma and rotated 10° toward red, producing median Lab (45.14, 32.64, −8.96) under D65 and (47.73, 38.01, 10.31) under 2856 K. South water remains blue-green. The corrected full solve chooses north −15° / s = 20 and south −175° / s = 20. With terrain modulation included, under 2856 K the north median/minimum ΔE76 is 30.70/20.98 with coherence 0.9993; the south is 17.95/5.56 with coherence 0.9735. Both arms have zero solver failures. The solve uses eight spawned worker processes with per-cell seeds derived from seed 1931, arm offset, and triangle index. A 256-cell sample in each arm reproduces bit-for-bit; the complete sweep, solve, and repeat check takes 84.47 seconds on the recorded run.

The production frame is north-up at 16:9, with the 4,198 ft extent occupying exactly 92% of its height. The valid source raster supplies the central land tessellation; where the side extensions exceed that raster, they continue the same land spectral family. Those out-of-data areas are **texture only**: they do not represent observed terrain and contain no metamers or water classes.

## Terrain relief

The optional relief pass uses the public, no-key [AWS Terrain Tiles dataset](https://registry.opendata.aws/terrain-tiles/) from `s3://elevation-tiles-prod` in [Terrarium PNG format](https://github.com/tilezen/joerd/blob/master/docs/formats.md), accessed 2026-10-13. The 144 zoom-11 tiles covering the full film frame report principally USGS NED/3DEP inputs in their object metadata, with SRTM/GMTED fallback sources on some tile margins. United States elevation inputs are public domain; the requested credit is “3DEP data courtesy of the U.S. Geological Survey,” as documented in the [Tilezen attribution notes](https://github.com/tilezen/joerd/blob/master/docs/attribution.md).

The DEM is reprojected to the 30 m UTM 12N film grid, Gaussian-smoothed at 300 m, and lit with a 315° azimuth / 45° altitude soft hillshade. Land, managed, and lost-water base spectra are scaled by 0.92–1.08; current-water spectra are unchanged. Each lost-water cell is then solved from its own modulated base, retaining exact D65 XYZ equality. The relief pass is accepted: the lost-to-land / land-to-land adjacency KS p-value is 0.624, their medians differ by 3.73%, the density difference remains 3.06%, and rendered D65 identity remains exactly zero pixel difference.

## Phase 3 outputs

Phase 3 renders 4000×2250 supersampled PNG stills and standalone SVG panels under D65 and 2856 K, plus a side-by-side diptych with the captions “D65 · daylight” and “2856 K · tungsten.” It also exports north-up 3840×2160 frames for the film and native-scale 600×600 crops around the Bear River Bay reference centroid. Generated files and their SHA-256 hashes are recorded in `out/stills/phase3_report.json`.

## Phase 4 preview

The timeboxed preview covers global film time 0:40–0:50 at 1920×1080 and 24 fps. Each frame mixes equal-luminance D65 and Illuminant A SPDs, applies the full-film smootherstep transition, and adapts from the mixture's own white using the established CAT02 degree. Because this is the first ten seconds of the specified 70-second transition, the Illuminant A mixture reaches 2.326% at 0:50.

Two ten-second encodes are exported: H.264 CRF 16 in `yuv420p` with channel-offset 8×8 ordered dither, and H.265 CRF 16 in `yuv420p10le` from a 16-bit RGB source. A 100% land-only midpoint crop shows no visible contour banding in either encode; the 8-bit version has slightly more fine dither/compression grain, while the 10-bit version is cleaner. Recorded mean image-generation time is 0.0703 seconds/frame at 1080p and 0.1455 seconds/frame at 4K, excluding the one-time triangle-ID raster and codec write time. Machine-readable details are in `out/film/preview_40_50/preview_report.json`.

## Run Phase 1

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python main.py phase1
```

Large and generated data are ignored by Git. The command fetches the three public JRC diagnostic layers and the two public bathymetry CSV files into `data/raw/`, writes native crops and UTM working rasters into `data/`, and writes the Phase 1 report and PNG diagnostics to `out/phase1/`.

Run the accepted-mask checks without changing those masks with:

```bash
.venv/bin/python main.py phase1-checks
```

Apply the approved managed exclusions, rerun calibration, and screen the Compass operating areas with:

```bash
.venv/bin/python main.py phase1-managed
```

Apply the uniform N1-N19 decision and run Phase 2 with:

```bash
.venv/bin/python main.py phase1-decision
.venv/bin/python main.py phase1-aoi
.venv/bin/python main.py phase2-density
.venv/bin/python main.py phase2-relief
.venv/bin/python main.py phase2
.venv/bin/python main.py phase3
.venv/bin/python main.py phase4-preview
```

The spectral observer, illuminants, adaptation, colour conversion, smooth-field, and Lloyd routines are vendored with attribution in `engine.py` from `/Users/nehanaik/Documents/pycon/under_another_light.py` and its [public source](https://github.com/neha9029/under-another-light).
