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

Phase 2 uses a 3,032-point shoreline-aware Delaunay mesh, two Lloyd relaxation passes, and fixed anchors on both accepted shorelines. The render is cropped to the largest all-valid rectangle and its boundary is explicitly sampled, eliminating no-data edge triangles. Each triangle owns a bounded 41-sample reflectance spectrum rather than a display colour: land, managed areas, and lost-water triangles share one continuous muted earth/sage `smooth_field` family. The railroad-causeway latitude splits the lake into Gunnison (north) and Gilbert (south) arms; Bear River and Farmington Bays stay with the south arm. Fixed, non-metameric current-water spectra render the north arm deep red-violet and the south arm deep blue-green in daylight.

For every lost-water triangle, SLSQP searches a smooth projected spectral basis whose perturbations are in the exact D65 XYZ nullspace. Under Illuminant A, the north arm sweeps a*b* angles 0°, +15°, and −15° while the south sweeps −180°, −165°, and −150°; each angle is tested with second-difference weights 20, 50, and 200. The objective is `−10·depth_strength·projection + 2·||orthogonal||² + s·||D2 r||²`, with zero L* weight and five seeded restarts projected from `clip(r0 + N(0, 0.15))`. Depth strength is `0.6 + 0.4 × normalised depth` and weights the projection rather than capping it. The fixed seed is 1931, spectra remain in [0.015, 0.96], and the final D65 image must be pixel-identical to the unmodified baseline. Previews are rendered under D65, a 50/50 D65–2856 K SPD mixture, and 2856 K.

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
.venv/bin/python main.py phase2
```

The spectral observer, illuminants, adaptation, colour conversion, smooth-field, and Lloyd routines are vendored with attribution in `engine.py` from `/Users/nehanaik/Documents/pycon/under_another_light.py` and its [public source](https://github.com/neha9029/under-another-light).
