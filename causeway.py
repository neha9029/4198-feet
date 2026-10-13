"""OpenStreetMap geometry for the Great Salt Lake railroad causeway.

The polyline is a 20 m Ramer-Douglas-Peucker simplification of current OSM
railway ways tagged ``alt_name=Lucin Cutoff`` and ``railway=rail``. It follows
the main-lake causeway from the west shore to Promontory Point; water east of
the Promontory endpoint remains in the south arm, including Bear River Bay.
"""

from __future__ import annotations

import numpy as np
from rasterio.warp import transform

from data import MEASUREMENT_CRS


OSM_COPYRIGHT_URL = "https://www.openstreetmap.org/copyright"
OSM_WIKI_URL = "https://wiki.openstreetmap.org/wiki/Utah/Railroads"
OSM_MAP_URL = (
    "https://api.openstreetmap.org/api/0.6/map.json?"
    "bbox=-113.25,41.15,-112.30,41.30"
)
OSM_FETCH_DATE = "2026-10-13"
OSM_WAY_IDS = (
    393048022,
    393048012,
    393048007,
    393047995,
    393047988,
    1098385651,
    964846965,
    101676655,
    1255373899,
    962069460,
    962069461,
    10076967,
)

# Longitude, latitude. These are actual OSM railway nodes retained by a 20 m
# simplification, not a latitude proxy or a hand-drawn boundary.
CAUSEWAY_WGS84 = (
    (-113.4489645, 41.2956644),
    (-113.1749461, 41.2661645),
    (-113.0610550, 41.2499280),
    (-112.8679637, 41.2226676),
    (-112.7629121, 41.2208172),
    (-112.7578466, 41.2214808),
    (-112.7502089, 41.2242286),
    (-112.7460843, 41.2245952),
    (-112.5403139, 41.2208717),
    (-112.5359200, 41.2203376),
    (-112.5280753, 41.2172132),
    (-112.5244556, 41.2165545),
    (-112.4848320, 41.2157410),
    (-112.4807644, 41.2153396),
    (-112.4777180, 41.2146310),
)

BEAR_RIVER_BAY_WGS84 = (-112.15, 41.43)
FARMINGTON_BAY_WGS84 = (-112.10, 40.96)


def causeway_utm() -> np.ndarray:
    """Return the west-to-Promontory causeway polyline in UTM Zone 12N."""
    longitude, latitude = np.asarray(CAUSEWAY_WGS84).T
    x, y = transform("EPSG:4326", MEASUREMENT_CRS, longitude, latitude)
    return np.column_stack([x, y])


def classify_utm(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Classify UTM points as 1=north or 0=south using the OSM causeway."""
    line = causeway_utm()
    order = np.argsort(line[:, 0])
    line = line[order]
    causeway_y = np.interp(x, line[:, 0], line[:, 1])
    promontory_x = float(line[-1, 0])
    return ((x <= promontory_x) & (y >= causeway_y)).astype(np.uint8)


def classify_grid(column_x: np.ndarray, row_y: np.ndarray) -> np.ndarray:
    """Classify a north-up raster without allocating full coordinate grids."""
    line = causeway_utm()
    order = np.argsort(line[:, 0])
    line = line[order]
    causeway_y = np.interp(column_x, line[:, 0], line[:, 1])
    west_of_promontory = column_x <= float(line[-1, 0])
    return (
        west_of_promontory[None, :]
        & (row_y[:, None] >= causeway_y[None, :])
    ).astype(np.uint8)


def classify_wgs84(longitude: float, latitude: float) -> int:
    """Classify one named WGS84 reference point for tests and reports."""
    x, y = transform("EPSG:4326", MEASUREMENT_CRS, [longitude], [latitude])
    return int(classify_utm(np.asarray(x), np.asarray(y))[0])


def source_metadata() -> dict:
    """Return reproducibility and attribution details for the arm boundary."""
    return {
        "source": "OpenStreetMap railway geometry",
        "map_query_url": OSM_MAP_URL,
        "railway_wiki_url": OSM_WIKI_URL,
        "copyright_url": OSM_COPYRIGHT_URL,
        "license": "Open Data Commons Open Database License (ODbL)",
        "fetched": OSM_FETCH_DATE,
        "way_ids": list(OSM_WAY_IDS),
        "simplification_tolerance_m": 20.0,
        "polyline_vertices": len(CAUSEWAY_WGS84),
        "scope": "main-lake causeway from west shore to Promontory Point",
    }
