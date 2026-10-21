import os
from pathlib import Path

import geopandas as gpd

DATA_ROOT = Path(os.environ.get("BASIN2VEC_DATA_ROOT", Path(__file__).resolve().parents[1] / "data"))
BASIN_DIR = DATA_ROOT / "raw/gages-ii/us_selected_basins"

gdf = gpd.read_file(BASIN_DIR / "us_selected_basins.shp")

# Reproject to WGS84
gdf = gdf.to_crs(epsg=4326)

# Keep only what we need
gdf = gdf[["GAGE_ID", "geometry"]]

# Heavy simplification for web
gdf["geometry"] = gdf.geometry.simplify(
    tolerance=0.01,  # ← increase if still large
    preserve_topology=True
)

# Optional: fix invalid shapes
gdf["geometry"] = gdf.geometry.buffer(0)

with open(BASIN_DIR / "basins.geojson", "w") as f:
    f.write(gdf.to_json())

print("Saved simplified GeoJSON")
