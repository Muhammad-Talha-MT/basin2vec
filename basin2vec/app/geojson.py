import geopandas as gpd

gdf = gpd.read_file("/data/basin2vec/raw/gages-ii/us_selected_basins/us_selected_basins.shp")

# Reproject to WGS84
gdf = gdf.to_crs(epsg=4326)

# Keep only what we need
gdf = gdf[["GAGE_ID", "geometry"]]

# 🔥 HEAVY simplification for web
gdf["geometry"] = gdf.geometry.simplify(
    tolerance=0.01,  # ← increase if still large
    preserve_topology=True
)

# Optional: fix invalid shapes
gdf["geometry"] = gdf.geometry.buffer(0)

with open("/data/basin2vec/raw/gages-ii/us_selected_basins/basins.geojson", "w") as f:
    f.write(gdf.to_json())

print("Saved simplified GeoJSON")
