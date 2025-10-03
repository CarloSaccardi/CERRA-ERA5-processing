import cdsapi

dataset = "reanalysis-cerra-single-levels"
request = {
    "variable": ["orography"],
    "level_type": "surface_or_atmosphere",
    "data_type": ["reanalysis"],
    "product_type": "analysis",
    "year": ["2024"],
    "month": ["01"],
    "day": ["01"],
    "time": ["00:00"],
    "data_format": "grib"
}

client = cdsapi.Client()
client.retrieve(dataset, request).download("/aspire/CarloData/CERRA-ERA5-processing/CERRA_processing/lambert_proj/single_levels/orography.grib")
