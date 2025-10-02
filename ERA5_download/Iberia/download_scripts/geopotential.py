import cdsapi

dataset = "reanalysis-era5-single-levels"
request = {
    "product_type": ["reanalysis"],
    "year": ["2024"],
    "month": ["01"],
    "day": ["01"],
    "time": ["01:00"],
    "data_format": "grib",
    "download_format": "unarchived",
    "variable": ["geopotential"],
    "area": [45, -15, 25, 5]
}

client = cdsapi.Client()
client.retrieve(dataset, request).download("/aspire/CarloData/CERRA-ERA5-processing/ERA5_download/Iberia/single_levels/geopotential_iberia.nc")