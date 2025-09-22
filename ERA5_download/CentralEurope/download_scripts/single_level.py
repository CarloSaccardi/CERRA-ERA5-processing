import cdsapi

dataset = "reanalysis-era5-single-levels"
request = {
    "product_type": ["reanalysis"],
    "variable": [
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "2m_temperature",
        "instantaneous_surface_sensible_heat_flux",
        "friction_velocity",
        "geopotential",
        "surface_pressure"
        
    ],
    "year": ["2015"],
    "month": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12"
    ],
    "day": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12",
        "13", "14", "15",
        "16", "17", "18",
        "19", "20", "21",
        "22", "23", "24",
        "25", "26", "27",
        "28", "29", "30",
        "31"
    ],
    "time": [
        "01:00", "04:00", "07:00",
        "10:00", "13:00", "16:00",
        "19:00", "22:00"
    ],
    "data_format": "netcdf",
    "download_format": "unarchived",
    "area": [60, -2, 40, 18]
}

client = cdsapi.Client()
client.retrieve(dataset, request).download()
