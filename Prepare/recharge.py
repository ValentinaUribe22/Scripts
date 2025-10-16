import imod
import xarray as xr
import pandas as pd
import os
import numpy as np
from datetime import date
from tqdm import tqdm

print("creating recharge")
# rulename = "make_recharge"
# path_snakefile = (
#     r"c:\Users\kelde_ts\data\3_projects\Terschelling\terschelling-gw-model\snakefile"
# )


# def read_snakemake_rule(path, rule: str) -> "snakemake.rules.Rule":
#     import snakemake as sm

#     workflow = sm.Workflow(snakefile="snakefile")
#     workflow.include(path)
#     return workflow.get_rule(rule)


# if "snakemake" not in globals():
#     os.chdir(r"c:\Users\kelde_ts\data\3_projects\Terschelling\terschelling-gw-model")
#     snakemake = read_snakemake_rule(path_snakefile, rule=rulename)

# ## Paths
# path_template = snakemake.input.path_template
# path_template_2d = snakemake.input.path_template_2d
# path_ref_evaporation_KNMI = snakemake.input.path_ref_evaporation_KNMI
# path_precipitation_KNMI = snakemake.input.path_precipitation_KNMI
# path_LGN = snakemake.input.path_LGN
# path_cropfactors = snakemake.input.path_cropfactors
# path_interception_factors = snakemake.input.path_interception_factors
# path_groundwater_depth = snakemake.input.path_groundwater_depth
# path_gidsgewas_LGN = snakemake.input.path_gidsgewas_LGN
# path_water_masks = snakemake.input.path_water_mask

# ## Output
# path_recharge = snakemake.output.path_recharge

# ## Params
# params = snakemake.params
# start_time = params["start_time"]
# end_time = params["end_time"]
# frequency = params["frequency"]
# rootzone_depth = params["rootzone_depth"]
# specific_yield = params["specific_yield_constant"]
# spin_up = params["spin_up"]
# spin_up_time = params["spin_up_time"]
# recharge_type = params["recharge_type"]
# start_time_average_recharge = params["start_time_average_recharge"]
# end_time_average_recharge = params["end_time_average_recharge"]

import numpy as np
import shutil
import pathlib

## Global variables
modelname= "rch_50"

# Run model after writing it
run_model = False

# Type of model "imod-wq" or "modflow6"
model_type = "imod_wq"

# If running an iMOD-swq model, specify the parameters below. MF6 is not able to run in parallel yet.
# Set to True if you want to run the model in parallel, set the cores to the amount of cores available
run_parallel = True
cores = 4

# Discretization
start_time = "01-01-2010"
end_time = "31-12-2021"
frequency = "MS" # QS, MS or W. For frequency options see: https://stackoverflow.com/questions/17001389/pandas-resample-documentation 
cellsize = 20
zmin = -150.
zmax = 28. # AHN indicates maximum of 31.26 m
z_discretization = np.array([-5.0] * 4 + [-1.0] * 4 + [-0.5] * 8 + [-1.0] * 10 + [-10.0] * 4 + [-20.0] * 5)

# Check if discretization matches with zmin and zmax, throw error if it doesn't match up:
assert z_discretization.sum() == (
    zmin - zmax
), "Take heed, your vertical discretization does not add up to (zmax - zmin)"

# Porosity 
porosity_subsoil = 0.3

# Calculate unconfined, set to True if unconfined, False if Confined.
unconfined = False

# Storage properties
specific_storage_constant = 1.e-5
specific_yield_constant = 0.2
rootzone_depth = 0.2

## General head boundary (sea) properties
ghb_conductance = 1000.
slope_density_conc = 1.25
density_ref = 1000.0
conc_ghb = 16.0
eb = -1.0
vloed = 0.8

## If you want to use the chloride outputs (last time step) from a previous model run set to True and specify modelname
use_spin_up_cl = True
modelname_spin_up = "terschelling_nulmodel_lang"

## Properties drain that simulates overland flow, 1 day for stability
resistance_drn = 1.0 # Weerstand in dagen
resistance_drn_lakes = 1.0 # Weerstand in dagen

## Recharge
rch_concentration = 0.0
# Set to true if you want to include a spin_up period, spin_up_time indicates the amount of years added before
spin_up = False
spin_up_time = 40

## Use new top, this was build into the model to indicate if we want to use the new bottom produced as part of the VCL project
new_top = False

## Set to yes if you want to save heads and conc to a netcdf, option build in for VCL project
save_netcdf = False

## All external input paths
external_path= "P:/11207941-005-terschelling-model/terschelling-gw-model/data"
modeldomain_path = f"{external_path}/1-external/aoi/aoi_model_adj.shp"
island_shape_path = f"{external_path}/1-external/aoi/aoi.shp"
coastline_path = f"{external_path}/1-external/coastline/nl_imergis_kustlijn_2018.shp"
path_dijken = f"{external_path}/1-external/aoi/dijken.shp"

# Modeltop
ahn_path = f"{external_path}/1-external/bathy_ahn/ahn2_20m_cmNAP.nc"
bathymetry_path = f"{external_path}/1-external/bathy_ahn/BATHY.idf"

# Subsurface
geotop_path = f"{external_path}/1-external/subsurface/geotop.nc"
geotop_water_path = f"{external_path}/1-external/subsurface/geotop_water_surface.nc"
geotop_table_path = f"{external_path}/1-external/subsurface/GeoTOP_k_values_kalibratie2.csv"
regis_path = f"{external_path}/1-external/subsurface/regis_v2_2.nc"

# Meteorology
path_ref_evaporation_KNMI = f"{external_path}/1-external/meteo/evaporation_19900101_20240622.nc"
path_precipitation_KNMI = f"{external_path}/1-external/meteo/precipitation_19900101_20240622.nc"

# Landuse
path_LGN = f"{external_path}/1-external/landuse/lgn7/lgn7.idf"
path_cropfactors = f"{external_path}/1-external/landuse/Gewasfactoren_GWZ_2016_BramBot.csv"
path_interception_factors = f"{external_path}/1-external/landuse/Interceptiefactoren_GWZ_2016_BramBot.csv"
path_gidsgewas_LGN = f"{external_path}/1-external/landuse/lgn7/vertaaltabel_LGN7_gewasfactoren.csv"

# Groundwater depth, used for recharge (to select areas with big unsaturated zone)
path_groundwater_depth = f"{external_path}/1-external/rch/depth_gws_Huidig.idf"

# Extraction wells
path_meta_data_wells = f"{external_path}/1-external/wells/Vitens_Winputten_Onttrekkingen/20210728_Pompputten_Vitens_Terschelling_metadata_csvfiles.csv"

# River network and stages
path_waterways = f"{external_path}/1-external/rivers/rivs_only.shp"
path_waterways_stages = f"{external_path}/1-external/oppervlaktewater/Peilkaarten shapefile/Peilbeheerkaart.shp"
path_waterways_lines = f"{external_path}/1-external/oppervlaktewater/Oppervlaktewater primair en secundair shapefile/Wateren schouwwateren en hoofdwater.shp"

# Kalibratie
path_waterlopen_westelijk_duingebied = f"{external_path}/1-external/kalibratie/waterlopen_westelijk_duingebied.shp"

# lakes
path_lakes = f"{external_path}/1-external/rivers/lakes.shp"

# Visualization
path_cross_section_shape = f"{external_path}/1-external/aoi/cross_section_lines.shp"

# Skytem
path_inversie = f"{external_path}/1-external/skytem/inversie/Q_Terschelling_19_Inv_inv.xyz"
where_skytem = f"{external_path}/1-external/aoi/waar_skytem.shp"

# budgets sluizen
path_sluis_both = f"{external_path}/1-external/sluizen/afvoer sluizen Terschelling_2011_2020.csv"
path_kinnum = f"{external_path}/1-external/sluizen/Kinnum.csv"
path_liessluis = f"{external_path}/1-external/sluizen/Liessluis.csv"

# VCL new bottom
path_new_top = f"{external_path}/1-external/bathy_ahn/vcl/nieuwe_bodem_v2.tif"

path_template = f"{external_path}/2-interim/{modelname}/template.nc"
path_template_2d = f"{external_path}/2-interim/{modelname}/template_2d.nc"
path_water_masks = f"{external_path}/2-interim/{modelname}/water_masks.nc"

frequency = "D"

## Opening data
## Open templates
like = xr.open_dataset(path_template)["template"]
like_2d = xr.open_dataset(path_template_2d)["template"]
dx, xmin, xmax, dy, ymin, ymax = imod.util.spatial_reference(like)
zmin = like.zbot.min()
zmax = like.ztop.max()
dz = like.dz

# Opening watermask
water_mask = xr.open_dataset(path_water_masks)["is_sea_2d"]

start_time = "01-08-2010"
end_time = "22-06-2024"
date_range = pd.date_range(start=start_time, end=end_time, freq='6m')

start_time=date_range[0]
end_time = date_range[1]

recharge_type =1
specific_yield = specific_yield_constant


for i in range (0,len(date_range)):
    start_time=(date_range[i] + pd.Timedelta(days=1)).strftime("%d-%m-%Y")
    if i == len(date_range)-1:
        end_time = "22-06-2024"
    else:
        end_time = (date_range[i+1]).strftime("%d-%m-%Y")
    print(start_time,end_time)
    path_recharge = f"{external_path}/2-interim/{modelname}/recharge_{start_time}_{end_time}.nc"
    print(path_recharge)


    ## Opening meteo
    if recharge_type == 1:  # 1=recharge corresponding to current modeling period
        ref_evaporation = (
            xr.open_dataset(path_ref_evaporation_KNMI)
            .sel(x=slice(xmin, xmax))
            .sel(y=slice(ymax, ymin))
            .sel(time=slice(start_time, end_time))
            # .compute()
        )
        precipitation = (
            xr.open_dataset(path_precipitation_KNMI)
            .sel(x=slice(xmin, xmax))
            .sel(y=slice(ymax, ymin))
            .sel(time=slice(start_time, end_time))
            # .compute()
        )
    elif recharge_type == 2:  # 2=average recharge corresponding to a specified period
        ref_evaporation = (
            xr.open_dataset(path_ref_evaporation_KNMI)
            .sel(x=slice(xmin, xmax))
            .sel(y=slice(ymax, ymin))
            .sel(time=slice(start_time_average_recharge, end_time_average_recharge))
            .compute()
        )
        precipitation = (
            xr.open_dataset(path_precipitation_KNMI)
            .sel(x=slice(xmin, xmax))
            .sel(y=slice(ymax, ymin))
            .sel(time=slice(start_time_average_recharge, end_time_average_recharge))
            .compute()
        )


    ## Opening landuse, crop and interception factors
    landuse = (
        imod.idf.open(path_LGN).sel(x=slice(xmin, xmax)).sel(y=slice(ymax, ymin)).compute()
    )
    interception_factors = pd.read_csv(path_interception_factors, index_col=0, header=0)
    landuse_gidsgewas = pd.read_csv(path_gidsgewas_LGN)
    cropfactors = pd.read_csv(path_cropfactors, index_col=0, header=0)

    groundwater_depth = (
        imod.idf.open(path_groundwater_depth)
        .sel(x=slice(xmin, xmax))
        .sel(y=slice(ymax, ymin))
    )


    ## Opening metaswap data
    # fact_svat = pd.read_fwf(path_fact_svat, widths=[6,6,8,8,8,8,8,8,8],header=None)
    # luse_svat = pd.read_fwf(path_luse_svat, header=None)

    ##-------------------------------------------
    ## Klaarzetten
    ##-------------------------------------------

    recharge = xr.Dataset()

    ## Time on chosen frequency, otherwise script won't run because of memory issues
    # ref_evaporation = ref_evaporation.resample(time=frequency).sum()
    precipitation = precipitation.resample(time=frequency).sum()

    ## Regrid to correct resolution
    mean_regridder = imod.prepare.Regridder(method="mean")
    recharge["ref_evaporation"] = (
        mean_regridder.regrid(ref_evaporation["evaporation"], like_2d) * 0.001
    )  # /1000 to get from mm to meter
    recharge["precipitation"] = (
        mean_regridder.regrid(precipitation["precipitation"], like_2d) * 0.001
    )  # /1000 to get from mm to meter
    mean_regridder = imod.prepare.Regridder(
        method="mean"
    )  # Opnieuw inladen nodig for some reason
    groundwater_depth = mean_regridder.regrid(groundwater_depth, like_2d)

    mode_regridder = imod.prepare.Regridder(method="mode")
    recharge["landuse"] = mode_regridder.regrid(landuse, like_2d)


    del landuse
    del ref_evaporation
    del precipitation

    ##-------------------------------------------
    ## calculate potential evaporation
    ##-------------------------------------------


    ## Make table with crop factors per month and per LGN7-type
    cropfactors["Geen"] = (
        1  # crops where I didn't have a factor for, get value 1 for all months
    )
    cropfactors_reshape = cropfactors.reset_index().melt(
        id_vars=["Maand"], var_name="landuse", value_name="factor"
    )
    cropfactors_gewassen = pd.merge(
        cropfactors_reshape, landuse_gidsgewas, left_on="landuse", right_on="Gidsgewas"
    )
    cropfactors_gewassen = cropfactors_gewassen[["Maand", "LGN_code", "factor"]]

    ## cropfactors to data-array
    cropfactors_gewassen = cropfactors_gewassen.set_index(["Maand", "LGN_code"])
    cropfactors_da = cropfactors_gewassen.to_xarray()

    ## Change nan to 16 (water)
    recharge["landuse"] = recharge["landuse"].where(recharge["landuse"].notnull(), 16)

    ## Combine crop factors and landuse map
    cropfactors_3d = cropfactors_da.sel(LGN_code=recharge["landuse"]).drop("LGN_code")

    ## From month to date
    # times = pd.date_range(start_time, end_time, name='time')
    ds = xr.Dataset({"time": recharge.time})
    ds["Maand"] = ds.time.dt.month.compute()
    recharge["evap_factor"] = (
        cropfactors_3d["factor"].sel(Maand=ds.Maand).drop("Maand").compute()
    )

    ## Calculate potential evaporation
    recharge["pot_evaporation"] = recharge["ref_evaporation"] * recharge["evap_factor"]#.compute()


    ##-------------------------------------------
    ## Calculate interception factor
    ##-------------------------------------------

    ## Columns to landuse factors
    interception_factors = interception_factors.rename(
        columns={"Loofhout": 11, "Zwaar naaldhout": 12, "Bebouwing": 18}
    )
    ## naaldhoud heeft twee landgebruisklassen (12 en 19)
    interception_factors[19] = interception_factors[12]

    ## Add other landuses to table, they get factor 0
    for i in range(0, int(recharge["landuse"].max().values + 1)):
        if i not in interception_factors.columns:
            interception_factors[i] = 0

    ## Reshape dataframe and convert to xarray
    reshape = interception_factors.reset_index().melt(
        id_vars=["Maand"], var_name="landuse_code", value_name="factor"
    )
    reshape2 = reshape.set_index(["Maand", "landuse_code"])
    da_interception_factors = reshape2.to_xarray()

    ## Combine interception factors and landuse map
    interceptionfactors_3d = da_interception_factors.sel(
        landuse_code=recharge["landuse"]
    ).drop("landuse_code")

    ## From month to date
    # times = pd.date_range(start_time, end_time, name='time')
    ds = xr.Dataset({"time": recharge.time})
    ds["month"] = ds.time.dt.month.compute()
    recharge["interception_factor"] = (
        interceptionfactors_3d["factor"].sel(Maand=ds.month).drop("Maand").compute()
    )

    ## Calculate net precipitation
    recharge["interception"] = recharge["precipitation"] * recharge["interception_factor"]#.compute()


    ##-------------------------------------------
    ## Calculate recharge with bucket model
    ##-------------------------------------------

    ## Bucket model is only applied in areas with high unsaturated zone
    treshold_groundwater_depth = 1.0  # meter
    high_unsat_zone = xr.full_like(groundwater_depth, 1).where(
        groundwater_depth > treshold_groundwater_depth
    )


    ## Calculate maximum amount of water (m) in the rootzone bucket
    max_water = rootzone_depth * specific_yield

    ## create new arrays to fill in the loop (bucket model)
    recharge["storage"] = recharge["pot_evaporation"].where(
        recharge["pot_evaporation"] == max_water, max_water
    )
    recharge["act_evaporation"] = recharge["pot_evaporation"].where(
        recharge["pot_evaporation"] == max_water, max_water
    )
    recharge["recharge_bucket"] = recharge["pot_evaporation"].where(
        recharge["pot_evaporation"] == max_water, max_water
    )

    for i in tqdm(range(0, len(recharge.time.values))):
        ## first guess of storage (only taking precipitation and interception into account)
        time_now = recharge.time.values[i]
        if time_now == recharge.time.values[0]:
            new_storage_concept = (
                recharge["precipitation"].sel(time=time_now)
                - recharge["interception"].sel(time=time_now)
                + max_water
            )  # '+max_water' because storage is full on first timestep
        else:
            time_previous = recharge.time.values[i - 1]
            new_storage_concept = (
                recharge["storage"].sel(time=time_previous)
                + recharge["precipitation"].sel(time=time_now)
                - recharge["interception"].sel(time=time_now)
            )
        ## evaporation cannot exceed storage
        pot_evaporation = recharge["pot_evaporation"].sel(time=time_now)
        act_evaporation = pot_evaporation.where(
            pot_evaporation < new_storage_concept, new_storage_concept
        )
        ## Calculate new storage
        new_storage = new_storage_concept - act_evaporation
        ## Storage cannot exceed maximum storage, excess of water becomes recharge
        recharge_tmp = new_storage - max_water
        recharge_tmp = recharge_tmp.where(recharge_tmp > 0, 0)
        new_storage = new_storage.where(new_storage < max_water, max_water)
        ## Store new data in array
        recharge["act_evaporation"] = recharge["act_evaporation"].where(
            recharge["act_evaporation"].time != time_now, act_evaporation
        )
        recharge["storage"] = recharge["storage"].where(
            recharge["storage"].time != time_now, new_storage
        )
        recharge["recharge_bucket"] = recharge["recharge_bucket"].where(
            recharge["recharge_bucket"].time != time_now, recharge_tmp
        )
    ## Apply only in area with high unsaturated zone
    recharge["act_evaporation"] = recharge["act_evaporation"].where(
        high_unsat_zone == 1, recharge["pot_evaporation"]
    )

    ##-------------------------------------------
    ## Calculate recharge
    ##-------------------------------------------

    recharge["recharge_m"] = (
        recharge["precipitation"] - recharge["pot_evaporation"] - recharge["interception"]
    )
    ## In dune area, take recharge from bucket model
    # Kalibratie
    recharge["recharge_m"] = recharge["recharge_m"].where(
        high_unsat_zone != 1, recharge["recharge_bucket"]
    )

    ##-------------------------------------------
    ## Creating final dataset
    ##-------------------------------------------

    starts = recharge.time.values[:-1]
    ends = recharge.time.values[1:]

    timedeltas = [end - start for start, end in zip(starts, ends)]
    duration = timedeltas / np.timedelta64(1, "D")
    length_last_timestep = duration[-1]
    duration = np.append(duration, length_last_timestep)

    list = []
    # Compute in m/d
    for i in range(0, len(duration)):
        select_rch = recharge["recharge_m"].isel(time=i)
        dur = duration[i]
        rch_new = select_rch / dur
        list.append(rch_new)
    rch_md = xr.concat(list, dim="time")

    # Slicing recharge at the right times and sampling to the correct frequency
    recharge_m = rch_md.sel(time=slice(start_time, end_time))
    # Ensure to add first time step before resample
    if frequency == "W":
        rch_first = recharge_m.isel(time=0)
        rch_first = rch_first.assign_coords(time=pd.to_datetime(start_time))
        recharge_m = xr.concat([rch_first, recharge_m], dim="time")
    recharge_m = recharge_m.resample(time=frequency).mean("time").ffill(dim="time")

    rch = recharge_m
    # # calculating average recharge, for first steady state timestep
    # rch_avg = recharge_m.mean("time")
    # timedelta = np.timedelta64(1, "D")  # 1 day duration for initial steady-state
    # starttime = recharge_m.coords["time"][0] - timedelta
    # rch_avg = rch_avg.assign_coords(time=starttime)
    # rch = xr.concat([rch_avg, recharge_m], dim="time")


    # Regrid to correct resolution
    rch_out = imod.prepare.Regridder(method="mean").regrid(rch, like_2d)

    # Masking where water is present according to water_mask
    # rch_out = imod.prepare.fill(rch_out).where(water_mask == 0)


    # Creating dataset and saving
    rch = xr.Dataset()
    rch["rch"] = rch_out  # *2 # kalibratie

    # Set spin up time
    if spin_up:
        rch_avg = rch["rch"].isel(time=0, drop=True)
        dates = pd.date_range(
            end=rch["time"].values[1], periods=spin_up_time + 1, freq="YS", closed="left"
        ).to_series()
        dates = pd.concat([dates[[0]] - pd.to_timedelta("1d"), dates])
        rch_rate = xr.DataArray(1.0, {"time": dates}, ["time"]) * rch_avg
        recharge_rate = xr.concat(
            [rch_rate, rch["rch"].isel(time=slice(1, None))],
            dim="time",
        )
        rch = xr.Dataset()
        rch["rch"] = recharge_rate

    if recharge_type == 2:  # 2=average recharge corresponding to a specified period
        rch_avg = rch["rch"].isel(
            time=0, drop=True
        )  # want die 1e tijdstap is eerder al als starttijd gezet
        timedelta = np.timedelta64(1, "D")  # 1 day duration for initial steady-state
        dates = pd.date_range(start=start_time, end=end_time, freq=frequency)
        dates = pd.concat(
            [(dates[[0]] - pd.to_timedelta("1d")).to_series(), dates.to_series()]
        )
        rch_rate = xr.DataArray(1.0, {"time": dates}, ["time"]) * rch_avg
        rch = xr.Dataset()
        rch["rch"] = rch_rate

    rch = rch.transpose("time", "y", "x")
    rch.to_netcdf(r"P:/11209740-nbracer/Valentina_Uribe/Terschelling_model/rch_50/recharge.nc")
    imod.idf.save(r"P:/11209740-nbracer/Valentina_Uribe/Terschelling_model/rch_50/rch/rch.idf", rch)
    del rch
