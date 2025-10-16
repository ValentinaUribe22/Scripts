import pandas as pd
import numpy as np
import geopandas as gpd
import os
import imod

rulename = 'get_skytem_data'
path_snakefile = r'c:\Users\pouwels\Documents\Terschelling\terschelling-gw-model\snakefile'

def read_snakemake_rule(path, rule: str) -> "snakemake.rules.Rule":
    import snakemake as sm
    workflow = sm.Workflow(snakefile="snakefile")
    workflow.include(path)
    return workflow.get_rule(rule)

if "snakemake" not in globals():
    os.chdir(r'c:\Users\pouwels\Documents\Terschelling\terschelling-gw-model')
    snakemake = read_snakemake_rule(path_snakefile, rule=rulename)

# Get paths
path_inversie = snakemake.input.path_inversie
path_skytem_out = snakemake.output.path_skytem_out

# lees geinverteerde data.
# kolommen:
# XUTM, YUTM: coordinaten wgs84
# ELEVATION: maaiveldhoogte
# DATASET: ? een of ander id
# DAPOS
# NUMLAYERS: aantal lagen per xy punt -> constant, 19
# per laag:
#  - RHOi: bulk weerstand in ? Ohmm?
#  - RHOSTDi: stdev RHOi
#  - THKi: dikte laag
#  - THKSTDi: stdev THKi
#  - DEPi: diepte laagmidden?
#  - DEPSTDi: stdev DEPi
nlay = 19
df_inversie = pd.read_fwf(path_inversie)
stdcols = ["XUTM","YUTM","ELEVATION"]
rhocols = [f"RHO{i}" for i in range(1,nlay+1)]
thkcols = [f"THK{i}" for i in range(1,nlay+1)]
depcols = [f"DEP{i}" for i in range(1,nlay+1)]

# supersimpel: zz-grens = RHO < 5.9
# upper RHO column where rho < 5.9
rho_thresh = 5.  # following Pedersen, J. B., Schaars, F. W., Christiansen, A. V, & Foged, N. (2017). Mapping the fresh-saltwater interface in the coastal zone using high-resolution airborne electromagnetics. First Break, 35, 57–61.
first_lowresis = (df_inversie[rhocols] < rho_thresh).values.argmax(axis=1) 
above = np.fmax(0,first_lowresis - 1)
# slice correct DEP column for first_lowresis per row (trick from https://stackoverflow.com/questions/23435782/numpy-selecting-specific-column-index-per-row-by-using-a-list-of-indexes)
lower_depth = df_inversie[depcols].values[np.arange(len(df_inversie)),first_lowresis]
upper_depth = df_inversie[depcols].values[np.arange(len(df_inversie)),above]
midpoint = 0.5*lower_depth + 0.5*upper_depth
z_mid = df_inversie["ELEVATION"] - midpoint

# add columns to df
df_inversie["RHO_BND"] = (df_inversie[rhocols] < rho_thresh).idxmax(axis=1)
df_inversie["BND_MID"] = midpoint
df_inversie["BND_Z"] = z_mid

# create geopandas
gdf = gpd.GeoDataFrame(df_inversie[stdcols+["RHO_BND","BND_MID","BND_Z"]],crs="EPSG:32631",geometry=gpd.points_from_xy(x=df_inversie["XUTM"],y=df_inversie["YUTM"]))
gdf = gdf.to_crs("EPSG:28992")
gdf.to_file(path_skytem_out)

# gdf["x"] = gdf["XUTM"]
# gdf["y"] = gdf["YUTM"]
# gdf = gdf.reset_index()
# gdf["id"] = "ID_"
# gdf["id"] = gdf["id"] + gdf["index"].astype(str)


# ipf_out = gdf[
#     ["x", "y", "id", "RHO_BND", "BND_MID", "BND_Z"]
# ]  

# # Saving IPF
# imod.ipf.save(
#     path=r"data/2-interim/validation_data/skytem/skytem_data.ipf",
#     df=ipf_out,
#     nodata=-999999,
# )

