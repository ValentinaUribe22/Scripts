import imod
import xarray as xr
import geopandas as gpd
import os
import pandas as pd


# This rulename should match one in the snakemake file
rulename = "make_ghb"
path_snakefile = (
    r"P:\11209740-nbracer\Valentina_Uribe\Prepare\snakefile"
)


def read_snakemake_rule(path, rule: str) -> "snakemake.rules.Rule":
    import snakemake as sm

    workflow = sm.Workflow(snakefile="snakefile")
    workflow.include(path)
    return workflow.get_rule(rule)

# path where your snakemake file is
if "snakemake" not in globals():
    os.chdir(r"P:\11209740-nbracer\Valentina_Uribe\Prepare")
    snakemake = read_snakemake_rule(path_snakefile, rule=rulename)

# ## Paths
# # Input
# External
path_dijken = snakemake.input.path_dijken
path_sealevel = snakemake.input.path_sealevel
# Interim
path_template = snakemake.input.path_template
path_template_2d = snakemake.input.path_template_2d
path_modeltop = snakemake.input.path_modeltop
# # Output
path_ghb = snakemake.output.path_ghb
path_is_sea = snakemake.output.path_is_sea

## Params
params = snakemake.params
start_time = params["start_time"]
end_time = params["end_time"]
conductance = params["ghb_conductance"]
slope_density_conc = params["slope_density_conc"]
density_ref = params["density_ref"]
conc_ghb = params["conc_ghb"]
eb = params["eb"]
vloed = params["vloed"]
sealevelrise = params["sealevelrise"]

# Opening data
# Open templates
like = xr.open_dataset(path_template)["template"]
like_2d = xr.open_dataset(path_template_2d)["template"]
dx, xmin, xmax, dy, ymin, ymax = imod.util.spatial_reference(like)
zmin = like.zbot.min()
zmax = like.ztop.max()
dz = like.dz

# modeltop
modeltop = xr.open_dataset(path_modeltop)["modeltop"]

# dikes
dikes_shp = gpd.read_file(path_dijken)
dikes = imod.prepare.rasterize(dikes_shp, like=like_2d, fill=0.0)

# Create ghb components
# Find upper active model layer, where the general head boundary will be applied
like_max_top = xr.full_like(like, 1.0).swap_dims({"z": "layer"})
like_max_top = like_max_top.where(like_max_top["zbot"] < modeltop)
upper_active_layer = imod.select.upper_active_layer(like_max_top, is_ibound=False)

# keep only the locations below flood line, outside of dikes
upper_active_layer = upper_active_layer.where(modeltop <= vloed).where(dikes == 0)
is_sea = upper_active_layer.where(upper_active_layer.isnull(), 1)
is_sea = is_sea.fillna(0.0)
is_sea.to_netcdf(path_is_sea)

# create the area where the GHB will be present
is_ghb = xr.full_like(like, 1.0).swap_dims({"z": "layer"})
is_ghb = is_ghb.where(is_ghb.layer == upper_active_layer)
is_ghb = is_ghb.swap_dims({"layer": "z"})

# Conductance
ghb_cond = xr.full_like(like, 1.0) * conductance

# Apply sealevel rise
if sealevelrise:
    sealevel = pd.read_csv(path_sealevel)
    start_sealevel = float(sealevel["ZSS"][sealevel["jaar"] == int(start_time[-4:])])
    eind_sealevel = float(sealevel["ZSS"][sealevel["jaar"] == int(end_time[-4:])])
    mean_sealevel = (start_sealevel + eind_sealevel) / 2
    eb = eb + mean_sealevel
    vloed = vloed + mean_sealevel

# set ghb's
# stage below eb line is equal to eb level
# fill fill entire array with average values
mean_level = (eb + vloed) / 2
mean_level_3d = xr.full_like(like, mean_level)

# Stage between eb and flood is set to (modeltop + vloed)/2
eb_vloed_level = (modeltop + vloed) / 2
eb_vloed_level = eb_vloed_level.where(modeltop >= eb)
# transforming it to 3d object
eb_vloed_3d = xr.full_like(like, 1.0) * eb_vloed_level

# Combining to one 3d object
ghb_stage = eb_vloed_3d.combine_first(mean_level_3d)
# ghb_stage = mean_level_3d # kalibratie


# Setting concentration
ghb_conc = xr.full_like(like, conc_ghb)
ghb_density = ghb_conc * slope_density_conc + density_ref

# making sure GHB is only present in the right layer
ghb_cond = ghb_cond.where(is_ghb.notnull())
ghb_stage = ghb_stage.where(is_ghb.notnull())
ghb_conc = ghb_conc.where(is_ghb.notnull())
ghb_density = ghb_density.where(is_ghb.notnull())

# asserting to ensure ghb is present in the right cells
assert ghb_cond.count() == ghb_stage.count() == ghb_conc.count() == ghb_density.count()

# Save
ds = xr.Dataset()
ds["stage"] = ghb_stage
ds["conc"] = ghb_conc
ds["cond"] = ghb_cond
ds["density"] = ghb_density
ds = ds.reindex_like(like).transpose("z", "y", "x")
ds.to_netcdf(path_ghb)