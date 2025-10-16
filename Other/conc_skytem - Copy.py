import pathlib

import geopandas as gpd
import imod
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.ndimage
import xarray as xr
import os

rulename = "validate_conc"
path_snakefile = (
    r"c:\Users\kelde_ts\data\3_projects\Terschelling\terschelling-gw-model\snakefile"
)


def read_snakemake_rule(path, rule: str) -> "snakemake.rules.Rule":
    import snakemake as sm

    workflow = sm.Workflow(snakefile="snakefile")
    workflow.include(path)
    return workflow.get_rule(rule)


if "snakemake" not in globals():
    os.chdir(r"c:\Users\kelde_ts\data\3_projects\Terschelling\terschelling-gw-model")
    snakemake = read_snakemake_rule(path_snakefile, rule=rulename)

# Parameters
# params
params = snakemake.params
modelname = params["modelname"]
zmin = params["zmin"]
zmax = params["zmax"]
start_time = params["start_time"]
end_time = params["end_time"]
# modelname = 'nulmodel_ongekalibreerd'
# zmin = -150.
# zmax = 28.

# Paths
# Interim
path_template = snakemake.input.path_template
path_template_2d = snakemake.input.path_template_2d
path_dem = snakemake.input.path_dem
path_skytem = snakemake.input.path_skytem
path_obs_wells = snakemake.input.path_obs_wells
path_val_da_vitens = snakemake.input.path_val_da_vitens
path_waterschap_data = snakemake.input.path_waterschap_data
path_regis = snakemake.input.path_regis
# External
aoi_shape = snakemake.input.aoi_shape
where_skytem = snakemake.input.where_skytem
path_cross_section_shape = snakemake.input.path_cross_section_shape


# Open templates
like = xr.open_dataset(path_template)["template"]
like_2d = xr.open_dataset(path_template_2d)["template"]
dx, xmin, xmax, dy, ymin, ymax = imod.util.spatial_reference(like)
zmin = like.zbot.min()
zmax = like.ztop.max()
dz = like.dz

# Open concentration data
conc = imod.idf.open(f"data/4-output/{modelname}/conc/conc*.idf")

# Take last 1 year and average plus find grensvlak
end_year = int(end_time[-4:])
conc = conc.sel(time=slice(f"30-09-{str(end_year-1)}", f"30-09-{str(end_year)}")).mean(
    "time"
)

# Open skytem data
sky_shp = gpd.read_file(path_skytem)

# Interpolate skytem data
sky_grid = imod.prepare.rasterize(sky_shp, column="BND_Z", like=like_2d)
# sky_full = imod.prepare.fill(sky_grid)
sky_full = imod.prepare.laplace_interpolate(sky_grid, mxiter=40, close=0.05)

kernel = np.ones((10, 10))
kernel /= kernel.sum()
sky_full.values = scipy.ndimage.convolve(sky_full.values, kernel)

# Mask skytem for where it's applicable
where_sky = gpd.read_file(where_skytem)
where_sky_raster = imod.prepare.spatial.rasterize(where_sky, like=conc.isel(layer=1))
sky_relevant = sky_full.where(where_sky_raster.notnull())
sky_relevant = sky_relevant.rename("skytem")

# Open cross sections
lines = gpd.read_file(path_cross_section_shape)
linestrings = [ls for ls in lines.geometry]
linenames = [ls for ls in lines.name]

# Get all obs data
# Open dataset to plot
# dino data
ds_obs = pd.read_csv(path_obs_wells)
# Vitens data
val_da_vitens = pd.read_csv(path_val_da_vitens)
# waterschap data
val_da_waterschap = pd.read_csv(path_waterschap_data)

# Combine datasets
ds_obs = pd.concat([ds_obs, val_da_vitens, val_da_waterschap])
# Select only location
ds_obs = ds_obs.groupby("id").mean()

# Extract modelled and skytem data
# Select points that are within skytem domain
tmp = imod.select.points_in_bounds(sky_relevant, x=ds_obs.x, y=ds_obs.y)
ds_obs = ds_obs[tmp]
ds_obs["id"] = ds_obs.index

# Select modelled dataset
model_ds = imod.select.points_values(conc, x=ds_obs.x, y=ds_obs.y).to_dataframe()
model_ds = model_ds.reset_index()
model_ds = model_ds.rename(columns={"index": "id"})

# Select skytem dataset
sky_ds = imod.select.points_values(sky_relevant, x=ds_obs.x, y=ds_obs.y).to_dataframe()
sky_ds = sky_ds.reset_index()
sky_ds = sky_ds.rename(columns={"index": "id"})

# Legend
modelled_l = mlines.Line2D([], [], color="blue", label="Gemodelleerde concentratie")
fresh_l = mlines.Line2D([], [], linestyle="--", color="r", label="FRESHEM")

# make dir
pathlib.Path(f"data/5-visualization/{modelname}/validate_conc").mkdir(
    exist_ok=True, parents=True
)

# Turn interactive plotting off
plt.ioff()

# Only select id's where skytem has data
ids = sky_ds[sky_ds["skytem"].notnull()]

# Open DEM
dem = xr.open_dataarray(path_dem)

# Create GDF for plotting purposes
gdf = gpd.GeoDataFrame(ids, geometry=gpd.points_from_xy(ids["x"], ids["y"]))

# Plot locations observations
plt.axis("scaled")
fig, ax = imod.visualize.plot_map(
    dem,
    colors="terrain",
    levels=np.linspace(-10, 10, 20),
    figsize=[20, 12],
)
gdf.plot(column="name", legend=False, color="k", ax=ax, zorder=2.5)
fig.delaxes(fig.axes[1])
ax.set_title("Location observations")
pathlib.Path(f"data/5-visualization/{modelname}/validate_conc/").mkdir(
    exist_ok=True, parents=True
)
fig.savefig(f"data/5-visualization/{modelname}/validate_conc/map.png", dpi=300)

for j in ids["id"]:
    print(j)
    to_plot = model_ds.loc[model_ds["id"] == j]
    ## Adding extra info to table
    to_plot["grensvlak"] = sky_ds.loc[sky_ds["id"] == j]["skytem"].values[0]
    to_plot["1.0"] = 1.0
    to_plot["8.0"] = 8.0

    fig, axs = plt.subplots(
        2, 1, figsize=(12, 14), sharey=False, sharex=False, squeeze=False
    )

    ## First plot conc in depth
    to_plot.plot(x="conc", y="z", legend=False, ax=axs[0][0], color="b", marker="o")
    to_plot.plot(
        x="conc", y="grensvlak", legend=False, ax=axs[0][0], linestyle="--", color="r"
    )
    to_plot.plot(x="1.0", y="z", legend=False, ax=axs[0][0], linestyle="--", color="k")
    to_plot.plot(x="8.0", y="z", legend=False, ax=axs[0][0], linestyle="--", color="k")

    axs[0][0].set_title(f"Chloride concentratie in de diepte, {j}")
    fig.legend(handles=[modelled_l, fresh_l], loc="upper right")
    axs[0][0].set_ylabel("Diepte (mNAP)")
    axs[0][0].set_xlabel("Cl-concentratie (g/l)")
    # axs[0][0].set_ylim([-25,2])

    ## Then plot map with observation point
    x_value = ds_obs.loc[ds_obs["id"] == j].x[0]
    y_value = ds_obs.loc[ds_obs["id"] == j].y[0]

    imod.visualize.plot_map(
        dem,
        colors="terrain",
        levels=np.linspace(-10, 10, 20),
        figsize=[12, 7],
        ax=axs[1, 0],
        fig=fig,
    )
    axs[1, 0].scatter(x=x_value, y=y_value, c="k", s=20, zorder=5)

    fig.savefig(
        f"data/5-visualization/{modelname}/validate_conc/modelled_obs_{j}.png",
        dpi=300,
    )
    plt.close()

# Bereken grensvlak conc
conc_grens = conc.copy()
conc_grens = conc_grens.swap_dims({"layer": "z"})

# Resample to higher resolution
z_new = np.linspace(int(zmin), int(zmax), ((abs(int(zmax - zmin))) * 10 + 1))
conc_grens = conc_grens.interp(z=z_new, method="linear")

# Calculating grensvlak
# For fresh take <= 1.0
grens_fresh = conc_grens["z"].where(conc_grens <= 1.0).min("z")

# For middle grensvlak take <= 8.0
grens_middle = conc_grens["z"].where(conc_grens <= 8.0).min("z")

grens_model = {"1": grens_fresh, "8": grens_middle}

# Plot grensvlakken
# Open overlay
overlays = [{"gdf": gpd.read_file(aoi_shape), "edgecolor": "black", "color": "none"}]

# Get area
area = gpd.read_file(aoi_shape)
area_raster = imod.prepare.spatial.rasterize(area, like=conc.isel(layer=1))

# create folder to save to
pathlib.Path(f"data/5-visualization/{modelname}/grensvlak").mkdir(
    exist_ok=True, parents=True
)

for name, grens in grens_model.items():
    ##### FIGUUR - DIEPTE GRENS MODEL 1 EN 8 G/L
    # Plotting grensvlak
    plt.axis("scaled")
    fig, ax = imod.visualize.plot_map(
        grens,
        colors="jet",
        levels=np.linspace(-100, 5, 22),
        figsize=[15, 10],
        kwargs_colorbar={"label": "Diepte grensvlak (m NAP)"},
        overlays=overlays,
    )
    lines.plot(column="name", ax=ax)
    print(grens.where(area_raster.notnull()).max().values, "max")
    print(grens.where(area_raster.notnull()).min().values, "min")

    # Plot AOI
    ax.set_title(f"Grensvlak {name} g/L")
    fig.savefig(
        f"data/5-visualization/{modelname}/grensvlak/grensvlak_{name}_gL.png", dpi=300
    )
    plt.close()
    imod.idf.write(
        f"data/5-visualization/{modelname}/grensvlak/grensvlak_{name}_gL.idf", grens
    )

    # Neem het verschil met skytem grensvlak
    # Skytem verschil met grens_fresh en grens_middle
    diff = grens - sky_full

    ##### FIGUUR - VERSCHIL 1 EN 8 G/L
    plt.axis("scaled")
    fig, ax = imod.visualize.plot_map(
        diff.where(where_sky_raster.notnull()),
        colors="bwr_r",
        levels=np.linspace(-50, 50, 51),
        figsize=[15, 10],
        kwargs_colorbar={"label": "Verschil diepte grensvlak (m)"},
        overlays=overlays,
    )
    lines.plot(column="name", ax=ax)
    maxi = (
        diff.where(area_raster.notnull()).where(where_sky_raster.notnull()).max().values
    )
    mini = (
        diff.where(area_raster.notnull()).where(where_sky_raster.notnull()).min().values
    )
    print(maxi, "max")
    print(mini, "min")

    # add text box
    textstr = "\n".join(
        (
            r"$\mathrm{Maximum:}%d$" % (maxi,),
            r"$\mathrm{Minimum:}%d$" % (mini,),
        )
    )

    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)

    # place a text box in upper left in axes coords
    ax.text(
        0.83,
        0.98,
        textstr,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=props,
    )

    # Plot AOI
    ax.set_title(f"Grensvlak {name} g/L, verschil met Skytem data")
    fig.savefig(
        f"data/5-visualization/{modelname}/grensvlak/grensvlak_verschil_skytem_{name}_gL.png",
        dpi=300,
    )
    plt.close()
    imod.idf.write(
        f"data/5-visualization/{modelname}/grensvlak/grensvlak_verschil_skytem_{name}_gL.idf",
        diff.where(where_sky_raster.notnull()),
    )

##### FIGUUR - DIEPTE GRENS SKYTEM
plt.axis("scaled")
fig, ax = imod.visualize.plot_map(
    sky_full.where(where_sky_raster.notnull()),
    colors="jet",
    levels=np.linspace(-100, 5, 22),
    figsize=[15, 10],
    kwargs_colorbar={"label": "Diepte grensvlak (m NAP)"},
    overlays=overlays,
)
lines.plot(column="name", ax=ax)
print(sky_full.where(area_raster.notnull()).max().values, "max")
print(sky_full.where(area_raster.notnull()).min().values, "min")

# Plot AOI
ax.set_title("Grensvlak SKYTEM")
fig.savefig(
    f"data/5-visualization/{modelname}/grensvlak/grensvlak_SKYTEM_gL.png", dpi=300
)
plt.close()
imod.idf.write(
    f"data/5-visualization/{modelname}/grensvlak/grensvlak_SKYTEM_gL.idf",
    sky_full.where(where_sky_raster.notnull()),
)

plt.axis("scaled")
fig, ax = imod.visualize.plot_map(
    sky_grid.where(where_sky_raster.notnull()),
    colors="jet",
    levels=np.linspace(-100, 5, 22),
    figsize=[15, 10],
    kwargs_colorbar={"label": "Diepte grensvlak (m NAP)"},
    overlays=overlays,
)
lines.plot(column="name", ax=ax)
print(sky_grid.where(area_raster.notnull()).max().values, "max")
print(sky_grid.where(area_raster.notnull()).min().values, "min")

# Plot AOI
ax.set_title("Grensvlak SKYTEM zonder interpolatie")
fig.savefig(
    f"data/5-visualization/{modelname}/grensvlak/grensvlak_SKYTEM_gL_geeninterpolatie.png",
    dpi=300,
)
plt.close()

## Dikte grensvlak
grens_1 = imod.idf.open(
    f"data/5-visualization/{modelname}/grensvlak/grensvlak_1_gL.idf"
)
grens_8 = imod.idf.open(
    f"data/5-visualization/{modelname}/grensvlak/grensvlak_8_gL.idf"
)
dikte_grensvlak = grens_1 - grens_8
imod.idf.write(
    f"data/5-visualization/{modelname}/grensvlak/dikte_grensvlak_model.idf",
    dikte_grensvlak,
)
plt.axis("scaled")
fig, ax = imod.visualize.plot_map(
    dikte_grensvlak,
    colors="jet",
    levels=np.linspace(0, 20, 20),
    figsize=[15, 10],
    kwargs_colorbar={"label": "Dikte [m]"},
    overlays=overlays,
)
lines.plot(column="name", ax=ax)
ax.set_title("Dikte grensvlak model")
fig.savefig(
    f"data/5-visualization/{modelname}/grensvlak/dikte_grensvlak_model.png", dpi=300
)
plt.close()


##### FIGUUR - CROSS-SECTIE
# Define formations
regis = xr.open_dataset(path_regis).sel(x=slice(xmin, xmax), y=slice(ymax, ymin))
formations = [fm for fm in regis.formation.values if fm[-2] == "k"]
formations.append("HLc")
aquitards = regis.sel(formation=formations)
aquitards = aquitards.dropna("formation", how="all")
is_aquitard = aquitards["top"].notnull()
is_aquitard = is_aquitard.assign_coords(top=aquitards["top"])
is_aquitard = is_aquitard.assign_coords(bottom=aquitards["bot"])
is_aquitard = is_aquitard.rename({"formation": "layer"})

like2 = like.swap_dims({"z": "layer"})
conc = conc.to_dataset(name="chloride")
conc = conc.assign_coords(top=like2["ztop"])
conc = conc.assign_coords(bottom=like2["zbot"])

sections = [imod.select.cross_section_linestring(conc, ls) for ls in linestrings]

skytem_grensvlakken = [
    imod.select.cross_section_linestring(sky_full.where(where_sky_raster.notnull()), ls)
    for ls in linestrings
]

aq_sections = [
    imod.select.cross_section_linestring(is_aquitard, ls).compute()
    for ls in linestrings
]
for i, (cross_section, aq, sktm) in enumerate(
    zip(sections, aq_sections, skytem_grensvlakken)
):
    cross_section = cross_section["chloride"]
    cross_section = cross_section.where(cross_section != cross_section.min())

    cross_section = cross_section.where(~(cross_section < 0.0), other=0.0)
    cross_section = cross_section.where(~(cross_section > 15.9), other=15.9)

    cross_section = cross_section.compute()

    fig, ax = plt.subplots()
    kwargs_aquitards = {
        "hatch": "/",
        "edgecolor": "k",
        "facecolor": "grey",
        "alpha": 0.3,
    }
    levels = np.array([0, 0.15, 0.5, 1, 2, 3, 5, 7.5, 10, 16])
    cmap = "jet"
    s = cross_section
    s = s.rename("chloride (g/L)")

    add_colorbar = True

    fig, ax = imod.visualize.cross_section(
        s,
        colors=cmap,
        levels=levels,
        kwargs_colorbar={
            "label": "Chloride",
            "whiten_triangles": False,
        },
    )
    imod.visualize.cross_sections._plot_aquitards(aq, ax, kwargs_aquitards)

    ## Grensvlak lijn plotten
    sktm.plot(ax=ax, x="s", color="white")
    # imod.visualize.cross_section(grensvlak[-1], ax)
    ax.set_ylim(bottom=zmin, top=10.0)

    title = f"Section {linenames[i]}"
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(
        f"data/5-visualization/{modelname}/grensvlak/cross_sectie {linenames[i]}.png",
        dpi=300,
    )
    ax.clear()
    fig.clf()
