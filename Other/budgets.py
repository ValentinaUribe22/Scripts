import pathlib
import os
import geopandas as gpd
import imod
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

# Functions
def get_budget_per_sluis(area_number, name, ds_bdg, modelname, year):
    # create folder to save to
    pathlib.Path(f"P:/11209740-nbracer/Valentina_Uribe/vizualizations/budgets/{modelname}").mkdir(
        exist_ok=True, parents=True
    )
    total_drn_drainage_m3dag = ds_bdg["bdgdrn"].sum(dim=["layer"]).mean(dim=["time"])

    imod.idf.write(
        f"P:/11209740-nbracer/Valentina_Uribe/vizualizations/budgets/{modelname}/total_drn_drainage_m3perdag_{modelname}_{year}.idf",
        total_drn_drainage_m3dag,
    )

    total_riv_drainage_m3dag = ds_bdg["bdgriv"].sum(dim=["layer"]).mean(dim=["time"])
    imod.idf.write(
        f"P:/11209740-nbracer/Valentina_Uribe/vizualizations/budgets/{modelname}/total_riv_drainage_m3perdag_{modelname}_{year}.idf",
        total_riv_drainage_m3dag,
    )

    # Calculate total sums
    area = identification.where(identification == area_number).count() * dx * dy * -1
    area.name = "area"
    area_sums = ds_bdg.where(identification == area_number).sum(dim=["layer", "y", "x"])

    df = area_sums.to_dataframe()
    df["identification"] = area_number
    df = df.set_index("identification")
    out = gdf.loc[gdf["identification"] == area_number][["identification", "geometry"]]
    out = out.set_index("identification")
    out = pd.merge(out, df, left_index=True, right_index=True)
    out["area"] = area.values
    for column in list(out.columns):
        if column == "geometry":
            continue
        out[column + "_mmd"] = out[column] / out["area"] * 1000.0

    out["model_sluis_m3/d"] = out["bdgdrn"] * -1 + out["bdgriv"] * -1
    out["time"] = area_sums["time"]

    # out.to_file(f"data/5-visualization/{modelname}/budgets/budgets_{name}.shp")
    out.drop(columns="geometry").round(2).to_csv(
        f"P:/11209740-nbracer/Valentina_Uribe/vizualizations/budgets/{modelname}/budgets_{name}_{modelname}_{year}.csv"
    )

    out_to_plot = out[["time", "model_sluis_m3/d"]]

    # Calulate Salt budgets
    # conc = imod.idf.open(path_conc)
    # conc_budget = conc

    # conc_sum = conc_budget.where(identification == area_number).sum(
    #     dim=["layer", "y", "x"]
    # )

    # df_conc = conc_sum.to_dataframe()
    # Save conc budget file
    # df_conc.round(2).to_csv(
    #     f"data/5-visualization/{modelname}/budgets/salt_budgets_kgd-1_{name}.csv"
    # )
    return out_to_plot  # , df_conc


def open_budgets(modelname,year):
    budgets = xr.Dataset()
    budgets["bdgriv"] = imod.idf.open(f"p:/11207941-005-terschelling-model/TERSCHELLING_IMOD_MODEL_50X50/RESULTS/{modelname}/BDGRIV/BDGRIV_{year}*.idf")
    budgets["bdgdrn"] = imod.idf.open(f"p:/11207941-005-terschelling-model/TERSCHELLING_IMOD_MODEL_50X50/RESULTS/{modelname}/BDGDRN/BDGDRN_{year}*.idf")
    return budgets

# Interim
external_path= "P:/11207941-005-terschelling-model/terschelling-gw-model/data"
path_template = f"{external_path}/2-interim/rch_50/template.nc"
path_2d_template = f"{external_path}/2-interim/rch_50/template_2d.nc"
path_dem = f"{external_path}/2-interim/rch_50/modeltop.nc"
# External
path_waterways_stages = f"{external_path}/1-external/oppervlaktewater/Peilkaarten shapefile/Peilbeheerkaart.shp"

# Create zone for budget is wanted
like_2d = xr.open_dataset(path_2d_template)["template"]
like = xr.open_dataset(path_template)["template"]
dx, xmin, xmax, dy, ymin, ymax = imod.util.spatial_reference(like)

# Open top
# Top
top = imod.idf.open(
    r"p:\11207941-005-terschelling-model\TERSCHELLING_IMOD_MODEL_50X50\TOP\*.idf"
).isel(layer=0)


# open data shapes
shape = gpd.read_file(path_waterways_stages)
kinnum = shape[shape["WATERSYSTE"] == "Kinnum"]
kinnum = kinnum.dissolve()
kinnum["name"] = "Kinnum"
liessluis = shape[shape["WATERSYSTE"] == "Liessluis Terschelling"]
liessluis = liessluis.dissolve()
liessluis["name"] = "Liessluis"
koreabos = shape[
    shape["PBHIDENT"].isin(
        [
            "PBH0092299",
            "PBH0091235",
            "PBH0089810",
            "PBH0093284",
            "PBH0094825",
            "PBH0095723",
            "PBH0096410",
            "PBH0095593",
            "PBH0093407",
            "PBH0095453",
        ]
    )
]
koreabos = koreabos.dissolve()
koreabos["name"] = "Koreabos"
gdf = pd.concat([kinnum, liessluis])
# gdf = gdf.append(koreabos)

gdf["identification"] = np.arange(1, gdf.shape[0] + 1)
identification = imod.prepare.rasterize(gdf, like=top, column="identification")
identification.name = "identification"

# Create budgetzone, MT polder until -10 m
budgetzone = xr.full_like(like, 1.0) * identification
# budgetzone = budgetzone.where(budgetzone.z > -10, np.nan)
budgetzone = budgetzone.swap_dims({"z": "layer"})


# plot map with locations
# Open DEM
dem = xr.open_dataarray(path_dem)

# PLot observation locations
plt.axis("scaled")
fig, ax = imod.visualize.plot_map(
    dem,
    colors="terrain",
    levels=np.linspace(-10, 10, 20),
    figsize=[10, 6],
)
gdf.plot(column="name", legend=True, ax=ax)
ax.set_title("Location budget zones")
pathlib.Path("P:/11209740-nbracer/Valentina_Uribe/vizualizations/budgets").mkdir(
    exist_ok=True, parents=True
)
fig.savefig("P:/11209740-nbracer/Valentina_Uribe/vizualizations/budgets/map.png", dpi=300)
plt.close()


for year in ["2021", "2050", "2100"]:
    print(year)
    # Open data
    print("opening referece..")
    reference = open_budgets("reference", year)
    print("opening Hd..")
    hd = open_budgets("Hd", year)
    print("opening Hn..")
    hn = open_budgets("Hn", year)

    # Get Budgets
    #Kinnum
    print("calculating budgets Kinnum")
    kinnum_ref = get_budget_per_sluis(1, "kinnum", reference, "reference", year)
    kinnum_hd = get_budget_per_sluis(1, "kinnum", hd, "hd", year)
    kinnum_hn = get_budget_per_sluis(1, "kinnum", hn, "hn", year)

    #Liessluis
    print("calculating budgets Liessluis")
    liessluis_ref = get_budget_per_sluis(2, "liessluis", reference, "reference", year)
    liessluis_hd  = get_budget_per_sluis(2, "liessluis", hd, "hd", year)
    liessluis_hn  = get_budget_per_sluis(2, "liessluis", hn, "hn", year)

    # Set legend properties
    l = mlines.Line2D([], [], color="black", label="Reference")
    m = mlines.Line2D([], [], color="blue", label="Hn")
    n = mlines.Line2D([], [], color="red", label="Hd")

    # Plot modelled data
    plotting = {
        "Kinnum": [kinnum_ref, kinnum_hd, kinnum_hn],
        "Liessluis": [liessluis_ref, liessluis_hd, liessluis_hn],
    }


    for name, plot_data in plotting.items():
        print(name)

        #Calculate percentage difference
        hd_perc = (plot_data[1]["model_sluis_m3/d"].sum() - plot_data[0]["model_sluis_m3/d"].sum()) / plot_data[0]["model_sluis_m3/d"].sum() *100
        hn_perc = (plot_data[2]["model_sluis_m3/d"].sum() - plot_data[0]["model_sluis_m3/d"].sum()) / plot_data[0]["model_sluis_m3/d"].sum() *100

        df_perc = pd.DataFrame([[hn_perc, hd_perc]], columns=['Hn percentage', 'Hd Percentage'])
        df_perc.to_csv(f"P:/11209740-nbracer/Valentina_Uribe/vizualizations/budgets/{name}_{year}_percentage_change.csv")

        # Plotting figures
        fig, axs = plt.subplots(
            1, 1, figsize=(10, 8), sharey=False, sharex=False, squeeze=False
        )

        # plot measurement data
        plot_data[0].plot(
            x="time",
            y="model_sluis_m3/d",
            color="k",
            linestyle="solid",
            linewidth=1.2,
            legend=False,
            ax=axs[0][0],
        )

        plot_data[1].plot(
            x="time",
            y="model_sluis_m3/d",
            legend=False,
            ax=axs[0][0],
            color="r",
            linestyle="solid",
            linewidth=1.2,
        )

        plot_data[2].plot(
            x="time",
            y="model_sluis_m3/d",
            legend=False,
            ax=axs[0][0],
            color="b",
            linestyle="solid",
            linewidth=1.2,
        )

        axs[0][0].set_title(f"Budget {name}, {year}")
        axs[0][0].set_ylabel("m3/d")
        fig.tight_layout(pad=4)
        fig.legend(handles=[l, m, n], loc="upper right")
        fig.savefig(
            f"P:/11209740-nbracer/Valentina_Uribe/vizualizations/budgets/modelled_obs_{name}_{year}.png",
            dpi=300,
        )
        plt.close()
