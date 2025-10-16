import os
import pandas as pd
import geopandas as gpd
import numpy as np
import xarray as xr
from tqdm import tqdm
import datetime
import matplotlib.pyplot as plt
import imod
import pathlib

rulename = "kalibratie_heads_westduin"
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


path_area_lage_heads = snakemake.input.path_area_lage_heads

params = snakemake.params
modelname = params["modelname"]
modelname_compare = params["modelname_compare"]
compare_other_run = params["compare_other_run"]
p_drive = params["p_drive"]

# modelname= "Nulmodel_Kalibratie4"
# compare_other_run = True
# p_drive = "p:/11207941-005-terschelling-model/terschelling-gw-model"
# modelname_compare = 'Nulmodel_Kalibratie2'
# path_area_lage_heads = "data/1-external/kalibratie/Lage_heads.shp"


if compare_other_run == True:

    ## read data
    heads = pd.read_csv(
        f"data/5-visualization/{modelname}/validate_heads/head-results.csv"
    )
    heads_compare = pd.read_csv(
        f"{p_drive}/data/5-visualization/{modelname_compare}/validate_heads/head-results.csv"
    )
    area_lage_heads = gpd.read_file(path_area_lage_heads)
    area_lage_heads = area_lage_heads.to_crs("epsg:28992")

    ## Make geodataframe
    heads_gdf = gpd.GeoDataFrame(
        heads, geometry=gpd.points_from_xy(heads.x, heads.y), crs="EPSG:28992"
    )
    heads_west = heads_gdf[heads_gdf.within(area_lage_heads.geometry[0])]
    heads_compare_gdf = gpd.GeoDataFrame(
        heads_compare,
        geometry=gpd.points_from_xy(heads_compare.x, heads_compare.y),
        crs="EPSG:28992",
    )
    heads_compare_west = heads_compare_gdf[
        heads_compare_gdf.within(area_lage_heads.geometry[0])
    ]

    ## Make path
    pathlib.Path(
        f"data/5-visualization/{modelname}/kalibratie_heads_westduin/diff_{modelname_compare}"
    ).mkdir(exist_ok=True, parents=True)

    ## Afwijking versus diepte plotten
    plt.scatter(
        heads_compare_west["diff"],
        heads_compare_west["filt_middle"],
        label=modelname_compare,
    )
    plt.scatter(heads_west["diff"], heads_west["filt_middle"], label=modelname)
    plt.axvline(0)
    plt.legend()
    plt.ylabel("diepte filter")
    plt.xlabel("verschil grondwater")
    plt.title("Afwijking versus diepte, alleen west Terschelling")
    plt.savefig(
        f"data/5-visualization/{modelname}/kalibratie_heads_westduin/diff_{modelname_compare}/afwijking_versus_diepte_WestTerschelling.png"
    )
    plt.close()

    ## En ingezoomed
    plt.scatter(
        heads_compare_west["diff"],
        heads_compare_west["filt_middle"],
        label=modelname_compare,
    )
    plt.scatter(heads_west["diff"], heads_west["filt_middle"], label=modelname)
    plt.axvline(0)
    plt.legend()
    plt.ylabel("diepte filter")
    plt.ylim(-20, 10)
    plt.xlabel("verschil grondwater")
    plt.title("Afwijking versus diepte, alleen west Terschelling")
    plt.savefig(
        f"data/5-visualization/{modelname}/kalibratie_heads_westduin/diff_{modelname_compare}/afwijking_versus_diepte_WestTerschelling_zoomed.png"
    )
    plt.close()

    ## Afwijking versus grondwaterstand
    plt.scatter(
        heads_compare_west["diff"],
        heads_compare_west["head_mean"],
        label=modelname_compare,
    )
    plt.scatter(heads_west["diff"], heads_west["head_mean"], label=modelname)
    plt.axvline(0)
    plt.legend()
    plt.ylabel("Gemiddelde head")
    plt.xlabel("verschil grondwater")
    plt.title("Afwijking versus head, alleen west Terschelling")
    plt.savefig(
        f"data/5-visualization/{modelname}/kalibratie_heads_westduin/diff_{modelname_compare}/afwijking_versus_head_WestTerschelling.png"
    )
    plt.close()

    ## Afwijking versus grondwaterstand bij diepte <-5m
    plt.scatter(
        heads_compare_west[heads_compare_west["filt_middle"] > -5]["diff"],
        heads_compare_west[heads_compare_west["filt_middle"] > -5]["head_mean"],
        label=modelname_compare,
    )
    plt.scatter(
        heads_west[heads_west["filt_middle"] > -5]["diff"],
        heads_west[heads_west["filt_middle"] > -5]["head_mean"],
        label=modelname,
    )
    plt.axvline(0)
    plt.legend()
    plt.ylabel("Gemiddelde head")
    plt.xlabel("verschil grondwater")
    plt.title(
        "Afwijking versus head, alleen west Terschelling, alleen bij ondiepe filters (ondieper dan -5m)"
    )
    plt.savefig(
        f"data/5-visualization/{modelname}/kalibratie_heads_westduin/diff_{modelname_compare}/afwijking_versus_head_WestTerschelling_ondiepeFilters.png"
    )
    plt.close()

    ## Gemeten versus berekende grondwaterstand plotten
    plt.scatter(
        heads_compare_west["head_mean"],
        heads_compare_west["modelled_head_mean"],
        label=modelname_compare,
    )
    plt.scatter(
        heads_west["head_mean"], heads_west["modelled_head_mean"], label=modelname
    )
    plt.plot([0, 5], [0, 5], color="red")
    plt.legend()
    plt.ylabel("Gemiddelde head (model) [m NAP]")
    plt.xlabel("Gemiddelde head (meting) [m NAP]")
    plt.title("Model versus meting, alleen west Terschelling")
    plt.savefig(
        f"data/5-visualization/{modelname}/kalibratie_heads_westduin/diff_{modelname_compare}/head_model_versus_meting_WestTerschelling.png"
    )
    plt.close()

    ## Gemeten versus berekende grondwaterstand plotten bij diepte <-5m
    plt.scatter(
        heads_compare_west[heads_compare_west["filt_middle"] > -5]["head_mean"],
        heads_compare_west[heads_compare_west["filt_middle"] > -5][
            "modelled_head_mean"
        ],
        label=modelname_compare,
    )
    plt.scatter(
        heads_west[heads_west["filt_middle"] > -5]["head_mean"],
        heads_west[heads_west["filt_middle"] > -5]["modelled_head_mean"],
        label=modelname,
    )
    plt.plot([0, 5], [0, 5], color="red")
    plt.legend()
    plt.ylabel("Gemiddelde head (model) [m NAP]")
    plt.xlabel("Gemiddelde head (meting) [m NAP]")
    plt.title(
        "Model versus meting, alleen west Terschelling, alleen bij ondiepe filders (ondieper dan -5m)"
    )
    plt.savefig(
        f"data/5-visualization/{modelname}/kalibratie_heads_westduin/diff_{modelname_compare}/head_model_versus_meting_WestTerschelling_ondiepeFilters.png"
    )
    plt.close()


else:
    print("compare_runs staat op False")
