import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.sankey import Sankey

snakemake = []


def plot_budget_sankey(snakemake):
    # params
    params = snakemake.params
    modelname = params["modelname"]

    df = pd.read_csv(f"data/5-visualization/{modelname}/budgets/budgets.csv")

    mm_columns = [
        "bdgbnd_mmd",
        "bdgdrn_mmd",
        "bdgghb_mmd",
        "bdgrch_mmd",
        "bdgriv_mmd",
        "bdgsto_mmd",
        "vertical_mmd",
        "horizontal_mmd",
    ]

    df_mm = df[mm_columns] * -1.0

    DIRECTIONS = {
        "bdgbnd_mmd": 0,
        "bdgdrn_mmd": 1,
        "bdgghb_mmd": 0,
        "bdgrch_mmd": 1,
        "bdgriv_mmd": 1,
        "bdgsto_mmd": 1,
        "vertical_mmd": -1,
        "horizontal_mmd": 0,
    }

    LABELS = {
        "bdgbnd_mmd": "Boundry condition",
        "bdgdrn_mmd": "Drainage",
        "bdgghb_mmd": "GHB",
        "bdgrch_mmd": "Recharge",
        "bdgriv_mmd": "River",
        "bdgsto_mmd": "Storage",
        "vertical_mmd": "Vertical",
        "horizontal_mmd": "Horizontal",
    }

    flows = df_mm.iloc[0].sort_values()
    labels = flows.index.values
    labels = labels[flows != 0]
    # labels = [label.replace("_mmd", "") for label in labels]
    flows = flows.values
    flows = flows[flows != 0]
    orientations = [DIRECTIONS[label] for label in labels]
    labels = [LABELS[label] for label in labels]
    scale = 1.0 / (np.abs(flows).sum())
    scale = 0.2

    fig, ax = plt.subplots(figsize=(10, 8))
    sankey = Sankey(
        ax=ax,
        flows=flows,
        orientations=orientations,
        labels=labels,
        scale=scale,
        unit="mm/d",
        pathlengths=0.5,
    ).finish()
    ax.set_xticks([])
    ax.set_yticks([])

    # save fig
    # fig.show()
    pathlib.Path(f"data/5-visualization/{modelname}/waterbalance").mkdir(
        exist_ok=True, parents=True
    )
    fig.savefig(f"data/5-visualization/{modelname}/waterbalance/sankey.png", dpi=300)
