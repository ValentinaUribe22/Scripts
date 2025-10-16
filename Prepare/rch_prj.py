import pathlib

import numpy as np
import pandas as pd
import xarray as xr

import imod
import glob
from datetime import datetime

paths = glob.glob(
    r"P:/11209740-nbracer/Valentina_Uribe/MAR/hn/rch/rch_*.idf"
)

list = []
for path in paths:
    z = path.split("_")[-1]
    z = z.split(".")[0]
    list.append(z)

list.sort()

da = []
for i in range(0, len(list)):
    print(list[i])
    da.append(datetime.strptime(list[i], "%Y%m%d%H%M%S"))
    da.append("002,001")
    
   
    rch_path = f"'P:/11209740-nbracer/Valentina_Uribe/MAR/hn/rch/rch_{list[i]}.idf'"

    da.append(
        f" 1,2,-001,   1000.000    ,   0.000000    ,   -999.9900    ,                                                                                   {rch_path} >>> (RCH) Recharge Rate (IDF) <<<"
    )
    da.append(
        " 1,1,-001,   1.000000    ,   0.000000    ,   0.000000    ,                                                                      '' >>> CONCENTRATION SPECIES 1 <<<"
    )

f = open(
    "P:/11209740-nbracer/Valentina_Uribe/MAR/hn/rch_to_prj.txt",
    "w",
)
for ele in da:
    f.write(str(ele) + "\n")

years_daily = ["2021", "2050", "2100"]
years_gxg = [
    "2016",
    "2017",
    "2018",
    "2019",
    "2020",
    "2022",
    "2023",
    "2047",
    "2048",
    "2049",
    "2051",
    "2052",
    "2053",
    "2054",
    "2097",
    "2098",
    "2099",
    "2101",
    "2102",
    "2103",
    "2104",
]
gxg_days = ["14000000", "28000000"]

total_saves = 0
da2 = []
da2.append(
    "00000000000000, 1,  1,  1.0000    ,  1.0000    ,  0.0000    ,  0.0000    ,    3000"
)
for i in range(0, len(list)):
    print(list[i])
    # Save every day for specific years
    if any(list[i].startswith(prefix) for prefix in years_daily):
        print("adding daily timestep")
        da2.append(
            f"{list[i]}, 1,  1,  1.0000    ,  1.0000    ,  0.0000    ,  0.0000    ,    3000"
        )
        total_saves = total_saves + 1

    # save every first timestep
    elif "0101000000" in list[i]:
        da2.append(
            f"{list[i]}, 1,  1,  1.0000    ,  1.0000    ,  0.0000    ,  0.0000    ,    3000"
        )
        total_saves = total_saves + 1
        print("adding_first timestep")

    # save every 14th and 28th for gxg's
    elif any(list[i].startswith(prefix) for prefix in years_gxg):
        if any(list[i].endswith(prefix) for prefix in gxg_days):
            da2.append(
                f"{list[i]}, 1,  1,  1.0000    ,  1.0000    ,  0.0000    ,  0.0000    ,    3000"
            )
            total_saves = total_saves + 1
            print("adding_gxg")
        else:
            da2.append(
                f"{list[i]}, 0,  1,  1.0000    ,  1.0000    ,  0.0000    ,  0.0000    ,    3000"
            )
            print("no saving")

    # Saving final timesteps
    elif i == (len(list) - 1):
        da2.append(
            f"{list[i]}, 1,  1,  1.0000    ,  1.0000    ,  0.0000    ,  0.0000    ,    3000"
        )
        total_saves = total_saves + 1
        print("adding_final timestep")
    else:
        da2.append(
            f"{list[i]}, 0,  1,  1.0000    ,  1.0000    ,  0.0000    ,  0.0000    ,    3000"
        )
        print("no saving")
print(total_saves)


f = open(
    r"P:\11209740-nbracer\Valentina_Uribe\MAR\hn\MAR_from_2024.tim", "w"
)
for ele in da2:
    f.write(str(ele) + "\n")
