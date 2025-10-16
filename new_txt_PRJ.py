import os
import pandas as pd

# Load CSV with year and addition values
csv_path = r"P:/11209740-nbracer/Valentina_Uribe/scenarios_files/hd_GSLR_S2/gslr_yearly.csv"

# Read CSV (semicolon separator)
df = pd.read_csv(csv_path, sep=';')
reference_levels = dict(zip(df['year'], df['addition']))

output_file = "ghb_all_years.txt"
output_path = r"P:\11209740-nbracer\Valentina_Uribe\scenarios_files\hd_GSLR_S2"

os.makedirs(output_path, exist_ok=True)
output = os.path.join(output_path, output_file)

with open(output, "w") as f:
    for year, value in reference_levels.items():

        if 2005 <= year <= 2048:
         name = r"GHB\2005\GHB_2005_GHB_"
         file = "2005"

        elif 2049 <= year <= 2098:
         name = r"GHB\2050\GHB_2050_GHB_"
         file = "2050"

        else:
         name = r"GHB\2100\GHB_2100_GHB_"
         file = "2100"


        # Header
        f.write(f"{year}-01-01 00:00:00\n")
        f.write("003,028\n")

        # CONDUCTANCE
        for layer in range(1, 29):
            f.write(
                f" 1,2,{layer:03d},    1.000000    ,   0.000000    ,  -999.9900    ,"
                f"'{name}COND_L{layer}.IDF' >>> (CON) CONDUCTANCE (IDF) <<<\n"
            )

        # REFERENCE LEVEL

        for layer in range(1, 29):
            f.write(
                f" 1,2,{layer:03d},    1.000000    ,   {value:10.4f}    ,  -999.9900    ,"
                f"'GHB\{file}\GHB_{file}_GHB_LEVEL_L{layer}.IDF' >>> (LVL) REFERENCE LEVEL (IDF) <<<\n"
            )

        # CONCENTRATION (same each year)
        for layer in range(1, 29):
            f.write(
                f" 1,1,{layer:03d},    1.000000    ,   0.000000    ,  16.0    ,"
                f"'GHB\\CONCENTRATION_MEAN.IDF' >>> (CON) CONCENTRATION (IDF) <<<\n"
            )

print(f"File '{output_file}' created with all years included.")