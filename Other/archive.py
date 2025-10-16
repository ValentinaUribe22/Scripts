import shutil
import os

# params
params = snakemake.params
modelname = params["modelname"]
p_drive = params["p_drive"]
archive_results = params["archive_results"]

# modelname= "Nulmodel_Kalibratie3"
# os.chdir(r'c:\Users\kelde_ts\data\3_projects\Terschelling\terschelling-gw-model')

if archive_results == True:

    ## Zip folders to p-drive
    shutil.make_archive(
        f"{p_drive}/data/2-interim/{modelname}", "zip", f"data/2-interim/{modelname}"
    )
    shutil.make_archive(
        f"{p_drive}/data/3-input/{modelname}", "zip", f"data/3-input/{modelname}"
    )
    shutil.make_archive(
        f"{p_drive}/data/4-output/{modelname}", "zip", f"data/4-output/{modelname}"
    )
    ## Copy visualisation to p-drive
    shutil.copytree(
        f"data/5-visualization/{modelname}",
        f"{p_drive}/data/5-visualization/{modelname}",
    )

    ## Remove local folders
    shutil.rmtree(f"data/2-interim/{modelname}")
    shutil.rmtree(f"data/3-input/{modelname}")
    shutil.rmtree(f"data/4-output/{modelname}")
    shutil.rmtree(f"data/5-visualization/{modelname}")

else:
    print("no archiving")
