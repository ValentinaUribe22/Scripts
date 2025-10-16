import pathlib
import imod
import geopandas as gpd
import numpy as np
import os
import glob


def modify_recharge(start_year, end_year, input_dir, output_dir, pond_path):
    
    # Load and reproject the pond shapefile
    pond = gpd.read_file(pond_path).to_crs(epsg=28992)
    
    # Create output directory if it doesn't exist
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    saved_files = []  # Collect output file paths
    winter_months = [10, 11, 12, 1, 2, 3]
    
    
    for year in range(start_year, end_year + 1):
        print(f"Processing year: {year}")
        
        modified_files = 0
       
        for month in range (1,13):
            file_pattern = os.path.join(input_dir, f"rch_{year}{month:02d}*.idf")
            file_list = glob.glob(file_pattern)
        
            
            for file_path in file_list:
                filename = os.path.basename(file_path)
                
                # Open the recharge file
                rch = imod.idf.open(file_path)
                rch_2d = rch.isel(time=0)
                
                # Reproject (if needed)
                rch_2drp = imod.prepare.reproject(source=rch_2d, like=rch_2d, src_crs="EPSG:28992", dst_crs="EPSG:28992")
                
                # Rasterize ponds
                pondr = imod.prepare.rasterize(pond, like=rch_2drp)
              
                if month in winter_months: 
                    
                    # Apply pond recharge modification
                    modified_rch = rch_2drp.where(pondr.isnull(), 0.025)
                    modified_rch = modified_rch.where(~pondr.isnull(), rch_2drp)
                    
                    
                    # Check if 0.025 is in the modified recharge grid
                    unique_values = np.unique(modified_rch.values)
                    if 0.025 not in unique_values:
                        print(f"Warning: 0.025 not found in {filename}")
                    
                    modified_files +=1
                    print(filename)
                    
                else: 
                    #No modification, keep the original file
                    modified_rch= rch_2drp
                    
                    
                # Save the modified recharge
                output_path = os.path.join(output_dir, filename)
                imod.idf.save(output_path, modified_rch)
               
                saved_files.append(filename)
              
  
    print(f" {modified_files} files modified for year {year}")

    
    return saved_files
        
modify_recharge(
    start_year=2024,
    end_year=2105,
    input_dir="p:/11207941-005-terschelling-model/TERSCHELLING_IMOD_MODEL_50X50/RCH/Hn/rch",
    output_dir="P:/11209740-nbracer/Valentina_Uribe/MAR/hn/rch",
    pond_path="P:/11209740-nbracer/Valentina_Uribe/Shapefiles/pond1.shp",
    
)
