"""
Namelist CMIP6
"""


#%% set the model and file name lists
models = ["ACCESS-CM2", "ACCESS-ESM1-5", "AWI-CM-1-1-MR", "BCC-CSM2-MR", "BCC-ESM1", "CAMS-CSM1-0", "CanESM5", 
          "CAS-ESM2-0", "CESM2", "CESM2-FV2", "CESM2-WACCM", "CESM2-WACCM-FV2", "CIESM", "CMCC-CM2-SR5", "CNRM-CM6-1", 
          "CNRM-CM6-1-HR", "CNRM-ESM2-1", "E3SM-1-0", "EC-Earth3", "EC-Earth3-Veg", "FGOALS-f3-L", "FGOALS-g3", 
          "FIO-ESM-2-0", "GFDL-CM4", "GFDL-ESM4","GISS-E2-1-G", "GISS-E2-1-H", "GISS-E2-2-G", "HadGEM3-GC31-LL", 
          "HadGEM3-GC31-MM", "IITM-ESM", "INM-CM4-8", "INM-CM5", "IPSL-CM6A-LR", "KACE-1-0-G", "MCM-UA-1-0", 
          "MIROC-ES2L", "MIROC6", "MPI-ESM1-2-HAM", "MPI-ESM1-2-HR", "MPI-ESM1-2-LR", "MRI-ESM2-0", "NESM3", "NorCPM1", 
          "NorESM2-LM", "NorESM2-MM", "SAM0-UNICON", "TaiESM1", "UKESM1-0-LL", "CMCC-ESM2", "EC-Earth3-AerChem", 
          "EC-Earth3-CC", "KIOST-ESM", "ICON-ESM-LR", "GISS-E2-2-H", "E3SM-2-0", "UKESM1-1-LL"]
models_n = ["ACCESS-CM2", "ACCESS-ESM1-5", "AWI-CM-1-1-MR", "BCC-CSM2-MR", "BCC-ESM1", "CAMS-CSM1-0", "CanESM5", 
            "CAS-ESM2-0", "CESM2", "CESM2-FV2", "CESM2-WACCM", "CESM2-WACCM-FV2", "CIESM", "CMCC-CM2-SR5", "CNRM-CM6-1", 
            "CNRM-CM6-1-HR", "CNRM-ESM2-1", "E3SM-1-0", "EC-Earth3", "EC-Earth3-Veg", "FGOALS-f3-L", "FGOALS-g3", 
            "FIO-ESM-2-0", "GFDL-CM4", "GFDL-ESM4","GISS-E2-1-G", "GISS-E2-1-H", "GISS-E2-2-G", "HadGEM3-GC31-LL", 
            "HadGEM3-GC31-MM", "IITM-ESM", "INM-CM4-8", "INM-CM5", "IPSL-CM6A-LR", "KACE-1-0-G", "MCM-UA-1-0", 
            "MIROC-ES2L", "MIROC6", "MPI-ESM1-2-HAM", "MPI-ESM1-2-HR", "MPI-ESM1-2-LR", "MRI-ESM2-0", "NESM3", "NorCPM1", 
            "NorESM2-LM", "NorESM2-MM", "SAM0-UNICON", "TaiESM1", "UKESM1-0-LL", "CMCC-ESM2", "EC-Earth3-AerChem", 
            "EC-Earth3-CC", "KIOST-ESM", "ICON-ESM-LR", "GISS-E2-2-H", "E3SM-2-0", "UKESM1-1-LL"]
models_pl = ["ACCESS-CM2", "ACCESS-ESM1.5", "AWI-CM-1.1-MR", "BCC-CSM2-MR", "BCC-ESM1", "CAMS-CSM1.0", "CanESM5", 
             "CAS-ESM2.0", "CESM2", "CESM2-FV2", "CESM2-WACCM", "CESM2-WACCM-FV2", "CIESM", "CMCC-CM2-SR5", "CNRM-CM6.1", 
             "CNRM-CM6.1-HR", "CNRM-ESM2.1", "E3SM-1.0", "EC-Earth3", "EC-Earth3-Veg", "FGOALS-f3-L", "FGOALS-g3", 
             "FIO-ESM2.0", "GFDL-CM4", "GFDL-ESM4","GISS-E2.1-G", "GISS-E2.1-H", "GISS-E2.2-G", "HadGEM3-GC31-LL", 
             "HadGEM3-GC31-MM", "IITM-ESM", "INM-CM4.8", "INM-CM5", "IPSL-CM6A-LR", "KACE-1.0-G", "MCM-UA-1.0", 
             "MIROC-ES2L", "MIROC6", "MPI-ESM1.2-HAM", "MPI-ESM1.2-HR", "MPI-ESM1.2-LR", "MRI-ESM2.0", "NESM3", 
             "NorCPM1", "NorESM2-LM", "NorESM2-MM", "SAM0-UNICON", "TaiESM1", "UKESM1.0-LL", "CMCC-ESM2", 
             "EC-Earth3-AerChem", "EC-Earth3-CC", "KIOST-ESM", "ICON-ESM-LR", "GISS-E2.2-H", "E3SM-2.0", "UKESM1.1-LL"]


#%% branch time indices
b_times = {"ACCESS-CM2":0, "ACCESS-ESM1-5":0, "AWI-CM-1-1-MR": 249, "BCC-CSM2-MR": 0, "BCC-ESM1": 0, "CAMS-CSM1-0": 130, 
           "CanESM5": 0, "CAS-ESM2-0":0, "CESM2": 500, "CESM2-FV2": 320, "CESM2-WACCM": 70, "CESM2-WACCM-FV2": 300, 
           "CIESM":0, "CMCC-CM2-SR5":0, "CNRM-CM6-1":0, "CNRM-CM6-1-HR":0, "CNRM-ESM2-1":0, "E3SM-1-0": 100, 
           "EC-Earth3":0, "EC-Earth3-Veg": 0, "FGOALS-f3-L": 34, "FGOALS-g3":0, "FIO-ESM-2-0":0, "GFDL-CM4":100, 
           "GFDL-ESM4":100, "GISS-E2-1-G": 0, "GISS-E2-1-H": 0, "GISS-E2-2-G": 0, "HadGEM3-GC31-LL":0, 
           "HadGEM3-GC31-MM":0, "IITM-ESM":0, "INM-CM4-8": 97, "INM-CM5": 103, "IPSL-CM6A-LR": 20, "KACE-1-0-G":0, 
           "MCM-UA-1-0":0, "MIROC-ES2L":0, "MIROC6": 0, "MPI-ESM1-2-HAM": 100, "MPI-ESM1-2-HR": 0, "MPI-ESM1-2-LR": 0, 
           "MRI-ESM2-0": 0, "NESM3": 50, "NorCPM1":0, "NorESM2-LM": 0, "NorESM2-MM": 0, "SAM0-UNICON": 273, "TaiESM1":0, 
           "UKESM1-0-LL":0, "CMCC-ESM2":0, "EC-Earth3-AerChem":0, "EC-Earth3-CC":0, "KIOST-ESM":0, "ICON-ESM-LR":0,
           "GISS-E2-2-H":0, "E3SM-2-0": 100, "UKESM1-1-LL":0}


#%% grid tags
grid_tg = {"ACCESS-CM2":"gn", "ACCESS-ESM1-5":"gn", "AWI-CM-1-1-MR":"gn", "BCC-CSM2-MR":"gn", "BCC-ESM1":"gn", 
           "CAMS-CSM1-0":"gn", "CanESM5":"gn", "CAS-ESM2-0":"gn", "CESM2":"gn", "CESM2-FV2":"gn", "CESM2-WACCM":"gn", 
           "CESM2-WACCM-FV2":"gn", "CIESM":"gr", "CMCC-CM2-SR5":"gn", "CNRM-CM6-1":"gr", "CNRM-CM6-1-HR":"gr", 
           "CNRM-ESM2-1":"gr", "E3SM-1-0":"gr", "EC-Earth3":"gr", "EC-Earth3-Veg":"gr", "FGOALS-f3-L":"gr", 
           "FGOALS-g3":"gn", "FIO-ESM-2-0":"gn", "GFDL-CM4":"gr1", "GFDL-ESM4":"gr1", "GISS-E2-1-G":"gn", 
           "GISS-E2-1-H":"gn", "GISS-E2-2-G":"gn", "HadGEM3-GC31-LL":"gn", "HadGEM3-GC31-MM":"gn", "IITM-ESM":"gn", 
           "INM-CM4-8":"gr1", "INM-CM5":"gr1", "IPSL-CM6A-LR":"gr", "KACE-1-0-G":"gn", "MCM-UA-1-0":"gn", 
           "MIROC-ES2L":"gn", "MIROC6":"gn", "MPI-ESM1-2-HAM":"gn", "MPI-ESM1-2-HR":"gn", "MPI-ESM1-2-LR":"gn", 
           "MRI-ESM2-0":"gn", "NESM3":"gn", "NorCPM1":"gn", "NorESM2-LM":"gn", "NorESM2-MM":"gn", "SAM0-UNICON":"gn", 
           "TaiESM1":"gn", "UKESM1-0-LL":"gn", "CMCC-ESM2":"gn", "EC-Earth3-AerChem":"gn", "EC-Earth3-CC":"gn", 
           "KIOST-ESM":"gn", "ICON-ESM-LR":"gn", "GISS-E2-2-H":"gn", "E3SM-2-0":"gr", "UKESM1.1-LL":"gn"}


#%% set up the CMIP6 version
cmip = "CMIP6"


#%% set up the abrupt-4xCO2 directory (it is abrupt4xCO2 in CMIP5)
a4x_dir = "abrupt-4xCO2_"
