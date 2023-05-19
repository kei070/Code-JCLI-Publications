"""
Namelist CMIP5
"""


#%% set the model and file name lists
models = ["ACCESS1_0", "ACCESS1_3", "BCC_CSM1_1", "BCC_CSM1_1_M", "BNU_ESM", "CanESM2", "CCSM4", "CNRM_CM5", 
          "FGOALS_S2", "GFDL_CM3", "GFDL_ESM2G", "GFDL_ESM2M", "GISS_E2_H", "GISS_E2_R", "HadGEM2_ES", "INMCM4", 
          "IPSL_CM5A_LR", "IPSL_CM5B_LR", "MIROC_ESM", "MIROC5", "MPI_ESM_LR", "MPI_ESM_MR", "MPI_ESM_P", "MRI_CGCM3", 
          "NorESM1_M"]
models_n = ["ACCESS1-0", "ACCESS1-3", "bcc-csm1-1", "bcc-csm1-1-m", "BNU-ESM", "CanESM2", "CCSM4", "CNRM-CM5", 
            "FGOALS-s2", "GFDL-CM3", "GFDL-ESM2G", "GFDL-ESM2M", "GISS-E2-H", "GISS-E2-R", "HadGEM2-ES", "inmcm4", 
            "IPSL-CM5A-LR", "IPSL-CM5B-LR", "MIROC-ESM", "MIROC5", "MPI-ESM-LR", "MPI-ESM-MR", "MPI-ESM-P", 
            "MRI-CGCM3", "NorESM1-M"]
models_pl = ["ACCESS1.0", "ACCESS1.3", "BCC-CSM1.1", "BCC-CSM1.1(m)", "BNU-ESM", "CanESM2", "CCSM4", "CNRM-CM5", 
             "FGOALS-s2", "GFDL-CM3", "GFDL-ESM2G", "GFDL-ESM2M", "GISS-E2-H", "GISS-E2-R", "HadGEM2-ES", "INMCM4", 
             "IPSL-CM5A-LR", "IPSL-CM5B-LR", "MIROC-ESM", "MIROC5", "MPI-ESM-LR", "MPI-ESM-MR", "MPI-ESM-P", 
             "MRI-CGCM3", "NorESM1-M"]


#%% branch time indices
b_times = {"ACCESS1_0": 0, "ACCESS1_3": 0, "BCC_CSM1_1": 160, "BCC_CSM1_1_M": 240, "BNU_ESM": 400, "CanESM2": 10, 
           "CCSM4": 1, "CNRM_CM5": 0, "FGOALS_S2": 0, "GFDL_CM3": 0, "GFDL_ESM2G": 0, "GFDL_ESM2M": 0, "GISS_E2_H": 10, 
           "GISS_E2_R": 12, "HadGEM2_ES": 0, "INMCM4": 240, "IPSL_CM5A_LR": 50, "IPSL_CM5B_LR": 20, "MIROC_ESM": 80, 
           "MIROC5": 100, "MPI_ESM_LR": 30, "MPI_ESM_MR": 0, "MPI_ESM_P": 16, "MRI_CGCM3": 40, "NorESM1_M": 0}


#%% grid tag
grid_tg = {"ACCESS1_0":"", "ACCESS1_3":"", "BCC_CSM1_1":"", "BCC_CSM1_1_M":"", "BNU_ESM":"", "CanESM2":"", 
           "CCSM4":"", "CNRM_CM5":"", "FGOALS_S2":"", "GFDL_CM3":"", "GFDL_ESM2G":"", "GFDL_ESM2M":"", "GISS_E2_H":"", 
           "GISS_E2_R":"", "HadGEM2_ES":"", "INMCM4":"", "IPSL_CM5A_LR":"", "IPSL_CM5B_LR":"", "MIROC_ESM":"", 
           "MIROC5":"", "MPI_ESM_LR":"", "MPI_ESM_MR":"", "MPI_ESM_P":"", "MRI_CGCM3":"", "NorESM1_M":""}


#%% set up the CMIP6 version
cmip = "CMIP5"


#%% set up the abrupt4xCO2 directory (it is abrupt-4xCO2 in CMIP6)
a4x_dir = "abrupt4xCO2_"
