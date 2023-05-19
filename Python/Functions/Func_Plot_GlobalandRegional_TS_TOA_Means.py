import numpy as np
import pylab as pl
from scipy import stats


def plot_region_3panel(glob_mean_ts, r1_mean_ts, r2_mean_ts,
                       glob_mean_toa, r1_mean_toa, r2_mean_toa, mod,
                       r1_acro, r2_acro):
    
    
    # linear regression with respect to time for TOA
    lp_yrs = np.arange(20, 150)  # set up late period years
    lr_toa_glob = stats.linregress(lp_yrs, glob_mean_toa[20:])
    lr_toa_r1 = stats.linregress(lp_yrs, r1_mean_toa[20:])
    lr_toa_r2 = stats.linregress(lp_yrs, r2_mean_toa[20:])
    
    # linear regression with respect to time for TS
    lr_ts_glob = stats.linregress(lp_yrs, glob_mean_ts[20:])
    lr_ts_r1 = stats.linregress(lp_yrs, r1_mean_ts[20:])
    lr_ts_r2 = stats.linregress(lp_yrs, r2_mean_ts[20:])
    
    # linear regression of EP-mean TOA on EP-mean TS as well as WP-mean on TS
    lr_toa_ts_gl = stats.linregress(glob_mean_ts[20:], glob_mean_toa[20:])
    lr_toa_ts_r1 = stats.linregress(r1_mean_ts[20:], r1_mean_toa[20:])
    lr_toa_ts_r2 = stats.linregress(r2_mean_ts[20:], r2_mean_toa[20:])
    
    # set colours for the plot
    r1_lr_c = "black"
    r2_lr_c = "gray"
    gl_lr_c = "orange"
    r1_c = "red"
    r2_c = "green"
    gl_c = "blue"
    lw = 3  # regression linewidth
    
    # start the plotting
    fig, (ax1, ax2, ax3) = pl.subplots(nrows=1, ncols=3, figsize=(25, 6))
    ax1.plot(r1_mean_ts, c=r1_c, label=r1_acro + " mean")
    ax1.plot(r2_mean_ts, c=r2_c, label=r2_acro + " mean")
    ax1.plot(glob_mean_ts, c=gl_c, label="global mean")
    ax1.plot(lp_yrs, lp_yrs*lr_ts_glob[0] + lr_ts_glob[1], c=gl_lr_c, 
             linewidth=lw,
             label="Gl. lin. reg. p=" + str(np.round(lr_ts_glob[3], 
                                                     decimals=3)) + 
             " sl=" + str(np.round(lr_ts_glob[0], decimals=3)) + " K/a")
    ax1.plot(lp_yrs, lp_yrs*lr_ts_r1[0] + lr_ts_r1[1], c=r1_lr_c, 
             linewidth=lw,
             label=r1_acro + "  lin. reg. p=" + str(np.round(lr_ts_r1[3], 
                                                             decimals=3)) + 
             " sl=" + str(np.round(lr_ts_r1[0], decimals=3)) + " K/a")
    ax1.plot(lp_yrs, lp_yrs*lr_ts_r2[0] + lr_ts_r2[1], c=r2_lr_c, 
             linewidth=lw,
             label=r2_acro + "  lin. reg. p=" + str(np.round(lr_ts_r2[3],
                                                             decimals=3)) + 
             " sl=" + str(np.round(lr_ts_r2[0], decimals=3)) + " K/a")
    ax1.set_ylim((0, 8.5))
    ax1.legend()
    ax1.set_title("TS Time Series " + mod)
    ax1.set_xlabel("year since abrupt4xCO2")
    ax1.set_ylabel("temperature change in K")
    
    ax2.plot(r1_mean_toa, c=r1_c, label=r1_acro + "  mean")
    ax2.plot(r2_mean_toa, c=r2_c, label=r2_acro + "  mean")
    ax2.plot(glob_mean_toa, c=gl_c, label="global mean")
    ax2.plot(lp_yrs, lp_yrs*lr_toa_glob[0] + lr_toa_glob[1], c=gl_lr_c, 
             linewidth=lw,
             label="Gl. lin. reg. p=" + str(np.round(lr_toa_glob[3], 
                                                     decimals=3)) +
             " sl=" + str(np.round(lr_toa_glob[0], decimals=3)) + " W/m$^2$/a")
    ax2.plot(lp_yrs, lp_yrs*lr_toa_r1[0] + lr_toa_r1[1], c=r1_lr_c, 
             linewidth=lw,
             label=r1_acro + "  lin. reg. p=" + str(np.round(lr_toa_r1[3], 
                                                    decimals=3)) +
             " sl=" + str(np.round(lr_toa_r1[0], decimals=3)) + " W/m$^2$/a")
    ax2.plot(lp_yrs, lp_yrs*lr_toa_r2[0] + lr_toa_r2[1], c=r2_lr_c, 
             linewidth=lw,
             label=r2_acro + "  lin. reg. p=" + str(np.round(lr_toa_r2[3], 
                                                    decimals=3)) +
             " sl=" + str(np.round(lr_toa_r2[0], decimals=3)) + " W/m$^2$/a")
    ax2.set_ylim((-6, 50))
    ax2.legend()
    ax2.set_title("TOA Imbalance Time Series " + mod)
    ax2.set_xlabel("year since abrupt4xCO2")
    ax2.set_ylabel("TOA net radiation change in W/m$^2$")
    
    ax3.scatter(r1_mean_ts[20:], r1_mean_toa[20:], c=r1_c, label=r1_acro + 
                "  means")
    ax3.scatter(r2_mean_ts[20:], r2_mean_toa[20:], c=r2_c, label=r2_acro + 
                "  means")
    ax3.scatter(glob_mean_ts[20:], glob_mean_toa[20:], c=gl_c, 
                label="global means")
    ax3.plot([np.min(glob_mean_ts[20:]), np.max(glob_mean_ts[20:])],
              np.array([np.min(glob_mean_ts[20:]), 
                        np.max(glob_mean_ts[20:])]) * 
              lr_toa_ts_gl[0] + lr_toa_ts_gl[1], c=gl_lr_c, 
              label="Gl. lin. reg. p=" + str(np.round(lr_toa_ts_gl[3], 
                                                      decimals=4)) +
              " sl=" + str(np.round(lr_toa_ts_gl[0], decimals=2)) + 
              " W/m$^2$/K",
              linewidth=lw)
    ax3.plot([np.min(r1_mean_ts[20:]), np.max(r1_mean_ts[20:])],
              np.array([np.min(r1_mean_ts[20:]), np.max(r1_mean_ts[20:])]) * 
              lr_toa_ts_r1[0] + lr_toa_ts_r1[1], c=r1_lr_c, 
              label=r1_acro + " lin. reg. p=" + str(np.round(lr_toa_ts_r1[3], 
                                                     decimals=4)) +
              " sl=" + str(np.round(lr_toa_ts_r1[0], decimals=2)) + 
              " W/m$^2$/K",
              linewidth=lw)
    ax3.plot([np.min(r2_mean_ts[20:]), np.max(r2_mean_ts[20:])],
              np.array([np.min(r2_mean_ts[20:]), np.max(r2_mean_ts[20:])]) * 
              lr_toa_ts_r2[0] + lr_toa_ts_r2[1], c=r2_lr_c, 
              label=r2_acro + " lin. reg. p=" + str(np.round(lr_toa_ts_r2[3], 
                                                     decimals=4)) +
              " sl=" + str(np.round(lr_toa_ts_r2[0], decimals=2)) + 
              " W/m$^2$/K",
              linewidth=lw)
    ax3.set_ylim((-6, 50))
    ax3.set_title(r1_acro + " & "  + r2_acro + " Mean TS v. " + r1_acro + " & "  + r2_acro + " Mean TOA Imbalance " + 
                  mod)
    ax3.set_xlabel("TS in K")
    ax3.set_ylabel("TOA in W/m$^2$")
    ax3.legend()
        
# end def plot_region_3panel()
    
