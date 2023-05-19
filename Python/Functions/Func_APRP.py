"""
Functions for APRP (approximate partial radiation perturbation), see Taylor et
al. (2007)
"""

# imports
import numpy as np
import numexpr as ne
import dask.array as da


# function for calculating the OC quantities
def calc_oc(clt, rsut, rsutcs, rsus, rsuscs, rsds, rsdscs):
    
    # TOA downwelling in-cloud (overcast) radiation - state 1
    rsutoc = ne.evaluate("(rsut - (1 - clt) * rsutcs) / clt")
    #
    # surface downwelling in-cloud (overcast) radiation - state 1
    rsdsoc = ne.evaluate("(rsds - (1 - clt) * rsdscs) / clt")
    #
    # surface upwelling in-cloud (overcast) radiation - state 1
    rsusoc = ne.evaluate("(rsus - (1 - clt) * rsuscs) / clt")
    
    result = dict()
    result["rsutoc"] = rsutoc
    result["rsdsoc"] = rsdsoc
    result["rsusoc"] = rsusoc
    
    return result
# end function calc_oc()

# function for calculating the OC quantities - dask version
def calc_oc_da(clt, rsut, rsutcs, rsus, rsuscs, rsds, rsdscs):
    
    # TOA downwelling in-cloud (overcast) radiation - state 1
    rsutoc = (rsut - (1 - clt) * rsutcs) / clt
    #
    # surface downwelling in-cloud (overcast) radiation - state 1
    rsdsoc = (rsds - (1 - clt) * rsdscs) / clt
    #
    # surface upwelling in-cloud (overcast) radiation - state 1
    rsusoc = (rsus - (1 - clt) * rsuscs) / clt
    
    result = dict()
    result["rsutoc"] = rsutoc.compute()
    result["rsdsoc"] = rsdsoc.compute()
    result["rsusoc"] = rsusoc.compute()
    
    return result
# end function calc_oc_da()
    
# function for correcting the simple model components according to clt
def corr_oc(clt_1, clt_2, oc_1, oc_2, 
            rsdscs_1, rsuscs_1, rsutcs_1, 
            rsdscs_2, rsuscs_2, rsutcs_2, thres=0.01, verbose=False):
    
    # set the state 1 oc coefs to state 2 oc coefs in cells where state 1 clt ~ 0 and vice versa
    s1_clt0 = np.where((clt_1 < thres) & (clt_2 > thres))
    s2_clt0 = np.where((clt_2 < thres) & (clt_1 > thres))
    
    # start a counter for counting the number of corrected cells ("cell count")
    cc = 0
    
    if len(s1_clt0[0]) > 0:
        oc_1["rsutoc"][s1_clt0] = oc_2["rsutoc"][s1_clt0]
        oc_1["rsusoc"][s1_clt0] = oc_2["rsusoc"][s1_clt0]
        oc_1["rsdsoc"][s1_clt0] = oc_2["rsdsoc"][s1_clt0]
        cc += len(s1_clt0[0])
    if len(s2_clt0[0]) > 0:
        oc_2["rsutoc"][s2_clt0] = oc_1["rsutoc"][s2_clt0]
        oc_2["rsusoc"][s2_clt0] = oc_1["rsusoc"][s2_clt0]
        oc_2["rsdsoc"][s2_clt0] = oc_1["rsdsoc"][s2_clt0]
        cc += len(s2_clt0[0])
    # end if
    
    # for cells with clt ~ 0 in both states set the oc coefs to the cs values
    s12_clt0 = np.where((clt_1 < thres) & (clt_2 < thres))

    if len(s12_clt0[0]) > 0:
        oc_1["rsutoc"][s12_clt0] = rsutcs_1[s12_clt0]
        oc_1["rsusoc"][s12_clt0] = rsuscs_1[s12_clt0]
        oc_1["rsdsoc"][s12_clt0] = rsdscs_1[s12_clt0]
        oc_2["rsutoc"][s12_clt0] = rsutcs_2[s12_clt0]
        oc_2["rsusoc"][s12_clt0] = rsuscs_2[s12_clt0]
        oc_2["rsdsoc"][s12_clt0] = rsdscs_2[s12_clt0]
        cc += len(s12_clt0[0])
    # end if
    
    # if requested print the number of corrected cells
    if verbose:
        print("\nCorrected " + str(cc) + " cells.\n")
    # end if
    
    return oc_1, oc_2
    
# end function corr_comp()

# calculate simple model coefficients and components
def simple_comp(clt, rsdt,
                rsds, rsdscs, rsdsoc, 
                rsus, rsuscs, rsusoc,
                rsut, rsutcs, rsutoc):

    # caluclate the net cs and oc downwelling quantities ----------------------
    # TOA net downwelling clear-sly radiation
    qt_net_cs = ne.evaluate("rsdt - rsutcs")
    #
    # TOA net downwelling in-cloud (overcast) radiation
    qt_net_oc = ne.evaluate("rsdt - rsutoc")
    #
    # -------------------------------------------------------------------------
    
    # calculate cs and oc planetary albedo ------------------------------------
    #   
    # planetary albedo clear-sky
    a_cs = ne.evaluate("1 - qt_net_cs / rsdt")
    #
    # planetary albedo in-cloud (only in overcast regions)
    a_oc = ne.evaluate("1 - qt_net_oc / rsdt")
    #
    # -------------------------------------------------------------------------
    
    # calculate cs and oc surface albedo --------------------------------------
    #
    # surface albedo clear-sky
    al_cs = ne.evaluate("rsuscs / rsdscs")
    #
    # surface albedo in-cloud
    al_oc = ne.evaluate("rsusoc / rsdsoc")
    #
    # -------------------------------------------------------------------------
    
    # calculate cs and oc "Q-hat" --------------------------------------------- 
    #
    # "Q-hat" clear-sky (= fraction of the radiation arriving at the surface)
    qh_cs = ne.evaluate("rsdscs / rsdt")
    #
    # "Q-hat" in-cloud (= fraction of the radiation arriving at the surface)
    qh_oc = ne.evaluate("rsdsoc / rsdt")
    #
    # -------------------------------------------------------------------------
    
    # calculate cs and oc mu --------------------------------------------------
    #
    # mu clear-sky (1-mu is atmospheric absorptance)
    mu_cs = ne.evaluate("a_cs + qh_cs * (1 - al_cs)")
    #
    # mu in-cloud (1-mu is atmospheric absorptance)
    mu_oc = ne.evaluate("a_oc + qh_oc * (1 - al_oc)")
    #
    # -------------------------------------------------------------------------
    
    # calculate cs and oc ga --------------------------------------------------
    #
    # gamma clear-sky (gamma is atmospheric scattering coefficient)
    ga_cs = ne.evaluate("((mu_cs - qh_cs) / (mu_cs - al_cs * qh_cs))")
    #
    # gamma in-cloud (gamma is atmospheric scattering coefficient)
    ga_oc = ne.evaluate("((mu_oc - qh_oc) / (mu_oc - al_oc * qh_oc))")
    #
    # -------------------------------------------------------------------------
    
    # calculate oc mu and oc ga -----------------------------------------------
    #
    # mu cloud (1-mu is atmospheric absorptance)
    mu_cl = ne.evaluate("mu_oc / mu_cs")
    #
    # gamma cloud (ga atmospheric scattering coefficient)
    ga_cl = ne.evaluate("1 - (1 - ga_oc) / (1 - ga_cs)")
    #
    # -------------------------------------------------------------------------
    
    # correct the clear sky coefs to range [0, 1]
    mu_cs[np.where(mu_cs < 0)] = 0
    mu_cs[np.where(mu_cs > 1)] = 1
    ga_cs[np.where(ga_cs < 0)] = 0
    ga_cs[np.where(ga_cs > 1)] = 1
    
    # convert NaN values to 0
    mu_cs[np.isnan(mu_cs)] = 0
    mu_cl[np.isnan(mu_cl)] = 0
    mu_oc[np.isnan(mu_oc)] = 0
    ga_cs[np.isnan(ga_cs)] = 0
    ga_cl[np.isnan(ga_cl)] = 0
    ga_oc[np.isnan(ga_oc)] = 0
    al_cs[np.isnan(al_cs)] = 0
    al_oc[np.isnan(al_oc)] = 0
    
    # create a dictionary to store the results
    result = dict()
    
    result["mu_cs"] = mu_cs
    result["mu_cl"] = mu_cl
    result["mu_oc"] = mu_oc
    result["ga_cs"] = ga_cs
    result["ga_cl"] = ga_cl
    result["ga_oc"] = ga_oc
    result["al_cs"] = al_cs
    result["al_oc"] = al_oc
    
    # return
    return result

# end function simple_comp()
    
# function for correcting the simple model components according to clt
def corr_comp(clt_1, clt_2, comp_1, comp_2, thres=0.01, verbose=False):
    
    # set the state 1 oc coefs to state 2 oc coefs in cells where state 1 clt ~ 0 and vice versa
    s1_clt0 = np.where((clt_1 < thres) & (clt_2 > thres))
    s2_clt0 = np.where((clt_2 < thres) & (clt_1 > thres))
    
    # start a counter for counting the number of corrected cells ("cell count")
    cc = 0
    
    if len(s1_clt0[0]) > 0:
        comp_1["al_oc"][s1_clt0] = comp_2["al_oc"][s1_clt0]
        comp_1["mu_oc"][s1_clt0] = comp_2["mu_oc"][s1_clt0]
        comp_1["ga_oc"][s1_clt0] = comp_2["ga_oc"][s1_clt0]
        cc += len(s1_clt0[0])
    if len(s2_clt0[0]) > 0:
        comp_2["al_oc"][s2_clt0] = comp_1["al_oc"][s2_clt0]
        comp_2["mu_oc"][s2_clt0] = comp_1["mu_oc"][s2_clt0]
        comp_2["ga_oc"][s2_clt0] = comp_1["ga_oc"][s2_clt0]
        cc += len(s2_clt0[0])
    # end if
    
    # for cells with clt ~ 0 in both states set the oc coefs to the cs values
    s12_clt0 = np.where((clt_1 < thres) & (clt_2 < thres))

    if len(s12_clt0[0]) > 0:
        comp_1["al_oc"][s12_clt0] = comp_1["al_cs"][s12_clt0]
        comp_1["mu_oc"][s12_clt0] = comp_1["mu_cs"][s12_clt0]
        comp_1["ga_oc"][s12_clt0] = comp_1["ga_cs"][s12_clt0]
        comp_2["al_oc"][s12_clt0] = comp_2["al_cs"][s12_clt0]
        comp_2["mu_oc"][s12_clt0] = comp_2["mu_cs"][s12_clt0]
        comp_2["ga_oc"][s12_clt0] = comp_2["ga_cs"][s12_clt0]
        cc += len(s12_clt0[0])
    # end if
    
    # if requested print the number of corrected cells
    if verbose:
        print("\nCorrected " + str(cc) + " cells.\n")
    # end if
    
    return comp_1, comp_2
    
# end function corr_comp()

# clear-sky component of planetary albedo
def a_cs(mu, ga, al):    
    return ne.evaluate("mu * ga + (mu * al * (1 - ga)**2) / (1 - al * ga)")
# end function a_cs

# in-cloud (overcast) component of planetary albedo
def a_oc(mu_cs, mu_cl, ga_cs, ga_cl, al):
    mu = ne.evaluate("mu_cs * mu_cl")
    ga = ne.evaluate("1 - (1 - ga_cs) * (1 - ga_cl)")
    
    return ne.evaluate("mu * ga + (mu * al * (1 - ga)**2) / (1 - al * ga)")
# end function a_oc

# in-cloud (overcast) component of planetary albedo - "direct" version
def a_oc2(mu, ga, al):
    return ne.evaluate("mu * ga + (mu * al * (1 - ga)**2) / (1 - al * ga)")
# end function a_oc2
    
# planetary albedo from components and clt
def plan_al(clt, a_cs, a_oc):
    return ne.evaluate("(1 - clt) * a_cs + clt * a_oc")
# end plan_al
    

# calculate the three components of the planetary albedo change
def pa_ch(clt_1, clt_2, comp_1, comp_2, method="12b"):
    
    """
    For the method parameter see Taylor et al. (2007) equations 12 b and c.
    """
    
    # calculate planetary albedo for state 1
    pa_1 = plan_al(clt_1, 
                   a_cs(comp_1["mu_cs"], comp_1["ga_cs"], 
                        comp_1["al_cs"]), 
                   a_oc(comp_1["mu_cs"], comp_1["mu_cl"], 
                        comp_1["ga_cs"], comp_1["ga_cl"], 
                        comp_1["al_oc"]))
    
    # calculate planetary albedo for state 2
    pa_2 = plan_al(clt_2, 
                   a_cs(comp_2["mu_cs"], comp_2["ga_cs"], 
                        comp_2["al_cs"]), 
                   a_oc(comp_2["mu_cs"], comp_2["mu_cl"], 
                        comp_2["ga_cs"], comp_2["ga_cl"], 
                        comp_2["al_oc"]))                    
    
    # calculate planetary albedo for state 1 with state 2 cld mu
    pa_1mu_cl2 = plan_al(clt_1, 
                         a_cs(comp_1["mu_cs"], comp_1["ga_cs"], 
                              comp_1["al_cs"]), 
                         a_oc(comp_1["mu_cs"], comp_2["mu_cl"], 
                              comp_1["ga_cs"], comp_1["ga_cl"], 
                              comp_1["al_oc"]))
    
    # calculate planetary albedo for state 2 with state 1 cld mu
    pa_2mu_cl1 = plan_al(clt_2, 
                         a_cs(comp_2["mu_cs"], comp_2["ga_cs"], 
                              comp_2["al_cs"]), 
                         a_oc(comp_2["mu_cs"], comp_1["mu_cl"], 
                              comp_2["ga_cs"], comp_2["ga_cl"], 
                              comp_2["al_oc"]))
                        
    # calculate planetary albedo for state 1 with state 2 cld ga
    pa_1ga_cl2 = plan_al(clt_1, 
                         a_cs(comp_1["mu_cs"], comp_1["ga_cs"], 
                              comp_1["al_cs"]), 
                         a_oc(comp_1["mu_cs"], comp_1["mu_cl"], 
                              comp_1["ga_cs"], comp_2["ga_cl"], 
                              comp_1["al_oc"]))
    
    # calculate planetary albedo for state 2 with state 1 cld ga
    pa_2ga_cl1 = plan_al(clt_2, 
                         a_cs(comp_2["mu_cs"], comp_2["ga_cs"], 
                              comp_2["al_cs"]), 
                         a_oc(comp_2["mu_cs"], comp_2["mu_cl"], 
                              comp_2["ga_cs"], comp_1["ga_cl"], 
                              comp_2["al_oc"]))
                        
    # calculate planetary albedo for state 1 with state 2 clt
    pa_1_clt2 = plan_al(clt_2, 
                        a_cs(comp_1["mu_cs"], comp_1["ga_cs"], 
                             comp_1["al_cs"]), 
                        a_oc(comp_1["mu_cs"], comp_1["mu_cl"], 
                             comp_1["ga_cs"], comp_1["ga_cl"], 
                             comp_1["al_oc"]))
    
    # calculate planetary albedo for state 2 with state 1 clt
    pa_2_clt1 = plan_al(clt_1, 
                        a_cs(comp_2["mu_cs"], comp_2["ga_cs"], 
                             comp_2["al_cs"]), 
                        a_oc(comp_2["mu_cs"], comp_2["mu_cl"], 
                             comp_2["ga_cs"], comp_2["ga_cl"], 
                             comp_2["al_oc"]))
                        
    # calculate the planetary albedo for state 1 with state 2 clear-sky alpha
    pa_1_alcs2 = plan_al(clt_1, 
                         a_cs(comp_1["mu_cs"], comp_1["ga_cs"], 
                              comp_2["al_cs"]), 
                         a_oc(comp_1["mu_cs"], comp_1["mu_cl"], 
                              comp_1["ga_cs"], comp_1["ga_cl"], 
                              comp_1["al_oc"]))
                         
    # calculate the planetary albedo for state 2 with state 1 clear-sky alpha
    pa_2_alcs1 = plan_al(clt_2, 
                         a_cs(comp_2["mu_cs"], comp_2["ga_cs"], 
                              comp_1["al_cs"]), 
                         a_oc(comp_2["mu_cs"], comp_2["mu_cl"], 
                              comp_2["ga_cs"], comp_2["ga_cl"], 
                              comp_2["al_oc"]))
    
    # calculate the planetary albedo for state 1 with state 2 overcast alpha
    pa_1_aloc2 = plan_al(clt_1, 
                         a_cs(comp_1["mu_cs"], comp_1["ga_cs"], 
                              comp_1["al_cs"]), 
                         a_oc(comp_1["mu_cs"], comp_1["mu_cl"], 
                              comp_1["ga_cs"], comp_1["ga_cl"], 
                              comp_2["al_oc"]))
                         
    # calculate the planetary albedo for state 2 with state 1 overcast alpha
    pa_2_aloc1 = plan_al(clt_2, 
                         a_cs(comp_2["mu_cs"], comp_2["ga_cs"], 
                              comp_2["al_cs"]), 
                         a_oc(comp_2["mu_cs"], comp_2["mu_cl"], 
                              comp_2["ga_cs"], comp_2["ga_cl"], 
                              comp_1["al_oc"]))
                         
    # calculate planetary albedo for state 1 with state 2 cs mu
    pa_1mu_cs2 = plan_al(clt_1, 
                         a_cs(comp_1["mu_cs"], comp_1["ga_cs"], 
                              comp_1["al_cs"]), 
                         a_oc(comp_2["mu_cs"], comp_1["mu_cl"], 
                              comp_1["ga_cs"], comp_1["ga_cl"], 
                              comp_1["al_oc"]))
    
    # calculate planetary albedo for state 2 with state 1 cs mu
    pa_2mu_cs1 = plan_al(clt_2, 
                         a_cs(comp_2["mu_cs"], comp_2["ga_cs"], 
                              comp_2["al_cs"]), 
                         a_oc(comp_1["mu_cs"], comp_2["mu_cl"], 
                              comp_2["ga_cs"], comp_2["ga_cl"], 
                              comp_2["al_oc"]))
                         
    # calculate planetary albedo for state 1 with state 2 cld ga
    pa_1ga_cs2 = plan_al(clt_1, 
                         a_cs(comp_1["mu_cs"], comp_1["ga_cs"], 
                              comp_1["al_cs"]), 
                         a_oc(comp_1["mu_cs"], comp_1["mu_cl"], 
                              comp_2["ga_cs"], comp_1["ga_cl"], 
                              comp_1["al_oc"]))
    
    # calculate planetary albedo for state 2 with state 1 cld ga
    pa_2ga_cs1 = plan_al(clt_2, 
                         a_cs(comp_2["mu_cs"], comp_2["ga_cs"], 
                              comp_2["al_cs"]), 
                         a_oc(comp_2["mu_cs"], comp_2["mu_cl"], 
                              comp_1["ga_cs"], comp_2["ga_cl"], 
                              comp_2["al_oc"]))
    
    if method == "12b":
        # calculate the planetary albedo change due to mu cld
        del_pa_mu_cl = ne.evaluate("1/2 * (pa_1mu_cl2 - pa_1) + 1/2 * (pa_2 - pa_2mu_cl1)")
        
        # calculate the planetary albedo change due to ga cld
        del_pa_ga_cl = ne.evaluate("1/2 * (pa_1ga_cl2 - pa_1) + 1/2 * (pa_2 - pa_2ga_cl1)")
        
        # calculate the planetary albedo change due to mu cld
        del_pa_clt = ne.evaluate("1/2 * (pa_1_clt2 - pa_1) + 1/2 * (pa_2 - pa_2_clt1)")
        
        
        # calculate the planetary albedo change due to clear-sky alpha
        del_pa_alcs = ne.evaluate("1/2 * (pa_1_alcs2 - pa_1) + 1/2 * (pa_2 - pa_2_alcs1)")
    
        # calculate the planetary albedo change due to clear-sky alpha
        del_pa_aloc = ne.evaluate("1/2 * (pa_1_aloc2 - pa_1) + 1/2 * (pa_2 - pa_2_aloc1)")
        
        
        # calculate the planetary albedo change due to clear-sky mu
        del_pa_mucs = ne.evaluate("1/2 * (pa_1mu_cs2 - pa_1) + 1/2 * (pa_2 - pa_2mu_cs1)")
        
        # calculate the planetary albedo change due to clear-sky ga
        del_pa_gaoc = ne.evaluate("1/2 * (pa_1ga_cs2 - pa_1) + 1/2 * (pa_2 - pa_2ga_cs1)")
        
    if method == "12c1":
        # calculate the planetary albedo change due to mu cld
        del_pa_mu_cl = ne.evaluate("pa_1mu_cl2 - pa_1")
        
        # calculate the planetary albedo change due to ga cld
        del_pa_ga_cl = ne.evaluate("pa_1ga_cl2 - pa_1")

        # calculate the planetary albedo change due to mu cld
        del_pa_clt = ne.evaluate("pa_1_clt2 - pa_1")
        
        
        # calculate the planetary albedo change due to clear-sky alpha
        del_pa_alcs = ne.evaluate("pa_1_alcs2 - pa_1")
        
        # calculate the planetary albedo change due to clear-sky alpha
        del_pa_aloc = ne.evaluate("pa_1_aloc2 - pa_1")
        

        # calculate the planetary albedo change due to clear-sky mu
        del_pa_mucs = ne.evaluate("pa_1mu_cs2 - pa_1")
    
        # calculate the planetary albedo change due to clear-sky ga
        del_pa_gaoc = ne.evaluate("pa_1ga_cs2 - pa_1")
        
    # end if
    if method == "12c2":
        # calculate the planetary albedo change due to mu cld
        del_pa_mu_cl = ne.evaluate("pa_2 - pa_2mu_cl1")
        
        # calculate the planetary albedo change due to ga cld
        del_pa_ga_cl = ne.evaluate("pa_2 - pa_2ga_cl1")

        # calculate the planetary albedo change due to mu cld
        del_pa_clt = ne.evaluate("pa_2 - pa_2_clt1")
        

        # calculate the planetary albedo change due to clear-sky alpha
        del_pa_alcs = ne.evaluate("pa_2 - pa_2_alcs1")
        
        # calculate the planetary albedo change due to clear-sky alpha
        del_pa_aloc = ne.evaluate("pa_2 - pa_2_aloc1")
        
        
        # calculate the planetary albedo change due to clear-sky mu
        del_pa_mucs = ne.evaluate("pa_2 - pa_2mu_cs1")
    
        # calculate the planetary albedo change due to clear-sky ga
        del_pa_gaoc = ne.evaluate("pa_2 - pa_2ga_cs1")
    # end if    
    
    # sum up the individual deltas to get the total delta of the planetary 
    # albedo due to clouds (the final APRP)
    del_pa_cld = ne.evaluate("del_pa_mu_cl + del_pa_ga_cl + del_pa_clt")
    
    # sum up the individual deltas to get the total delta of the planetary 
    # albedo due to clouds (the final APRP)
    del_pa_sfc = ne.evaluate("del_pa_alcs + del_pa_aloc")
    
    # sum up the individual deltas to get the total delta of the planetary
    # albedo due to clouds (the final APRP)
    del_pa_ncl = ne.evaluate("del_pa_mucs + del_pa_gaoc")
    
    return del_pa_cld, del_pa_sfc, del_pa_ncl

# end function
    
    
    