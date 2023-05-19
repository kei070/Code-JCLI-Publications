import copy
from netCDF4 import Dataset
import numpy as np
import pylab as pl
import cartopy
import cartopy.util as cu
import cartopy.crs as ccrs
import cartopy.feature as cfeature


def extract_region(x1, x2, y1, y2, lat, lon, field, test_plt=False, plot_title="", cen_lat=180):
    
    """
    Function for extracting a given geographic region from a given field
    
    Parameters:
        :param x1: Left hand side longitude.
        :param x2: Right hand side longitude.
        :param y1: Lower latitude.
        :param y2: Upper latitude.
        :param lat: 1d array of latitudes of field. Range -90 to +90.
        :param lon: 1d array of longitudes of field. Range -180 to +180.
        :param field: Array from which the geographic region is to be extracted. Can have 2, 3, or 4 dimensions. lat and 
                      lon must be the two last dimensions in this order.
        :param test_plt: Logical. If True, a test plot will be drawn with the part masked (value=np.nan) that will be 
                         extraced. Defaults to False.
        :param plot_title: String. If test_plt=True this will be used as the title for the test plot. Defaults to "".                         
    Return: 
        1. The extracted part of the input field.
        2. The array of the lat and lon indices: [x1_i, x2_i, y1_i, y2_i]
        
    #  x1,y2.....x2,y2  
    #    .         .
    #    .         .      East ->
    #    .         .
    #  x1,y1.....x2,y1
        
    """
    
    # if the lat array is the "wrong way around", i.e., lat[0] is ca. -90, flip it and the field
    if lat[0] > lat[-1]:
        print("\nlat array is 'the wrong way around'; flipping it and the field\n")
        lat = lat[::-1]
        
        if len(np.shape(field)) == 2:
            field = field[::-1, :]
        elif len(np.shape(field)) == 3:
            field = field[:, ::-1, :]
        elif len(np.shape(field)) == 4:
            field = field[:, :, ::-1, :]
        # end if elif
    # end if
    
    # implement some checks on the given coordinate points
    if y1 > y2:
        raise Exception('Coordinate Error', 'y1 must be < y2')
    # end if
    
    # check if lat and lon conform to their intervals
    if np.abs(np.max(lat)) > 90:
        raise Exception('Grid Error', 'Lat interval is not -90 to +90')
    if np.abs(np.max(lon)) > 180:
        # raise Exception('Grid Error', 'Lon interval is not -180 to +180')
        print("Warning: Note that your lon range is 0 to 360! Be sure to choose your x1 and x2 values accordingly!")
    # end if
        
    y1_i = np.argmin(abs(lat - y1))
    y2_i = np.argmin(abs(lat - y2)) + 1
    y_i = np.arange(y1_i, y2_i)

    if x1 > x2: 
        x1_i = np.argmin(abs(lon - x1))
        x2_i = np.argmin(abs(lon - x2)) + 1

        x_i = np.concatenate((np.arange(x1_i, len(lon)), np.arange(0, x2_i)))
        # raise Exception('Coordinate Error', 'x1 must be < x2')
    else:
        x1_i = np.argmin(abs(lon - x1))
        x2_i = np.argmin(abs(lon - x2)) + 1    
        
        x_i = np.arange(x1_i, x2_i)
    # end if else        
    
    x_m, y_m = np.meshgrid(x_i, y_i)
    
    
    if len(np.shape(field)) == 2:
        if test_plt:
            test_pl = copy.deepcopy(field)
            test_pl[y_m, x_m] = np.nan
            try:
                x, y = np.meshgrid(lon, lat)
                proj = ccrs.Robinson(central_longitude=cen_lat)
                fig, ax1 = pl.subplots(ncols=1, nrows=1, figsize=(12, 6), subplot_kw=dict(projection=proj))
                p1 = ax1.contourf(x, y, test_pl, transform=ccrs.PlateCarree(), extend="both")
                ax1.coastlines(resolution="50m", color="black", linewidth=0.5, zorder=101)
                ax1.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
                ax1.set_global()
                pl.colorbar(p1, ax=ax1)
                ax1.set_title(plot_title)
                pl.show()
                pl.close()                
            except:
                print("\nCartopy map could not be drawn, using imshow() without geographic reference...\n")              
                pl.imshow(test_pl, origin="lower")
                pl.colorbar()
                pl.title(plot_title)
                pl.show()
            # end try except                
        # end if
        return [field[y_m, x_m], [y_m, x_m]]
    elif len(np.shape(field)) == 3:
        if test_plt:
            test_pl = copy.deepcopy(field)
            test_pl[:, y_m, x_m] = np.nan
            try:
                x, y = np.meshgrid(lon, lat)
                proj = ccrs.Robinson(central_longitude=cen_lat)
                fig, ax1 = pl.subplots(ncols=1, nrows=1, figsize=(12, 6), subplot_kw=dict(projection=proj))
                p1 = ax1.contourf(x, y, test_pl[0, :, :], transform=ccrs.PlateCarree(), extend="both")
                ax1.coastlines(resolution="50m", color="black", linewidth=0.5, zorder=101)
                ax1.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
                ax1.set_global()
                pl.colorbar(p1, ax=ax1)
                ax1.set_title(plot_title)
                pl.show()
                pl.close()                
            except:
                print("\nCartopy map could not be drawn, using imshow() without geographic reference...\n")            
                pl.imshow(test_pl[0, :, :], origin="lower")
                pl.colorbar()
                pl.title(plot_title)
                pl.show()
            # end try except    
        # end if
        return [field[:, y_m, x_m], [y_m, x_m]]
    elif len(np.shape(field)) == 4:
        if test_plt:
            test_pl = copy.deepcopy(field)
            test_pl[:, :, y_m, x_m] = np.nan
            try:
                x, y = np.meshgrid(lon, lat)
                proj = ccrs.Robinson(central_longitude=cen_lat)
                fig, ax1 = pl.subplots(ncols=1, nrows=1, figsize=(12, 6), subplot_kw=dict(projection=proj))
                p1 = ax1.contourf(x, y, test_pl[0, 0, :, :], transform=ccrs.PlateCarree(), extend="both")
                ax1.coastlines(resolution="50m", color="black", linewidth=0.5, zorder=101)
                ax1.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
                ax1.set_global()
                pl.colorbar(p1, ax=ax1)
                ax1.set_title(plot_title)
                pl.show()
                pl.close()                
            except:
                print("\nCartopy map could not be drawn, using imshow() without geographic reference...\n")
                pl.imshow(test_pl[0, 0, :, :], origin="lower")
                pl.colorbar()
                pl.title(plot_title)
                pl.show()
            # end try except    
        # end if
        return [field[:, :, y_m, x_m], [y_m, x_m]]
    # end if elif
    
# end def extract_region()


""" test the function

# set path to some file
data_path = "/media/kei070/Seagate/Uni/PhD/Tromsoe_UiT/Work/CMIP6/Data/CanESM5/"
f_name = "tas_Amon_CanESM5_abrupt-4xCO2_r1i1p1f1_gn_185001-200012.nc"

# load the file
nc = Dataset(data_path + f_name)

# load the data, lat, and lon
field = nc.variables["tas"][0, :, :]
lat = nc.variables["lat"][:]
lon = nc.variables["lon"][:]

# set up some coordinates
x1 = 0
x2 = 360
y1 = 75
y2 = 90

# extract the region
field_reg = extract_region(x1, x2, y1, y2, lat, lon, field, test_plt=True, plot_title="Test", cen_lat=180)

# extract the indices for max and min for lat and lon
min_lat_i = field_reg[1][0][0, 0]
min_lon_i = field_reg[1][1][0, 0]
max_lat_i = field_reg[1][0][-1, 0]
max_lon_i = field_reg[1][1][0, -1]

# print the max and min lat and lon values
try:
    print(f"lat[min-1]: {lat[min_lat_i-1]}")
except:
    print("Nothing to print for lat[min-1]")
# end try except    
print(f"lat[min]: {lat[min_lat_i]}")
try:
    print(f"lat[min+1]: {lat[min_lat_i+1]}")
except:
    print("lat[min+1]: nothing to print")
# end try except
print("\n")
try:
    print(f"lat[max-1]: {lat[max_lat_i-1]}")
except:
    print("lat[max-1]: nothing to print")
# end try except    
print(f"lat[max]: {lat[max_lat_i]}")
try:
    print(f"lat[max+1]: {lat[max_lat_i+1]}")
except:
    print("lat[max+1]: nothing to print")
# end try except

print("\n")
print(f"lon[min]: {lon[min_lon_i]}")
print(f"lon[max]: {lon[max_lon_i]}")

"""