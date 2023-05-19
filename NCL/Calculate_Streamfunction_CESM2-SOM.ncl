begin

;; calculate the stream function for CESM runs
;; --> makes possible the calculation of the Hadley cell strength

case = "dQ01yr_4xCO2"

data_path = ""
out_path = ""

;; generate the output directory
test = systemfunc("mkdir " + out_path)

f_list = systemfunc("ls " + data_path + "/V_Files/*.nc | sort")

;; load the surface pressure file
ps_file = systemfunc("ls " + data_path + "PS_*.nc")
pf = addfile(ps_file, "r")

i = 0
n_time_o = 0
do_n = dimsizes(f_list) - 1
do while(i.le.do_n)
	;; print(i)
        in_file = f_list(i)

	;; print(in_file)

	;; split up the input file name at "_" and use the different parts to build up the output file name
	in_split = str_split(in_file, "_")
	in_split = in_split(::-1)  ;; invert the order of elements
	f_end = in_split(0)
	
	out_file = out_path + "/mpsi_" + case + "_" + f_end

	;; load the file
	vaf = addfile(in_file, "r")

	va_red = vaf->V(:, 3:, :, :)
	;; va_red = vai_red(:, ::-1, :, :)  --> not needed for CESM2-SOM output data

	dims = dimsizes(va_red)
	n_time_n = n_time_o + dims(0) - 1 ;; note that indexing is different than in Python: in NCL the last index is included
	print(dims)	

	ps_red = pf->PS(n_time_o:n_time_n, :, :)
	print(dimsizes(ps_red))
	plev_red = vaf->lev(3:)
	plev_red = plev_red * 100
	;; print(plev_red)
	;; plev_red = plev(3:)
	;; plev_red = plev_red(::-1)  --> not needed for CESM2-SOM output data
	
	print("Data loaded. Proceeding to streamfunction calculation.")

	lat = vaf->lat
	lon = vaf->lon
	time_red = vaf->time
	;; time_red = time(:n_time)
	print(dimsizes(time_red))

	;; number of pressure levels and latitudes and time steps
	n_plev = dimsizes(plev_red)
	n_lat = dimsizes(lat)
	;; n_time = dimsizes(time)

	;; calculate the stream function
	mo_streamf = zonal_mpsi_Wrap(va_red, lat, plev_red, ps_red)  ;; _Wrap is used to retain the attributes
	print("Streamfunction calculated. Proceeding to storing data.")


	;; store the streamfunction as netcdf file

	;; Define global attributes for output file 
	globeAtt               = 1
	globeAtt@title         = "meridional overturning streamfunction as calculated from NCL function zonal_mpsi"
	globeAtt@source_file   = in_file
	globeAtt@creation_date = systemfunc ("date")

	;; Open the output netcdf file and provide attributes
	system ("/bin/rm -f " +  out_file)    ; remove any pre-existing file
	ncdf   = addfile(out_file, "c")
	fileattdef(ncdf, globeAtt)        ; create file attributes

	;; Define dimensions
	dim_names = (/"time", "plev", "lat"/)
	dim_sizes = (/n_time_n-n_time_n, n_plev, n_lat/)
	dim_unlimited = (/True, False, False/)

	;; Write coordinate variables
	ncdf->time = time_red
	ncdf->plev = plev_red
	ncdf->lat = lat
	ncdf->mpsi = mo_streamf
  	
	n_time_o = n_time_n + 1
	i=i+1
	delete(va_red)
	delete(ps_red)
	delete(time_red)
	delete(mo_streamf)
end do


end
