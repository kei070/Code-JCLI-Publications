begin

;; calculate the stream function for CESM runs
;; --> makes possible the calculation of the Hadley cell strength

model = "SAM0-UNICON"
cmip = "6"
print(model)
forc_exp = "abrupt-4xCO2"
;; forc_exp = "piControl"
;; ens_dr = "_r1i1p1f2"
ens_dr = ""
;; rg = "_rg"
rg = ""

data_path = ""
out_path = ""

;; generate the output directory
test = systemfunc("mkdir " + out_path)

f_list = systemfunc("ls " + data_path + forc_exp + "_va_Files" + ens_dr + "/*.nc | sort")

;; load the surface pressure file
ps_file = systemfunc("ls " + data_path + "ps" + rg + "_Amon_*_" + forc_exp + "*")
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
	in_split = in_split(::-1)
	f_end = in_split(0)
	gr_tag = in_split(1)
	ens = in_split(2)
	
	out_file = out_path + "/" + model + "_mpsi_" + forc_exp + "_" + ens + "_" + gr_tag + "_" + f_end

	;; load the file
	vaf = addfile(in_file, "r")

	va_red = vaf->va(:, :16, :, :)
	va_red = va_red(:, ::-1, :, :)

	dims = dimsizes(va_red)
	n_time_n = n_time_o + dims(0) - 1 ;; note that indexing is different than in Python: in NCL the last index is included
	print(dims)	

	ps_red = pf->ps(n_time_o:n_time_n, :, :)
	print(dimsizes(ps_red))
	plev_red = vaf->plev(:16)
	;; plev_red = plev(:16)
	plev_red = plev_red(::-1)
	
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
