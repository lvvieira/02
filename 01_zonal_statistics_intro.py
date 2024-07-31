#!/usr/bin/env python
# coding: utf-8

# # Zonal Statistics

# RSGISLib is a set of Python modules which have been developed over the years to support our data processing needs.
# 
# The modules provide a set of high level functions, at a similar level to ArcGIS / QGIS toolbox functions or tools in ENVI or Erdas Imagine. The idea is that you can join a number of functions together to undertake your analysis or put those functions within a loop and process a number of input images.
# 
# In this tutorial we will look at populating various vector formats (points and polygons) from raster images for looking at fallow, growing and cropping cycles in agricultural/rice paddies in Vietnam, using radar imagery. Radar data is measured in backscatter where high values are associated with high structure (vegetation) and low values are associated with low structure (non_vegetated/water/bare).
# 
# You will learn the following:
# 
# - Extract raster values to point for specifc locations
# - Extract raster statistics to buffered points (polygons)
# - Extract raster stack statistics for agricultural fields (image objects)
# - Explore extracted time-series signal

# ## Running Notebook
# 
# The notebook has been run and saved with the outputs so you can see what the outputs should be and so the notebook and be browsed online without having to run the notebook for it to make sense. 
# 
# If you are running the notebook for yourself it is recommended that you clear the existing outputs which can be done by running one of the following options depending on what system you are using:
# 
# **Jupyter-lab**:

# > \> _Edit_ \> _'Clear All Outputs'_
# 
# **Jupyter-notebook**:
# 
# > \> _Cell_ \> _'All Outputs'_ \> _Clear_
# 
# 

# # 1. Importing Modules
# 
# The first step is to import the modules for the analysis we are to undertake:

# Import Geopandas for vector analysis (https://geopandas.org)
import geopandas

# Additional modules for visualization
import matplotlib
import matplotlib.pyplot as plt

# Import Numpy array module
import numpy

# Import rsgislib modules
import rsgislib
import rsgislib.tools.mapping

# Note, you can import a list of modules rather than 
# listing each individually.
from rsgislib import imageutils, vectorutils, zonalstats


# # 2. Data
# 
# Firstly, lets look at our imagery. Here you will find a stack of Sentinel-1 radar imagery captured over a small region of the Mekong Delta in Vietnam. Each band of the image is a monthly average over the course of a year, from January through December. The image is is linear radar units (power) but is scaled by 10,000. We can look at some basic image metadata and parameters using:

input_img = "../data/Sentinel1_MonthlyMean_2018_lin_100000_Geog_subset.tif"


# Call function to get the input image size:
x_size, y_size = rsgislib.imageutils.get_img_size(input_img)
print(f"Image dimensions: {x_size} x {y_size}")

# Call function to get the input image number of bands:
n_bands = rsgislib.imageutils.get_img_band_count(input_img)
print(f"Number of Bands: {n_bands}")

# Call function to get the input image band names:
band_names = rsgislib.imageutils.get_band_names(input_img)
print(f"Band Names: {band_names}")

# Call function to get the input image resolution:
x_res, y_res = rsgislib.imageutils.get_img_res(input_img)
print(f"Image Resolution: ({round(x_res, 2)}, {round(y_res,2)}) m")


# A point shapefile has been provided for you, which we will use to extrcat pixel values.

points_vec_file = "../data/sample_locations_utm.gpkg"


# The first step is to check the number of field points that we have

feat_num = vectorutils.get_vec_feat_count(points_vec_file)
print(f"number of points: {feat_num}")


# Let us also use rsgislib tools to visualize the radar image with the points overlain. Try changing the band combination `img_bands=[1,2,3]` to visualise the different bands. Try `img_bands=[1]`, `img_bands=[2]`, `img_bands=[3]` and `img_bands=[6,7,8]`, for example.

# Create a plot using the matplotlib library
fig, ax = plt.subplots(figsize=(10, 10))
# Add the stretched image data to the plot and define the spatial
# extent so the axis labels are the coordinates.
rsgislib.tools.mapping.create_raster_img_map(ax, input_img, img_bands=[1, 2, 3], img_stch=rsgislib.IMG_STRETCH_CUMULATIVE)

# Read points to geopandas dataframe
points_gdf = geopandas.read_file(points_vec_file)
# Plot the plots over the image data
points_gdf.plot(ax=ax, color="yellow")


# # 3. Extract values to point

# Firsty, let us consider the scenario whereby image values are needed at field plot locations. For a band of interest we can extract the pixel values at these point locations. 
# 
# We will start by reading the vector points into memory so it is stored as a vector object, using:

# Get list of layer names in vector file
vec_lyrs = rsgislib.vectorutils.get_vec_lyrs_lst(points_vec_file)
print(f"list of vector layers: {vec_lyrs}")

# Get first name from vec_lyrs list
lyr_name = vec_lyrs[0]

# Read vector layer to memory
vec_ds_obj, vec_lyr_obj = rsgislib.vectorutils.read_vec_lyr_to_mem(
    points_vec_file, lyr_name
)


# This creates a vector layer objet (vec_lyr_obj) which stores the vector in memory. Next we will define the input parameters for the extract to point command.

# The image band in the raster stack
img_band = 2
# The minimum value that the point will extract
min_thres = 1
# The maximum value that the point will extract
max_thres = 10000
# The no data value which will not be extracted, but will be used in the case where there is no valid pixel value
out_no_data_val = 0
# The name of the output field in the att table
out_field = "February"
# whether to reproject the vector on-the-fly to match the image
reproj_vec = False
# Specify the epsg for the vector if the WKT is not well defined
vec_def_epsg = None

rsgislib.zonalstats.ext_point_band_values(
    vec_lyr_obj,
    input_img,
    img_band,
    min_thres,
    max_thres,
    out_no_data_val,
    out_field,
    reproj_vec,
    vec_def_epsg,
)


# Now that we have extrcated the pixel values, the final step is to write the vector layer back to the points shapefile.

# The layer in memory which now had pixel attributes
mem_lyr = vec_lyr_obj
# the output file to write the data to.
points_stats_vec_file = "sample_locations_utm_stats.gpkg"
# OGR layer name (matches input layer name in this case)
layer_name = lyr_name
# OGR format. Should match input type if overwriting
ogr_format = "GPKG"

rsgislib.vectorutils.write_vec_lyr_to_file(
    mem_lyr,
    points_stats_vec_file,
    layer_name,
    ogr_format,
    options=["OVERWRITE=YES", "SPATIAL_INDEX=YES"],
)


# While this shows how layers can be read into memory, populated and written back to a shapefile (GeoPackage), this is mostly used when lots of image values are populated into dataset in one go. RSGISLib also has a function to achieve our workflow in one step using the command: ext_point_band_values
# 
# Now we have written the updated layer to the dataset, we can open the vector and look at the values we have extracted:

# Open the gpkg file with geopandas
gdf = geopandas.read_file(points_stats_vec_file)

# View the first 5 attributes
gdf.head()


# # 4. Extract image statistics within a polygon

# Extracting values to a point location is useful, but it is not always known if the pixel value represents the point location accuratley. Often it is preferred that a number of pixels are sampled and represented by a single statistic (e.g., mean) which may be more representative and reduces error due to geolocation etc.
# 
# To do this, firstly we need to buffer our points to get polygons. We will buffer our points by 100 m.
# 
# The points vector is already open from the above step:
# 
# gdf = gpd.read_file(points_vec_file)
# 
# Therefore we can buffer the points with:

# Buffer the points geometry with 100 m
gdf["geometry"] = gdf.geometry.buffer(100)

# Buffered points output file name and path
buffered_points_vec_file = "sample_locations_utm_buffer100.gpkg"

# Write the buffered points to a new file with points buffered by 100 m
gdf.to_file(buffered_points_vec_file, "GPKG")


# We can visualize the buffered points (yellow) behind our original points on a figure using:

# Create a plot using the matplotlib library
fig, ax = plt.subplots(figsize=(10, 10))
# Add the image to the axis
rsgislib.tools.mapping.create_raster_img_map(ax, input_img, img_bands=[1, 2, 3], img_stch=rsgislib.IMG_STRETCH_CUMULATIVE)

# Read points to geopandas dataframe
buff_points_gdf = geopandas.read_file(buffered_points_vec_file)

# Plot the buffered plots and points over the image data
buff_points_gdf.plot(ax=ax, color="yellow")
points_gdf.plot(ax=ax, color="red", markersize=5)


# Now that we have buffered points we can now use commands within RSGISLib to extract image statistics from image pixels that are overlapped by the polygons. Each pixel below the polygon will be included in the statistics, provided that the polygon intersects the pixel centroid (middle point).
# 
# We begin again by getting the vector layer name that we want to use:

# Get list of layer names in vector file
vec_lyrs = rsgislib.vectorutils.get_vec_lyrs_lst(buffered_points_vec_file)
# Get first name from vec_lyrs list
lyr_name = vec_lyrs[0]
print(f"layer name: {lyr_name}")


# As we know that the pixel size is smaller than our polygons, we can use the following command to get the min, max and mean pixel values. Here we will not read the layer into memory first as we extracting values from only one image band so we will use the following command:

# vector file
vec_file = buffered_points_vec_file
# vector layer name
vec_lyr = lyr_name
# raster stack image band
img_band = 2
# The minimum value that the point will extract
min_thres = 1
# The maximum value that the point will extract
max_thres = 10000
# The no data value which will not be extracted, but will be used in the case where there is no valid pixel value
out_no_data_val = 0
# The name of the output fields to write to the att table
mean_field_name = "Feb_mean"
min_field_name = "Feb_min"
max_field_name = "Feb_max"
# Specify the epsg for the vector if the WKT is not well defined
vec_def_epsg = None

rsgislib.zonalstats.calc_zonal_band_stats_test_poly_pts_file(
    vec_file=buffered_points_vec_file,
    vec_lyr=lyr_name,
    input_img=input_img,
    img_band=2,
    min_thres=1,
    max_thres=10000,
    out_no_data_val=0,
    min_field=min_field_name,
    max_field=max_field_name,
    mean_field=mean_field_name,
)


# We can now look at these pixel stats in the attribute table of the vector

# Open the gpkg file with geopandas
gdf = geopandas.read_file(buffered_points_vec_file)

# View the first 5 attributes
gdf.head()


# Here we can see the column 'february' which was extracted to the points vector file and the three february statistics that were populated inot the polygons (buffered points)

# # 5. Time-series zonal stats analysis

# We have seen how you can populate polygons with statistics from an image. Now we will look at an applied example, where we will look at time-series signals from agricultural fields to understand cropping cycles. We will look specifically at a region in the Mekong Delta of Vietnam and evaluate how many cropping cycles there are over the course of a year. This example is taken from work in:  https://doi.org/10.3390/rs12203459
# 
# You have been provided with a shapefile of field boundaries created from VHR Worldview imagery. We will begin by looking at this shapefile and the Sentinal-1 radar imagery that we have used so far in this tutorial.

# ---
# In this case as the GPKG file with the field boundaries is quite large as an uncompressed file we will read it as a compressed dataset. Note, to do this we have to reference the file path slightly differently, specifying the path as a zip compressed file and then the path within the zip file to the GPKG file we are wanting to open.
# 
# You cannot write to a zipped vector layer but if the layer is only going to be used for reading and not to be written to it can save a lot of disk space! In this case `tile_segs_mskd_lbl_vec111.gpkg.zip` is **87 %** smaller than `tile_segs_mskd_lbl_vec111.gpkg`.
# 
# ---

# Field polygons - Note. /vsizip/ specifies that we are openning a zip file.
# The path to the zip file is: ../data/tile_segs_mskd_lbl_vec111.gpkg.zip
# The path to the GPKG file we want to open within the zip file is:
# tile_segs_mskd_lbl_vec111.gpkg
in_field_vec = (
    "/vsizip/../data/tile_segs_mskd_lbl_vec111.gpkg.zip/tile_segs_mskd_lbl_vec111.gpkg"
)

# Read points to geopandas dataframe
field_polys = geopandas.read_file(in_field_vec)

# We already opened the image and calculated the image stretch parameters from
# stretch parameters from the first visualization

# Create a plot using the matplotlib library
fig, ax = plt.subplots(figsize=(10, 10))
# Add the image to the axis
rsgislib.tools.mapping.create_raster_img_map(ax, input_img, img_bands=[1, 2, 3], img_stch=rsgislib.IMG_STRETCH_CUMULATIVE)


# Plot the fields over the image data
field_polys.plot(ax=ax, color="yellow")


# Here we will replicate the zonal statistics like in section 4, but will iterate the command over each image band.
# 
# First we need the vector layer name:

# Get list of layer names in vector file
vec_lyrs = rsgislib.vectorutils.get_vec_lyrs_lst(in_field_vec)

# Get first name from vec_lyrs list
lyr_name = vec_lyrs[0]
print(f"layer name: {lyr_name}")


# We will now need to create a list to iterate over each band of the image and assign a unique field name for each image band:

# Create an array of image band numbers using the module numpy
image_bands = numpy.arange(1, 13, 1)
print(f"image bands: {image_bands}")

# Create a list of attribute fields we will populate
out_fields_list = [
    "jan_mean",
    "feb_mean",
    "mar_mean",
    "apr_mean",
    "may_mean",
    "jun_mean",
    "jul_mean",
    "aug_mean",
    "sep_mean",
    "oct_mean",
    "nov_mean",
    "dec_mean",
]


# As we are reading 12 bands into the layer, it would be inefficient to do this in the same manner as above as we will have to read, populate and write the vector each time. Instead, it is more efficient to read the layer into memory once, populate the values and write out the data once.
# 
# We will start by reading the layer into memory:

# Read vector layer to memory
vec_ds_obj, vec_lyr_obj = rsgislib.vectorutils.read_vec_lyr_to_mem(
    in_field_vec, lyr_name
)


# We can now iterate over these lists, running the command a total of 12 times:

# set the command constants
# memory object
mem_obj = vec_lyr_obj

# set up a for loop to iterate over the image bands list and execute the command
for band in image_bands:
    # specify the image band
    img_band_num = int(band)
    print(f"band number = {img_band_num}")
    # specify the out field (out fields list indexed to band, minus 1)
    mean_field_name = out_fields_list[band - 1]
    print(f"out field name = {mean_field_name}")
    # Run the command
    rsgislib.zonalstats.calc_zonal_band_stats_test_poly_pts(
        mem_obj,
        input_img,
        img_band=img_band_num,
        min_thres=1,
        max_thres=10000,
        out_no_data_val=0,
        min_field=None,
        max_field=None,
        mean_field=mean_field_name,
    )


# These values are held in the memory object so should be written back to a new file to save them without overwriting the original file:

# Memory object
mem_obj_vec = mem_obj
# output vector file path - changed to not overwrite the input file
field_stats_vec = "tile_segs_mskd_lbl_vec111_stats.gpkg"
# out layer name (same as input)
layer_name = lyr_name
# Vector format
vec_format = "GPKG"

rsgislib.vectorutils.write_vec_lyr_to_file(
    mem_obj_vec,
    field_stats_vec,
    layer_name,
    vec_format,
    options=["OVERWRITE=YES", "SPATIAL_INDEX=YES"],
)


# Now we can look at the attribute table that we have populated:

# Open the gpkg file with geopandas
gdf = geopandas.read_file(field_stats_vec)

# View the first 5 attributes
gdf.head()


# ### Time-series analysis

# For this we will look at one specific field, by selecting a specific row (field) from the geopandas dataframe. Here we will use a combination of geopandas, numpy and matplotlib (within geopandas) to show how rsgisliob can be combined with other python modules for you to  complete your analysis

# read row (field) number 207 and remove the FID and geometry info which are not needed
field = gdf.iloc[207].drop(["PXLVAL", "geometry"])
print(field)


# Which is visualized here:

# The fields vector file was read in above to look at the attribute table

# Create a plot using the matplotlib library
fig, ax = plt.subplots(figsize=(10, 10))
# Add the image to the axis
rsgislib.tools.mapping.create_raster_img_map(ax, input_img, img_bands=[1, 2, 3], img_stch=rsgislib.IMG_STRETCH_CUMULATIVE)

# Plot the fields over the image data
field_polys.plot(ax=ax, color="yellow")
# plot the field location geometry
gdf.loc[[207], "geometry"].plot(ax=ax, color="red")


# Make a list of the 12 months of data from the dataframe

# Convert the dataframe to a list, ignoring the fid [0] and geometry [-1]
field_vals = field.to_numpy().astype(float)
field_vals


# The values we currently have are in linear power and multiplied by a scale factor of 10000. We should convert these values to decibel for plotting and visualization.
# 
# Firstly we will remove the scale factor:

# divide by 10000
field_vals_pwr = field_vals / 10000
field_vals_pwr


# We can now convert these values to log decibel (dB) using the standard formula:

# $10*log10(x^2)$

field_vals_dB = 10 * numpy.log(field_vals_pwr ** 2)
field_vals_dB


# We can plot this data like this:

plt.scatter(out_fields_list, field_vals_dB)
plt.xlabel("Date")
plt.ylabel("Backkscatter (dB)")
plt.xticks(rotation=90)
plt.show()


# Finally, we can use the scipy module to fit a curve to our data points so we can better visualize the trend:

from scipy.optimize import curve_fit


# Fourier Series 1 Term (scaled X) from zunzun.com
def func(x, offset, a1, b1, c1):
    return a1 * numpy.sin(c1 * x) + b1 * numpy.cos(c1 * x) + offset


# these are the same as the scipy defaults
init_parameters = numpy.array([1.0, 1.0, 1.0, 1.0])

x_axis = numpy.arange(1, 13, 1)
x_model = numpy.arange(1, 13, 0.2)

fitted_parameters, pcov = curve_fit(func, x_axis, field_vals_dB, init_parameters)

model_predictions = func(x_model, *fitted_parameters)
mdl_pred = func(x_axis, *fitted_parameters)

plt.scatter(out_fields_list, field_vals_dB)
plt.plot(x_model, model_predictions, c="black")
plt.xlabel("Date")
plt.ylabel("Backscatter (dB)")
plt.xticks(rotation=90)
plt.show()


# Here we can see that there are three distinct peaks associated with cropping cyles. The Mekong Delta is one of the only places on Earth where three rice cropping cycles are possible.
