
from matplotlib import pyplot as plt
import numpy as np
import pyclesperanto_prototype as cle
from skimage import exposure, img_as_ubyte
from skimage import io
from skimage import measure
import pandas as pd

img = io.imread(r'C:/Users/Akshai JK/Desktop/Image_Analysis/Nuclei.tif')

plt.imshow(img, cmap='gray')

print(img.shape)

DAPI_8bit = img_as_ubyte(exposure.rescale_intensity(img)) # Convert the image into 8 bit

plt.imshow(DAPI_8bit, cmap='gray')

print("Available OpenCL devices:" + str(cle.available_device_names()))

# insert the name of GPU from previous code here
device = cle.select_device('Intel(R) HD Graphics 5500')
print("Used GPU: ", device)

# Push the image to gpu memory
DAPI_gpu = cle.push(img)
cle.imshow(DAPI_gpu, color_map='gray')

############ voronoi_otsu_labeling library ##################
# voronoi_otsu_labeling(image, spot_sigma=some_number, outline_sigma=another_number)
#spot_sigma= depends on how close the detected objects can be. 
#Low number may divide large objects into multiple objects.
#outline_sigma = how precise the outline needs to be for the segmented objects (use a low number)
segmented = cle.voronoi_otsu_labeling(DAPI_gpu, spot_sigma=5, 
                                      outline_sigma=0.5)
cle.imshow(segmented, labels=True)

#Remove edge touching objects
segmented_excl_edges = cle.exclude_labels_on_edges(segmented)
cle.imshow(segmented_excl_edges, labels=True)

# Number of objects segmented?
#This will be the maximum label assigned to an object 9as each object is assigned unique label value)
num_objects = cle.maximum_of_all_pixels(segmented_excl_edges)
print("Total objects detected are: ", num_objects)

#Save segmented image to disk
# save image to disk
from skimage.io import imsave
segmented_array = cle.pull(segmented_excl_edges)
#This is a uint32 labeled image with each object given an integer value.
plt.imshow(segmented_array)  

imsave("result.tif", segmented_array)  #Open in imageJ for better visualization

###############################################################


#Pixel count map - map by object size
pixel_count_map = cle.label_pixel_count_map(segmented_excl_edges)
cle.imshow(pixel_count_map, colorbar=True, color_map='jet')


fig, axs = plt.subplots(1, 1)
cle.imshow(segmented_excl_edges, continue_drawing=True, plot=axs)
cle.imshow(cle.reduce_labels_to_label_edges(segmented_excl_edges), labels=True, plot=axs, continue_drawing=True, alpha=0.7)


# Analyse objects
cle.imshow(segmented_excl_edges, labels=True)

props = measure.regionprops_table(segmented_excl_edges)

props_trial = measure.regionprops_table(segmented_excl_edges, properties=('centroid',
                                                                          'orientation',
                                                                          'axis_major_length',
                                                                          'axis_minor_length'))




props = measure.regionprops_table(segmented_excl_edges, properties=('label', 'centroid', 'area','bbox', 'area_bbox', 'area_convex', 'convex_image', 'image', 'inertia_tensor', 'inertia_tensor_eigvals', 'local_centroid', 'major_axis_length', 'minor_axis_length', 'moments', 'moments_central', 'moments_normalized', 'slice', 'orientation', 'axis_major_length', 'axis_minor_length'))

pd.DataFrame(props)

# Intensity measurements not working yet. 