# This code takes 3D images and plots the given image in xy, yz and zx after correcting for anisotropy.  it could be 
# before runnning cellpose in 3D, try different values of diameter and flow_threshold in XY,YZ and ZX images. 
# Use those values of diameter and threshold that is able to segment xy, yz and zx images equally efficiently. 
# If segmentation is working in XY, but not in yz and zy due to anisotropy, use cellpose by stiching. 



from cellpose import models
from cellpose.io import imread
from skimage import io
from matplotlib import pyplot as plt
import numpy as np
import pyclesperanto_prototype as cle
from skimage import exposure, img_as_ubyte
from skimage import io
from skimage import measure
import pandas as pd
from cellpose import plot, utils
from skimage.transform import rescale


img_xyz = io.imread(r'C:/Users/Akshai JK/Desktop/Image_Analysis/lat_line_3D_10.tiff')

print(img_xyz.shape)

viewer = napari.view_image(img_xyz)

img_interpolated = io.imread(r'C:/Users/Akshai JK/Desktop/Image_Analysis/Membrane_interpolated_image.tif')

# set the original vozel size. Get this infor from imagej > properties. 

voxelsize_x = 0.10
voxelsize_y = 0.10
voxelsize_z = 0.22

#upscaling - can increase the size
img_xyz_rescaled = rescale(img_xyz, [2.5, 1, 1])

print("Available OpenCL devices:" + str(cle.available_device_names()))

# insert the name of GPU from previous code here
device = cle.select_device('Intel(R) HD Graphics 5500')
print("Used GPU: ", device)

# Push the image to gpu memory
img_gpu = cle.push(img_xyz)
cle.imshow(DAPI_gpu, color_map='gray')

# downscaling - reduces file size, but loss of information. 
img_downscaled = cle.scale(img_xyz, factor_x = voxelsize_x, factor_y = voxelsize_y, factor_z = voxelsize_z, auto_size = True)


# Visual Inspection in Xy, XZ and YZ

# ZX plots
img_xz =  img_xyz[:,:,250]
img_xz_interpolated = img_interpolated[:,:,250]

# xy plots
img_xy_interpolated = img_interpolated[90,:,:]
plt.imshow(img_xy_interpolated, cmap='Greys_r')

# YZ plots
plt.imshow(img_XYZ[:,:,250], cmap='Greys_r')
plt.imshow(img_XYZ_rescaled[:,:,250], cmap='Greys_r')


# Generating new 2D images from img_XYZ in XZ and YZ directions for cellpose test

cle.imshow(img_XY, color_map='gray')


#GPU from previous code
device = cle.select_device('Intel(R) HD Graphics 5500')
print("Used GPU: ", device)

# Push the image to gpu memory
DAPI_gpu = cle.push(img)
cle.imshow(DAPI_gpu, color_map='gray')


# the cellpose codes are taken from here https://cellpose.readthedocs.io/en/latest/settings.html
# gpu (bool (optional, default False)) – whether or not to use GPU, will check if GPU available
# model_type (str (optional, default 'cyto')) – ‘cyto’=cytoplasm model; ‘nuclei’=nucleus model; ‘cyto2’=cytoplasm model with additional user images



model = models.Cellpose(gpu=True, model_type='cyto')


# for batch processing the code can be as follows
# files = ['img0.tif', 'img1.tif']
# imgs = [imread(f) for f in files]
# masks, flows, styles, diams = model.eval(imgs, diameter=None, channels=[0,0],
#                                         flow_threshold=0.4, do_3D=False)



# CHANNELS - 0=grayscale, 1=red, 2=green, 3=blue
# For automated estimation set diameter = None. However, if this estimate is incorrect please set the diameter by hand.


channels = [0,0]  # This means we are processing single-channel greyscale images.


#this is the main function in cellpose (default fow_threshold and diameter)
# masks, flows, styles, diams = model.eval(img_xz, diameter=None, channels=[0,0],
 #                                       flow_threshold=0.4, do_3D=False)


# in the XZ direction, a flow threshold of 0.8 and diameter of 60 is working well. 
masks, flows, styles, diams = model.eval(img_xz_interpolated, diameter=60, channels=[0,0],
                                         flow_threshold=0.8, do_3D=False)


fig = plt.figure(figsize=(12,5))
plot.show_segmentation(fig, img_xz_interpolated, masks, flows[0], channels=channels)
plt.tight_layout()
plt.show()


# try different valiues of threshold and diameter in xy, yz and zx.







