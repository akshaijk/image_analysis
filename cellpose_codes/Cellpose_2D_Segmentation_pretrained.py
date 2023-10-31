# this code could be run locally
# works for 2D images 
# uses the pretrained cellpose model.
# For opening the created masks, use the code named 'opening cellpose masks'.
# the code works on colab/cluster as well.


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

img = io.imread(r'C:/Users/Akshai JK/Desktop/Image_Analysis/Membrane_analysis.tif')

# GPU
device = cle.select_device('Intel(R) HD Graphics 5500')
print("Used GPU: ", device)

# Push the image to gpu memory
DAPI_gpu = cle.push(img)
cle.imshow(DAPI_gpu, color_map='gray')


# the cellpose codes are taken from here https://cellpose.readthedocs.io/en/latest/settings.html
# gpu (bool (optional, default False)) – whether or not to use GPU, will check if GPU available
# model_type (str (optional, default 'cyto')) – ‘cyto’=cytoplasm model; ‘nuclei’=nucleus model; ‘cyto2’=cytoplasm model with additional user images


model = models.Cellpose(gpu=False, model_type='cyto')


# for batch processing the code can be as follows
# files = ['img0.tif', 'img1.tif']
# imgs = [imread(f) for f in files]
# masks, flows, styles, diams = model.eval(imgs, diameter=None, channels=[0,0],
#                                         flow_threshold=0.4, do_3D=False)



# CHANNELS - 0=grayscale, 1=red, 2=green, 3=blue
# For automated estimation set diameter = None. However, if this estimate is incorrect please set the diameter by hand.


channels = [0,0]  # This means we are processing single-channel greyscale images.


#this is the main function in cellpose
masks, flows, styles, diams = model.eval(img, diameter=None, channels=[0,0],
                                         flow_threshold=0.4, do_3D=False)



fig = plt.figure(figsize=(12,5))
plot.show_segmentation(fig, img, masks, flows[0], channels=channels)
plt.tight_layout()
plt.show()



# for opening the individual masks and work with them, use the code 'Cellpose open scripts'
