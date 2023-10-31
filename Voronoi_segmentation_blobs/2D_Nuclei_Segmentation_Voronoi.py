import matplotlib.pyplot as plt
import numpy as np
from skimage import exposure, io, util, measure
from skimage import img_as_ubyte
from scipy.spatial import Voronoi, voronoi_plot_2d
from matplotlib import pyplot as plt
from skimage import io, filters


img = io.imread(r'C:/Users/Akshai JK/Desktop/Image_Analysis/Nuclei.tif')

plt.imshow(img, cmap='gray')

#Step 1 - Gaussian blur to averagelocal intensity variations
img_blurred = filters.gaussian(img, sigma=5) #sigma represents the amount of blurness
plt.imshow(img_blurred, cmap='gray')
plt.axis('off')

#Step 2: Find the points representing each object, to be used for Voronoi
from skimage.feature import peak_local_max
coordinates = peak_local_max(img_blurred, min_distance=25, 
                             exclude_border=False)


plt.imshow(img_blurred, cmap='gray')
plt.plot(coordinates[:, 1], coordinates[:, 0], 'r.')  # Coordinates is a numpy object. [ : , 0 ] means (more or less) [ first_row:last_row , column_0 ]. If you have a 2-dimensional list/matrix/array, this notation will give you all values in column 0 (from all rows). r. means plot as red dots 
plt.axis('off')


#Step 3: Vronoi regions
vor3 = Voronoi(coordinates)

fig3 = voronoi_plot_2d(vor3)
plt.axis('off')
plt.show()

vor = Voronoi(points)

vor_regions = vor.regions
print(vor_regions)