from matplotlib import pyplot as plt
import pyclesperanto_prototype as cle
from skimage import io

img = io.imread(r'C:/Users/Akshai JK/Desktop/Image_Analysis/Nuclei.tif')

plt.imshow(img, cmap='gray')


print("Available OpenCL devices:" + str(cle.available_device_names()))

# insert the name of GPU from previous code here
device = cle.select_device('Intel(R) HD Graphics 5500')
print("Used GPU: ", device)


# Push the image to gpu memory
img_gpu = cle.push(img)
cle.imshow(img_gpu, color_map='gray')


# Step 1: heavy gaussian blur the image (e.g., sigma=12) and detect maxima for each nuclei
# heavy gaussian blurring assists in detecting maxima that reflects the objects.
# If objects are closer, you may want to decrease the amount of blurring.
img_gaussian = cle.gaussian_blur(img, sigma_x=9, sigma_y=9, sigma_z=12)
plt.imshow(img_gaussian, cmap='gray')

# Find out the maxima locations for each 'blob'
img_maxima_locations = cle.detect_maxima_box(
    img_gaussian, radius_x=0, radius_y=0, radius_z=0)

# Number of maxima locations (= number of objects)
# This number depend on the amount of Gaussian blur
number_of_maxima_locations = cle.sum_of_all_pixels(img_maxima_locations)
print("number of detected maxima locations", number_of_maxima_locations)

# View the blurred image and corresponding maxima locations
fig, axs = plt.subplots(1, 2, figsize=(15, 15))
cle.imshow(img_gaussian, plot=axs[0], color_map='gray')
cle.imshow(img_maxima_locations, plot=axs[1], color_map='gray')







# Step 2: threshold the input image after applying light gaussian blur (sigma=1)
img_gaussian2 = cle.gaussian_blur(img, sigma_x=1, sigma_y=1, sigma_z=1)
img_thresh = cle.threshold_otsu(img_gaussian2)

fig, axs = plt.subplots(1, 2, figsize=(15, 15))
cle.imshow(img_gaussian2, plot=axs[0], color_map='gray')
cle.imshow(img_thresh, plot=axs[1], color_map='gray')

#Step 3: Exclude maxima locations from the background, to make sure we only include the ones from nuclei
# We can do this by using binary and operation
img_relevant_maxima = cle.binary_and(img_thresh, img_maxima_locations)

number_of_relevant_maxima_locations = cle.sum_of_all_pixels(img_relevant_maxima)
print("number of relevant maxima locations", number_of_relevant_maxima_locations)

fig, axs = plt.subplots(1, 3, figsize=(15, 15))
cle.imshow(img_maxima_locations, plot=axs[0], color_map='gray')
cle.imshow(img_thresh, plot=axs[1], color_map='gray')
cle.imshow(img_relevant_maxima, plot=axs[2], color_map='gray')




#Step 4: Separate maxima locations into labels using masked voronoi
voronoi_separation = cle.masked_voronoi_labeling(img_relevant_maxima, img_thresh)

fig, axs = plt.subplots(1, 2, figsize=(15, 15))
cle.imshow(img, plot=axs[0], color_map='gray')
cle.imshow(voronoi_separation, labels=True, plot=axs[1])

cle.imshow(img_relevant_maxima, labels=True)
