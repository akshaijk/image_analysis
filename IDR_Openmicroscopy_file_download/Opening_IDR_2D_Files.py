# works for 2D files, but not for 3D or time lapse files. USe the colaab script for downloading 3D and timelapse


from tifffile import imsave, imwrite
from skimage.io import imread, imshow
import numpy



original_link =' https://idr.openmicroscopy.org/webclient/render_image_download/9837024/?format=tif'


edited_link = 'https://idr.openmicroscopy.org/webclient/render_image/9837024/'



image = imread(edited_link)

# get the dimension of the image 

image.shape

imshow(image)

img_xz =  image[:,:,2]

imshow(img_xz)

imshow(image[:,:,3], cmap='gray')



imwrite("C:/Users/Akshai JK/Desktop/Image_Analysis/lateral_line_3D.tif", image, imagej=True)
