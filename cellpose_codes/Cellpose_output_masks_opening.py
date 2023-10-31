# the codes help in locally opening masks generated in cellpose using napari
import napari
import numpy

# loads the _seg.npy file from Cellpose Colab notebook
bi_dat = np.load(r'C:/Users/Akshai JK/Desktop/Image_Analysis/Cellpose_Test_binary_gaussian/Cellpose_Test/binary_segments_seg_seg.npy', allow_pickle=True).item()

gu_dat = np.load(r'C:/Users/Akshai JK/Desktop/Image_Analysis/Cellpose_Test_binary_gaussian/Cellpose_Test/gaussian_segments_seg_seg.npy', allow_pickle=True).item()


# plt.imshow(dat['masks'])

mask_bi = bi_dat['masks']

mask_gu = gu_dat['masks']

# img = io.imread(r'C:/Users/Akshai JK/Desktop/Image_Analysis/Nuclei_3D_rescaled2.tif')

viewer = napari.view_image(mask_bi)

viewer = napari.view_image(mask_gu)


#viewer = napari.view_image(img)


# Makes an overlay of the segmented masks with the original image. 
# mask_RGB = plot.mask_overlay(dat['img'], dat['masks'])
# plt.imshow(mask_RGB)

# for o in outlines:
#    plt.plot(o[:,0], o[:,1], color='r')