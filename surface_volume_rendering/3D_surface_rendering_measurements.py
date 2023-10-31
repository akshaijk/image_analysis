import pyclesperanto_prototype as cle
import vedo
import napari
import napari_process_points_and_surfaces as nppas
from skimage import io

binary_image = io.imread(r'C:\Users\Akshai JK\Desktop\Image_Analysis\Segmenting_Lateraline_Final\Binary_Mask_Not2.tif')

binary_image

surface = nppas.all_labels_to_surface(binary_image)

viewer = napari.Viewer(ndisplay=3)

viewer.add_surface(surface)


smoothed_surface = nppas.smooth_surface(surface)

smoothed_surface4 = nppas.smooth_surface(surface, pass_band=0.001)

simplified_surface = nppas.decimate_pro(smoothed_surface, fraction=0.5)

simplified_surface2 = nppas.decimate_quadric(smoothed_surface, fraction=0.1)



point_cloud = nppas.sample_points_from_surface(surface, distance_fraction=0.01)


points_layer = viewer.add_points(point_cloud, size=1)




# we need to smooth the surface before quantitative measurements. 
viewer.add_surface(smoothed_surface)

viewer.add_surface(smoothed_surface4)

viewer.add_surface(simplified_surface2)

surface2 = nppas.add_quality(surface, nppas.Quality.SPHERE_FITTED_CURVATURE_HECTA_VOXEL)
surface2.azimuth = -90
surface2

viewer.add_surface(surface2, colormap=surface2.cmap)

# save the surface using .ply format
mesh = nppas.to_vedo_mesh(surface4)
filename = r"C:\Users\Akshai JK\Desktop\Image_Analysis\Segmenting_Lateraline_Final\lateraline_surface.ply"
_ = vedo.write(mesh, lateraline_surface.ply)

import vedo

mesh = nppas.to_vedo_mesh(smoothed_surface4)
filename = r"C:\Users\Akshai JK\Desktop\Image_Analysis\Segmenting_Lateraline_Final\lateraline_surface.ply"

vedo.write(mesh, filename)  # Corrected the filename variable

#quantification of surfaces
requested_measurements = [nppas.Quality.AREA, 
                          nppas.Quality.ASPECT_RATIO,
                          nppas.Quality.GAUSS_CURVATURE, 
                          nppas.Quality.MEAN_CURVATURE,
                          nppas.Quality.SPHERE_FITTED_CURVATURE_DECA_VOXEL,
                          nppas.Quality.SPHERE_FITTED_CURVATURE_HECTA_VOXEL,
                          nppas.Quality.SPHERE_FITTED_CURVATURE_KILO_VOXEL,
                         ]

df = nppas.surface_quality_table(surface2, requested_measurements)


df


quantified_surface = nppas.set_vertex_values(surface2, curvature)
quantified_surface

nppas.show(quantified_surface, azimuth=-90, cmap='hsv')
