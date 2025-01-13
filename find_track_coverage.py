#!/usr/bin/env python3

########################
# Written by L. Marconato, 2025
########################

import numpy as np
import os, sys
from osgeo import gdal
from scipy.ndimage import gaussian_filter
import cv2
from shapely.geometry import Polygon
import argparse

#######################

def calculate_angle(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm_product == 0:  # Évite les divisions par zéro
        return 0
    angle = np.arccos(np.clip(dot_product / norm_product, -1.0, 1.0))  # Limite pour éviter les erreurs numériques
    return np.degrees(angle)

#######################

# Read args

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', type=str, help='Path to the geocoded LOS file, default: ./CNES_CosENU_geo_8rlks.tiff')
parser.add_argument('--filter_width', type=float, default=5, help='Size of the filter to apply before contour detection')
parser.add_argument('--max_vertices', type=int, default=100, help='Maximum number of vertices wanted for polygon detection')
parser.add_argument('--filter_angle', type=float, default=None, help='If set, will filter out all vertices with angles larger than the value provided')
parser.add_argument('-o', '--outfile', type=str, default='./coverage.txt',  help='Path for saving output txt, default: ./coverage.txt')

args = parser.parse_args()

file = args.file
filter_width = args.filter_width
max_vertices = args.max_vertices
filter_angle = args.filter_angle
outfile = args.outfile

#######################

# Open file
ds = gdal.Open(file, gdal.GA_ReadOnly)
image = ds.GetRasterBand(1).ReadAsArray()
geotransform = ds.GetGeoTransform()

# Convert to 0/255 image
image_binary = np.where(image==0, 0, 255).astype(np.uint8)

# Filter
image_filtered = gaussian_filter(image_binary, filter_width)

# Reconvert to binary
image_binary = np.where(image_filtered<128, 0, 255).astype(np.uint8)

# Détection des contours
contours, _ = cv2.findContours(image_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Keep only the main contour
contours = [contour for contour in contours if len(contour)>15]

# Check we have only one left
if len(contours) == 0:
    sys.exit("No contour detected")
elif len(contours) > 1:
    sys.exit("Several contours detected")
else:
    num_vertices = 1e18
    smooth = 0.
    while num_vertices > max_vertices:

        smooth += 1e-4

        # Approximation du contour pour réduire les points redondants
        epsilon = smooth * cv2.arcLength(contours[0], True)
        approx = cv2.approxPolyDP(contours[0], epsilon, True)

        # Conversion en polygone Shapely
        polygon = Polygon(approx.squeeze())
        #print("Polygone détecté :", polygon)

        num_vertices = len(list(polygon.exterior.coords))-1

print(num_vertices, 'vertices detected')

############ (Filter by angle) ############

# Récupérer les coordonnées du polygone
coords = np.array(polygon.exterior.coords)  # Tableau Nx2 des coordonnées (x, y)

if filter_angle is not None: # Filtrer les sommets en fonction de l'angle
    filtered_coords = []
    for i in range(len(coords) - 1):  # Le dernier point est une copie du premier (polygone fermé)
        prev_point = coords[i - 1]  # Point précédent
        curr_point = coords[i]      # Point courant
        next_point = coords[(i + 1) % (len(coords) - 1)]  # Point suivant

        # Vecteurs
        v1 = prev_point - curr_point
        v2 = next_point - curr_point

        # Calcul de l'angle entre v1 et v2
        angle = calculate_angle(v1, v2)

        # Filtrage : conserver uniquement si l'angle < 135°
        if angle < filter_angle:
            filtered_coords.append(curr_point)

    if not (filtered_coords[-1] == filtered_coords[0]).all(): # Close the polygon if necessary
        filtered_coords.append(filtered_coords[0])

    new_coords = np.array(filtered_coords)

else:
    new_coords = np.copy(coords)

x = new_coords[:,0]
y = new_coords[:,1]

# Convert to geographical coordinates
geo_x = geotransform[0] + geotransform[1]*x
geo_y = geotransform[3] + geotransform[5]*y

# Save
np.savetxt(outfile, np.c_[geo_x, geo_y], fmt='%.6f')
