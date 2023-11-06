import nvdiffrast.torch as dr
import argparse
import random 
import kornia 
import glob
import json
import pyrr 
import time
import trimesh 
import numpy as np 
import torch
import os 
import cv2
from icecream import ic
print = ic 


rast_out = torch.load(f'renders/rast_out.pt')
pos = torch.load(f'renders/pos.pt')
pos = torch.cat([pos, torch.ones([pos.shape[0],pos.shape[1], 1]).cuda()], axis=2)
pos_idx = torch.load(f'renders/pos_idx.pt')
cam_mtx = torch.load(f'renders/cam_mtx.pt')
cam_proj = torch.load(f'renders/cam_mtx.pt')

# pick 2 views
p1,p2 = random.sample(list(range(rast_out.shape[0])), 2)


# Resolution
image_width, image_height = 1000, 1000

image_ = torch.tensor([image_width, image_height]).cuda()

visible_vertices = []
for camera_matrix in cam_mtx:
    # Apply the camera transformation
    transformed_vertices = torch.matmul(pos[0], camera_matrix)

    # Apply the projection matrix
    projected_vertices = torch.matmul(transformed_vertices, cam_proj[0])

    print(projected_vertices.shape)

    # Clip and normalize
    visible_vertices_mask = (
        (projected_vertices[:, 0] >= -1)
        & (projected_vertices[:, 0] <= 1)
        & (projected_vertices[:, 1] >= -1)
        & (projected_vertices[:, 1] <= 1)
    )

    # Map to image space
    normalized_coordinates = ((projected_vertices[:, :2] + 1) / 2) * image_

    # Determine visibility
    visible_indices = torch.where(visible_vertices_mask)[0]

    for index in visible_indices:
        visible_vertices.append({
            'vertex_index': index,
            '2D_coordinates': normalized_coordinates[index].cpu().numpy()
        })

    image = np.zeros((image_height, image_width, 3), dtype=np.uint8)

    # Define the color of the green points (in BGR format)
    green_color = (0, 255, 0)
    white_color = (255, 255, 255)

    # Iterate through the visible vertices and draw green points
    
    for triangle_indices in pos_idx[0]:

        triangle_vertices = [visible_vertices[i]['2D_coordinates'] for i in triangle_indices]

        for i in range(3):
            start_point = tuple(triangle_vertices[i].astype(int))
            end_point = tuple(triangle_vertices[(i + 1) % 3].astype(int))
            cv2.line(image, start_point, end_point, white_color, 1)

    for vertex_data in visible_vertices:
        vertex_x, vertex_y = vertex_data['2D_coordinates'].astype(int)
        cv2.circle(image, (vertex_x, vertex_y), radius=1, color=green_color, thickness=-1)
    cv2.imwrite('tmp.png',image)
    break