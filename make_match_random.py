import torch
import cv2 
import numpy as np
import random
import colorsys

from icecream import ic 
print = ic

def hsv_to_rgb(hsv_color):
    # Convert HSV color to RGB
    rgb_color = colorsys.hsv_to_rgb(hsv_color[0] / 179.0, hsv_color[1] / 255.0, hsv_color[2] / 255.0)
    
    # Scale the RGB values to the range 0-255 and round to integers
    rgb_color = tuple(int(val * 255) for val in rgb_color)
    
    return rgb_color

def random_hsv_color():
    # Generate random Hue (0-179, as OpenCV uses values in this range)
    hue = random.randint(0, 179)
    
    # Generate random Saturation (0-255)
    saturation = random.randint(0, 255)
    
    # Generate random Value (0-255)
    value = random.randint(0, 255)
    
    # Return the random HSV color as a tuple
    return hsv_to_rgb([hue, saturation, value])



def get_projected_and_unique_vertices(mtx,cam_proj,pos,pos_idx,list_triangles):
    # Apply the camera transformation

    final_mtx_proj = torch.matmul(cam_proj,mtx)

    projected_vertices = torch.matmul(pos, final_mtx_proj.transpose(0,1))
    projected_vertices = projected_vertices[...,:2]/projected_vertices[...,3].unsqueeze(-1)


    # check the list of triangles 

    list_triangles = list_triangles[list_triangles>0]-1
    list_triangles = list_triangles.unique().long()
    valid_vertex = pos_idx[list_triangles].flatten().unique()

    return projected_vertices,valid_vertex



torch.cuda.synchronize()
# Load your data
pos = torch.load('renders/pos.pt')
pos = torch.cat([pos, torch.ones([pos.shape[0],pos.shape[1], 1]).cuda()], axis=2)[0]
pos_idx = torch.load('renders/pos_idx.pt')[0]  # Triangle indices for the entire mesh
cam_mtx = torch.load('renders/cam_mtx.pt')
cam_proj = torch.load('renders/cam_proj.pt')[0]

rast_out = torch.load(f'renders/rast_out.pt')

# percentage display 
percentage_to_display = 0.1

# Resolution
image_width, image_height = 400, 400
image_ = np.array([image_width, image_height])

# select 2 random poses
p1,p2 = random.sample(list(range(rast_out.shape[0])), 2)


proj1,valid1 = get_projected_and_unique_vertices(cam_mtx[p1], cam_proj, pos, pos_idx, rast_out[p1,:,:,-1])
proj2,valid2 = get_projected_and_unique_vertices(cam_mtx[p2], cam_proj, pos, pos_idx, rast_out[p2,:,:,-1])

proj1 = proj1.cpu().numpy()
valid1 = valid1.cpu().numpy()
proj2 = proj2.cpu().numpy()
valid2 = valid2.cpu().numpy()
print(valid1)
print(valid2)
union_result = np.intersect1d(valid1, valid2)

sample_size = int(percentage_to_display * len(union_result))
union_result = np.random.choice(union_result, sample_size, replace=False)


image1 = cv2.imread(f'renders/{str(p1).zfill(5)}.png')
image1 = cv2.flip(image1, 0)

image2 = cv2.imread(f'renders/{str(p2).zfill(5)}.png')
image2 = cv2.flip(image2, 0)

image = cv2.hconcat([image1, image2])



for i_vertex in union_result:
    vertex_point1 = proj1[i_vertex]
    vertex_point1 = ((vertex_point1 + 1)*.5) * image_
    vertex_point1 = (int(vertex_point1[0]),int(vertex_point1[1]))
    
    vertex_point2 = proj2[i_vertex]
    vertex_point2 = ((vertex_point2 + 1)*.5) * image_
    vertex_point2 = (int(vertex_point2[0])+image_width,int(vertex_point2[1]))

    cv2.line(image, vertex_point1, vertex_point2, random_hsv_color(), thickness=1)

    if vertex_point1[0]>0 and vertex_point1[0]<image_width and vertex_point1[1]>0 and vertex_point1[1]<image_height:
        cv2.circle(image, vertex_point1, radius=1, color=(0,255,0), thickness=-1)

    
    if vertex_point2[0]>image_width and vertex_point2[0]<image_width*2 and vertex_point2[1]>0 and vertex_point2[1]<image_height:
        cv2.circle(image, vertex_point2, radius=1, color=(0,255,0), thickness=-1)



# image = cv2.flip(image,0)
image = cv2.flip(image, 0)
cv2.imwrite('match.png', image)


