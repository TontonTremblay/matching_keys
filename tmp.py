import torch
import cv2 
import numpy as np

# Load your data
pos = torch.load('renders/pos.pt')
pos = torch.cat([pos, torch.ones([pos.shape[0],pos.shape[1], 1]).cuda()], axis=2)[0]
pos_idx = torch.load('renders/pos_idx.pt')[0]  # Triangle indices for the entire mesh
cam_mtx = torch.load('renders/cam_mtx.pt')
cam_proj = torch.load('renders/cam_proj.pt')[0]

rast_out = torch.load(f'renders/rast_out.pt')



# Resolution
image_width, image_height = 400, 400
image_ = torch.tensor([image_width, image_height]).cuda().float()

from icecream import ic 
print = ic

for i_cam, camera_matrix in enumerate(cam_mtx):
    
    visible_triangles = rast_out[i_cam,:,:,-1]
    visible_triangles = visible_triangles[visible_triangles>0] + 1
    visible_triangles = torch.unique(visible_triangles)
    # Apply the camera transformation
    transformed_vertices = torch.matmul(pos, torch.inverse(camera_matrix))

    # Apply the projection matrix
    projected_vertices = torch.matmul(transformed_vertices, cam_proj)
    # print(projected_vertices.min(),projected_vertices.max())
    # print(projected_vertices.shape)
    projected_vertices = projected_vertices[...,:2]/projected_vertices[...,-1].unsqueeze(-1)
    normalized_coordinates = ((projected_vertices + 1) / 2) * image_
    image = np.zeros((image_height, image_width, 3), dtype=np.uint8)

    # for triangle_index in visible_triangles:
    #     triangle_indices = pos_idx[triangle_index.long()]

    #     for i in range(3):
    #         start_vertex = projected_vertices[triangle_indices[i]].cpu().numpy()
    #         end_vertex = projected_vertices[triangle_indices[(i + 1) % 3]].cpu().numpy()
    #         start_point = (int((start_vertex[0] + 1) * (image_width - 1) / 2), int((start_vertex[1] + 1) * (image_height - 1) / 2))
    #         end_point = (int((end_vertex[0] + 1) * (image_width - 1) / 2), int((end_vertex[1] + 1) * (image_height - 1) / 2))
    #         cv2.line(image, start_point, end_point, (255, 255, 255), 1)

    for vertex_index in range(len(pos)):
        if vertex_index in visible_triangles:
            vertex_coords = projected_vertices[vertex_index].cpu().numpy()
            vertex_point = (int((vertex_coords[0] + 1) * (image_width - 1) / 2), int((vertex_coords[1] + 1) * (image_height - 1) / 2))
            if vertex_point[0]>0 and vertex_point[0]<image_width and vertex_point[1]>0 and vertex_point[1]<image_height:
                cv2.circle(image, vertex_point, 2, (0, 255, 0), -1)  # Draw green circles for visible vertices

    image = cv2.flip(image,0)
    cv2.imwrite('tmp.png',image)

    image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    image = cv2.imread(f'renders/{str(i_cam).zfill(5)}.png')
    # image = cv2.flip(image,0)

    for vertex_data in normalized_coordinates:
        # print(vertex_data)
        vertex_x, vertex_y = vertex_data.cpu().numpy().astype(int)
        vertex_point = (vertex_x, vertex_y)
        if vertex_point[0]>0 and vertex_point[0]<image_width and vertex_point[1]>0 and vertex_point[1]<image_height:
            cv2.circle(image, (vertex_x, vertex_y), radius=1, color=(0,255,0), thickness=-1)

    image = cv2.flip(image,0)
    cv2.imwrite('tmp2.png',image)


    break