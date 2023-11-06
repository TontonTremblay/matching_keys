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

parser = argparse.ArgumentParser(description='Rendering reference views')
parser.add_argument('--model',
    default="models/obj_000001.ply",
    help='path to 3d models')
parser.add_argument('--outf',
    default='renders/',
    help='where to place the camera_pose translation and quaternion in opengl format')
parser.add_argument('--res',
    type=int,
    default=400)
parser.add_argument('--fx',
    type=int,
    default=640)
parser.add_argument('--zoom',
    type=float,
    default=4)
opt = parser.parse_args()

class CameraIntrinsicSettings(object):
    # DEFAULT_ZNEAR = 0.1
    # DEFAULT_ZFAR = 100000.0
    # DEFAULT_ZFAR = DEFAULT_ZNEAR
    
    def __init__(self,
            res_width = 640.0, res_height = 480.0,
            fx = 640.0, fy = 640.0,
            cx = 320.0, cy = 240.0,
            projection_matrix = None,
            near=0.01,far=5000):
        self.res_width = res_width
        self.res_height = res_height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

        self.znear = near
        self.zfar = far

        self.projection_matrix = projection_matrix
    def get_projection_matrix(self):
        if (self.projection_matrix is None):
            self._calc_calib_proj()
            #self.calculate_projection_matrix()

        return self.projection_matrix
    def _calc_calib_proj(self):
        """Conversion of Hartley-Zisserman intrinsic matrix to OpenGL proj. matrix.

        Ref:
        1) https://strawlab.org/2011/11/05/augmented-reality-with-OpenGL
        2) https://github.com/strawlab/opengl-hz/blob/master/src/calib_test_utils.py

        :param K: 3x3 ndarray with the intrinsic camera matrix.
        :param x0 The X coordinate of the camera image origin (typically 0).
        :param y0: The Y coordinate of the camera image origin (typically 0).
        :param w: Image width.
        :param h: Image height.
        :param nc: Near clipping plane.
        :param fc: Far clipping plane.
        :param window_coords: 'y_up' or 'y_down'.
        :return: 4x4 ndarray with the OpenGL projection matrix.
        """

        K = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ])
        x0 = 0
        y0 = 0
        w = self.res_width
        h = self.res_height
        nc = self.znear
        fc = self.zfar

        window_coords='y_down'

        depth = float(fc - nc)
        q = -(fc + nc) / depth
        qn = -2 * (fc * nc) / depth

        # Draw our images upside down, so that all the pixel-based coordinate
        # systems are the same.
        if window_coords == 'y_up':
            proj = np.array([
            [2 * K[0, 0] / w, -2 * K[0, 1] / w, (-2 * K[0, 2] + w + 2 * x0) / w, 0],
            [0, -2 * K[1, 1] / h, (-2 * K[1, 2] + h + 2 * y0) / h, 0],
            [0, 0, q, qn],  # Sets near and far planes (glPerspective).
            [0, 0, -1, 0]
            ])

        # Draw the images upright and modify the projection matrix so that OpenGL
        # will generate window coords that compensate for the flipped image coords.
        else:
            assert window_coords == 'y_down'
            proj = np.array([
            [2 * K[0, 0] / w, -2 * K[0, 1] / w, (-2 * K[0, 2] + w + 2 * x0) / w, 0],
            [0, 2 * K[1, 1] / h, (2 * K[1, 2] - h + 2 * y0) / h, 0],
            [0, 0, q, qn],  # Sets near and far planes (glPerspective).
            [0, 0, -1, 0]
            ])
        
        self.projection_matrix = proj
        #return proj.T

# ==============================================================================================
#  Helper functions
# ==============================================================================================
def get_image(img_batch,ii,depth=False):

    img = img_batch[ii,...]
    img = img.mul(255).clamp_(0, 255).to("cpu", torch.uint8).numpy()
    img = cv2.flip(img,0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if depth is True:
        img = cv2.applyColorMap((img).astype(np.uint8), cv2.COLORMAP_JET)

    return img

def load_mesh(mesh_path):

    # load the mesh
    mesh = trimesh.load(
            mesh_path,
            force='mesh'
        )

    pos = np.asarray(mesh.vertices)/100
    pos_idx = np.asarray(mesh.faces)

    normals = np.asarray(mesh.vertex_normals)
    
    # load vertex color
    # Create position/triangle index tensors
    pos_idx = torch.from_numpy(pos_idx.astype(np.int32)).cuda()
    vtx_pos = torch.from_numpy(pos.astype(np.float32)).cuda()
    vtx_normals = torch.from_numpy(normals.astype(np.float32)).cuda()



    bounding_volume = [
        [torch.min(vtx_pos[:,0]),torch.min(vtx_pos[:,1]),torch.min(vtx_pos[:,2])],
        [torch.max(vtx_pos[:,0]),torch.max(vtx_pos[:,1]),torch.max(vtx_pos[:,2])]
    ]

    dimensions = [
        bounding_volume[1][0] - bounding_volume[0][0], 
        bounding_volume[1][1] - bounding_volume[0][1], 
        bounding_volume[1][2] - bounding_volume[0][2]
    ]
    center_point = [
        ((bounding_volume[0][0] + bounding_volume[1][0])/2).item(), 
        ((bounding_volume[0][1] + bounding_volume[1][1])/2).item(), 
        ((bounding_volume[0][2] + bounding_volume[1][2])/2).item()
    ]



    if isinstance(mesh.visual, trimesh.visual.texture.TextureVisuals):
        tex = np.array(mesh.visual.material.image)
        uv = mesh.visual.uv
        uv[:,1] = 1 - uv[:,1]
        # uv[:,1] = uv[:,1]
        uv_idx = np.asarray(mesh.faces)

        tex = np.array(mesh.visual.material.image)/255.0

        tex     = torch.from_numpy(tex.astype(np.float32)).cuda()
        uv_idx  = torch.from_numpy(uv_idx.astype(np.int32)).cuda()
        vtx_uv  = torch.from_numpy(uv.astype(np.float32)).cuda()    


        result = {
            "pos_idx": pos_idx,
            "pos": vtx_pos,
            "tex": tex,
            "uv": vtx_uv,
            "uv_idx": uv_idx,
            'bounding_volume':bounding_volume,
            'dimensions':dimensions,
            'center_point':center_point,
            'vtx_normals':vtx_normals,
        }
    else:
        vertex_color = mesh.visual.vertex_colors[...,:3]/255.0
        # print(vertex_color)
        vertex_color     = torch.from_numpy(vertex_color.astype(np.float32)).cuda()

        result = {
            "pos_idx": pos_idx,
            "pos": vtx_pos,
            'vtx_color':vertex_color,
            'bounding_volume':bounding_volume,
            'dimensions':dimensions,
            'center_point':center_point,
            'vtx_normals':vtx_normals,
        }
    return result

def interpolate(attr, rast, attr_idx, rast_db=None):
    return dr.interpolate(attr.contiguous(), rast, attr_idx, rast_db=rast_db, diff_attrs=None if rast_db is None else 'all')


def render_texture_batch(glctx, proj_cam, mtx, pos, pos_idx, 
        resolution, 
        uv=None, uv_idx=None, tex=None, 
        enable_mip=True, max_mip_level=1,
        near =.1 ,far =10,
        return_rast_out= False,
        vtx_color=None,
        dome=None,
        normals=None
    ):

    if not type(resolution) == list:
        resolution = [resolution,resolution]
    posw = torch.cat([pos, torch.ones([pos.shape[0],pos.shape[1], 1]).cuda()], axis=2)
    mtx_transpose = torch.transpose(mtx, 1, 2)
    # proj_cam_transpose = torch.transpose(proj_cam, 1, 2)

    # pos_in_cam = torch.matmul(posw, mtx_transpose)
    # pos_clip    = torch.matmul(pos_in_cam,proj_cam_transpose)

    # print("proj_cam",proj_cam.shape,proj_cam[0,...])
    # print("mtx",mtx.shape,mtx[0,...])

    final_mtx_proj = torch.matmul(proj_cam,mtx)

 
    pos_clip_ja = torch.matmul(posw,final_mtx_proj.transpose(1,2))


    # print("final_mtx_proj",final_mtx_proj.shape,final_mtx_proj[0,...])
    # print("pos",pos.shape,pos[0,0,...])
    # print("pos_clip",pos_clip.shape,pos_clip[0,0,...])
    # print("pos_clip_ja",pos_clip_ja.shape,pos_clip_ja[0,0,...])
    # print("posw_clip",posw_clip.shape,posw_clip[0,0,...])

    # raise()
    rast_out, rast_out_db = dr.rasterize(glctx, pos_clip_ja, pos_idx[0], resolution=resolution)


    # compute the depth
    gb_pos, _ = interpolate(posw, rast_out, pos_idx[0],rast_db=rast_out_db)
    shape_keep = gb_pos.shape
    gb_pos = gb_pos.reshape(shape_keep[0],-1,shape_keep[-1])
    # depth2 = torch.matmul(gb_pos, mtx_transpose)
    # print(gb_pos[0,0])

    if vtx_color is None:
        texc, texd = dr.interpolate(
            # uv[None, ...], 
            uv, 
            rast_out, 
            uv_idx[0], 
            rast_db=rast_out_db, 
            diff_attrs='all'
        )
        color = dr.texture(
            # tex[None, ...], 
            tex, 
            texc, 
            texd, 
            filter_mode='linear', 
            # filter_mode='linear-mipmap-linear', 
            # max_mip_level=max_mip_level
        )

        color = color * torch.clamp(rast_out[..., -1:], 0, 1) # Mask out background.
        if return_rast_out: 
            return color, rast_out
        return color
    else:
        # print(vtx_color)
        color, _ = dr.interpolate(vtx_color, rast_out, pos_idx[0])

        # img = out.cpu().numpy()[0, ::-1, :, :] # Flip vertically.
        # color = np.clip(np.rint(img * 255), 0, 255).astype(np.uint8) # Quantize to np.uint8
        color = color * torch.clamp(rast_out[..., -1:], 0, 1) # Mask out background.
        if return_rast_out: 
            return color, rast_out
        return color

def sphere_renders(
        nb_planes,
        nb_circle,
        elevation_range = [0,180],
        tetha_range = [0,360]
    ):

    positions_to_render = []
    for i_plane in range(nb_planes):
        elevation = np.deg2rad(  elevation_range[0] + \
                                ((i_plane+1) * (elevation_range[1]-elevation_range[0])/(nb_planes+1)))
        for i_circle in range(nb_circle):
            azimuth = np.deg2rad(tetha_range[0]+((i_circle+1) * (tetha_range[1]-tetha_range[0])/(nb_circle+1)))
            eye_position = [
                np.sin(elevation)*np.cos(azimuth),
                np.sin(elevation)*np.sin(azimuth),
                np.cos(elevation),
            ]
            positions_to_render.append(eye_position)
    return positions_to_render


def calculate_fov(fx, sensor_width):
    fov = 2 * np.arctan(sensor_width / (2 * fx))
    return np.degrees(fov)




################################################################# MAIN

result = load_mesh(opt.model)


cam_proj_obj = CameraIntrinsicSettings(
    res_width = opt.res, 
    res_height = opt.res,
    fx = opt.fx, 
    fy = opt.fx,
    cx = opt.res/2, 
    cy = opt.res/2,
    far = 100
)

cam_proj = torch.tensor(cam_proj_obj.get_projection_matrix())




positions = sphere_renders(
      nb_planes = 7, 
      nb_circle = 7,
      elevation_range = [2,178],
      tetha_range = [0,359]
  )

# make the look at matrix 
trans = []

print()
print(opt.model)
print(result["dimensions"])
print()

array = [result["dimensions"][0].item(),result["dimensions"][1].item(),result["dimensions"][2].item()]
max_object_dimension = max(array)




camera_fov = calculate_fov(opt.fx, opt.res)
dx, dy, dz = array

# Unpack image resolution
image_width, image_height = opt.res,opt.res

# Find the maximum dimension of the object
max_dimension = max(dx, dy, dz)

# print(camera_fov)
camview = 2*np.tan(np.deg2rad(camera_fov))
distance = max_dimension/camview
scaling_factor =distance
# print(scaling_factor)
# raise()
for p in positions:
    t = pyrr.matrix44.create_look_at(
        eye = np.array(p)*scaling_factor*opt.zoom,
        target = np.array([0,0,0]),
        up = [0,0,1]
    ).T.tolist()
    trans.append(t)
cam_mtx = torch.tensor(trans).cuda().float()

cam_proj = cam_proj.cuda().float()
cam_proj = torch.stack([cam_proj] * cam_mtx.shape[0], dim=0)

for key in result:
    try:
        result[key] = torch.stack([result[key]] * cam_mtx.shape[0], dim=0)
    except:
        pass
glctx = dr.RasterizeGLContext()


##### 
# rendering the views
if 'vtx_color' in result:
    rgbs, rast_out = render_texture_batch(
                    glctx=glctx,
                    proj_cam=cam_proj, 
                    mtx = cam_mtx, 
                    pos = result["pos"], 
                    pos_idx = result["pos_idx"], 
                    vtx_color=result['vtx_color'],
                    resolution = opt.res,
                    return_rast_out = True
                )
else:
    rgbs, rast_out = render_texture_batch(
                    glctx=glctx,
                    proj_cam=cam_proj, 
                    mtx = cam_mtx, 
                    pos = result["pos"], 
                    pos_idx = result["pos_idx"], 
                    uv = result["uv"], 
                    uv_idx = result["uv_idx"],
                    tex= result['tex'], 
                    resolution = opt.res,
                    return_rast_out = True
                )   

# visualizing / saving 

os.makedirs(opt.outf,exist_ok=True)

for i in range(rgbs.shape[0]):
    im = get_image(rgbs, i)
    cv2.imwrite(f'{opt.outf}/{str(i).zfill(5)}.png',im)

torch.save(rast_out,f'{opt.outf}/rast_out.pt')
