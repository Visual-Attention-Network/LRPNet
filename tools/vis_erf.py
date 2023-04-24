import torch 
import numpy as np
import math 
import voxelnet.modules.functional as VF
from voxelnet.models.lrpnet import LRPNet
import torch.nn.functional as F
import cv2
import trimesh
import os
import torchsparse
from torchsparse import SparseTensor

def read_data(data_file,rotate=True,full_scale = [4096, 4096, 4096]):
    data = torch.load(data_file)
    labels = data['labels'].cuda()
    vertices_ori = data['vertices'].cuda() 
    src_vertices = vertices_ori.clone()
    colors = data['colors'].cuda()   
    src_colors = (colors.clone()+1)*255/2
    # Affine linear transformation
    trans_m = np.eye(3)

    trans_m *= 50
    theta = np.random.rand() * 2 * math.pi
    if rotate:
        trans_m = np.matmul(trans_m, [[math.cos(theta), math.sin(theta), 0], [-math.sin(theta), math.cos(theta), 0], [0, 0, 1]])
    
    trans_m = trans_m.astype(np.float32)
    
    # vertices_ori = np.matmul(vertices_ori, trans_m)
    vertices_ori = torch.matmul(vertices_ori, torch.tensor(trans_m,device=vertices_ori.device))
    
    # Random placement in the receptive field
    vertices_min = torch.min(vertices_ori, dim=0)[0].cpu().numpy()
    vertices_max = torch.max(vertices_ori, dim=0)[0].cpu().numpy()
    offset = -vertices_min + np.clip(full_scale - vertices_max + vertices_min - 0.001, 0, None) * np.random.rand(3) \
        + np.clip(full_scale - vertices_max + vertices_min + 0.001, None, 0) * np.random.rand(3)
    
    vertices_ori += torch.tensor(offset,device=vertices_ori.device)

    # Voxelization
    coords_v = vertices_ori.int()

    # Remove duplicate items
    _, unique_idxs,unique_reidx = VF.unique(coords_v, dim=0, return_index=True,return_inverse=True)
    coords_v = coords_v[unique_idxs]
    colors_v = colors[unique_idxs]
    labels_v = labels[unique_idxs]

    # Put into containers
    coords_v_b = torch.cat([coords_v, torch.full(size=(coords_v.shape[0], 1),fill_value=0,device=coords_v.device,dtype=torch.int)], 1)

    return coords_v_b, colors_v, labels_v, unique_reidx, unique_idxs, src_vertices,src_colors

def save_pointcloud(vertices,colors,faces,save_file):
    trimesh.Trimesh(vertices,faces=faces,vertex_colors=colors).export(save_file)

def draw_heatmap(grad_map,vertices,position,name,src_colors,faces,ply_vertices):
    grad_map = np.log10(grad_map+1)
    grad_map /= 0.8
    
    grad_map = np.clip(grad_map,0.,1.)
    grad_map = np.sqrt(grad_map)

    heatmap = cv2.applyColorMap(np.uint8(255 * grad_map), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = heatmap[:,0]
    
    src_colors = src_colors.cpu().numpy()

    fuse_colors = grad_map[:,None]*heatmap+(1-grad_map[:,None])*src_colors
    fuse_colors = np.clip(fuse_colors,0,255).astype(np.uint8)
    
    vertices = ply_vertices
    colors = fuse_colors

    os.makedirs("heatmaps",exist_ok=True)

    save_pointcloud(vertices,colors,faces,f"heatmaps/heatmap_{name}.ply")

def forward_model(model,coords_v_b,colors_v,unique_reidx, unique_idxs, src_vertices,position):
    colors_v = torch.autograd.Variable(colors_v, requires_grad=True)
    # input
    indices = coords_v_b.int()
    x = SparseTensor(colors_v, indices)

    en0 = model.input_conv(x)

    en_tensors = [en0]
    for i in range(model.depth):
        x = model.downsamples[i](en_tensors[-1])
        x = model.encoders[i](x)
        if i<model.depth-1:
            en_tensors.append(x)

    for i in range(model.depth):
            x = model.upsamples[i](x)
            x = torchsparse.cat([x,en_tensors[model.depth-i-1]])
            x = model.decoders[i](x)
            
    feats = x.feats
    coords = x.coords[:,:3]
    stride = x.s[0]
    
    tposition = torch.tensor(position).float().cuda()
    distance = torch.norm(src_vertices-tposition[None],dim=-1)
    sorted,idx = torch.sort(distance)
    idx = idx[:1]
    coords_xyz = coords_v_b[unique_reidx[idx].unique()][:,:3]
    coords_xyz = coords_xyz//stride*stride
    f_idx = torch.nonzero((((coords[:,None,:] - coords_xyz[None,:,:]) == 0).sum(dim=-1) == 3).sum(dim=-1)>0,as_tuple=True)[0] 

    center_point = F.relu(feats[f_idx]).sum()
    grad = torch.autograd.grad(center_point, colors_v)[0]
    grad = F.relu(grad)
    grad = grad[unique_reidx]

    grad_map = grad.sum(dim=1).cpu().numpy()
    return grad_map
        

def build_lrp_model(use_trained = True):
    model = LRPNet(
        in_channels=3,
        out_channels=20,
        encoder_channels=[32,64,96,128,128],
        decoder_channels=[128,128,128,128,128],
    )
    state_dict = torch.load("work_dirs/scannet_largenet_f10_scale/checkpoints/ckpt_epoch_500.pt")['model']
    for key in list(state_dict.keys()):
        if key.startswith("module."):
            new_key = key.replace("module.", "")
            state_dict[new_key] = state_dict[key]
            del state_dict[key]

    if use_trained:
        model.load_state_dict(state_dict)
    model = model.cuda()
    return model 


def vis_heatmap(model,name,data_file,position,faces,vertices,position2=None):
    coords_v, colors_v, labels, unique_reidx,unique_idxs,src_vertices,src_colors = read_data(data_file,rotate=True) 
    grad_map = np.zeros((src_vertices.shape[0],))
    for i in range(8):
        coords_v, colors_v, labels, unique_reidx,unique_idxs,src_vertices,src_colors = read_data(data_file,rotate=True) 
        grad_map_once = forward_model(model,coords_v,colors_v,unique_reidx,unique_idxs,src_vertices,position)
        grad_map += grad_map_once
        print("once",i)
    if position2 is None:
        draw_heatmap(grad_map,src_vertices,position,name,src_colors,faces,vertices)
        return 
    for i in range(8):
        coords_v, colors_v, labels, unique_reidx,unique_idxs,src_vertices,src_colors = read_data(data_file,rotate=True) 
        grad_map_once = forward_model(model,coords_v,colors_v,unique_reidx,unique_idxs,src_vertices,position2)
        grad_map += grad_map_once
        print("once",i)
    draw_heatmap(grad_map,src_vertices,position,name,src_colors,faces,vertices)

scene_name = "0011_00"
mesh = trimesh.load(f"scannet/scans/scene{scene_name}/scene{scene_name}_vh_clean_2.ply",process=False)
faces = mesh.faces
vertices = np.asarray(mesh.vertices)
data_file = f"data/scannet/val/scene{scene_name}.pt"
coords_v, colors_v, labels, unique_reidx,unique_idxs,src_vertices,src_colors = read_data(data_file,rotate=True) 
position = [1.,-2.9,-0.2]
position2 = [-2.5, 1.6,0.3]
model = build_lrp_model(True)
vis_heatmap(model,f"lrp_{scene_name}",data_file,position,faces,vertices,position2)