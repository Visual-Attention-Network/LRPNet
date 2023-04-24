import argparse
import glob
import os
import numpy as np
import open3d
import torch
from functools import partial
from plyfile import PlyData

from voxelnet.utils.general import multi_process


remapper = np.ones(150) * (-100)
for i, x in enumerate([1,2,3,4,5,6,7,8,9,10,11,12,14,16,24,28,33,34,36,39]):
    remapper[x] = i

def collect_point_label(room,data_dir,save_dir,task):
    ply_file = os.path.join(data_dir,"scans" if task != "test" else "scans_test",room,room+"_vh_clean_2.ply")
    save_file = os.path.join(save_dir,room+".pt")

    mesh = open3d.io.read_triangle_mesh(ply_file)
    vertices = np.asarray(mesh.vertices)
    vertices = vertices - vertices.mean(0)

    labels_file_path = ply_file.replace('.ply', '.labels.ply')
    if os.path.exists(labels_file_path):
        vertex_labels = np.asarray(PlyData.read(labels_file_path)['vertex']['label'])
        vertex_labels = remapper[vertex_labels]
    else:
        vertex_labels = None

    data = dict(
        vertices = torch.from_numpy(vertices).float(),
        colors =torch.from_numpy(np.asarray(mesh.vertex_colors) * 2 - 1).float(),
        labels = torch.from_numpy(vertex_labels).long() if vertex_labels is not None else vertex_labels
    )
    torch.save(data, save_file)

def process(data_dir,save_dir,meta_dir):
    for file in glob.glob(f"{meta_dir}/scannetv2_*.txt"):
        task = file.split("_")[-1].split(".")[0]
        with open(file, "r") as f:
           rooms = [line.strip() for line in f.readlines() if len(line.strip())>0]
        
        task_save_dir = os.path.join(save_dir,task)
        os.makedirs(task_save_dir,exist_ok=True)
        func = partial(collect_point_label,data_dir=data_dir,save_dir=task_save_dir,task=task)
        multi_process(func,rooms,processes=64)            

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Scannet Data Preparation')
    parser.add_argument('--in_path', default=None, type=str, required=True,
                        help='path to scene data (default: None)')
    parser.add_argument('--out_path', default=None, type=str, required=True,
                        help='path for saving processed data (default: None)')
    parser.add_argument("--meta_path", default=None, type=str, required=True,
                        help='meta path for split data (default: None)')
    args = parser.parse_args()
    
    process(args.in_path,args.out_path,args.meta_path)