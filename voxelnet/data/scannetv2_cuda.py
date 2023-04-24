from re import L
import numpy as np
import torch
import math
import torch.utils.data
import glob
import voxelnet.modules.functional as VF
from voxelnet.utils.registry import DATASETS

class LoadDataset:
    def __init__(self,files):
        self.files = files
    def __getitem__(self, index):
        return (torch.load(self.files[index]),index)
    def __len__(self):
        return len(self.files)

@DATASETS.register_module()
class ScanNetCuda:
    def __init__(self, data_path, mode = "train", shuffle=True, crop_by_limit = False,
                 scale = 50, 
                 batch_size = 8, 
                 full_scale = [4096, 4096, 4096], 
                 limit_numpoints = 1000000, 
                 num_workers = 4,
                 rotate=True,
                 flip_x = True,
                 rank=-1):
        
        self.rotate = rotate
        self.flip_x = flip_x
        self.rank=rank
        self.crop_by_limit = crop_by_limit
        self.mode = "val"
        val_files_path = data_path + '/val'
        test_files_path = data_path + '/test'
        val_files = sorted(glob.glob(val_files_path + '/*.pt'))
        test_files = sorted(glob.glob(test_files_path + '/*.pt'))
        self.infer_files = val_files if mode == "val" else test_files
        # self.test_files = test_files
        self.val = mode == "val" or mode == "test"
        
        self.shuffle = shuffle
        self.scale = scale
        self.batch_size = batch_size
        self.full_scale = full_scale
        self.limit_numpoints = limit_numpoints
        self.num_workers = num_workers


    def infer_data_loader(self):
        
        infer_offsets=[0]
        if self.val:
            infer_labels=[]
        for _, infer_file in enumerate(self.infer_files):
            # print("load",infer_file)
            data = torch.load(infer_file)
            infer_offsets.append(infer_offsets[-1] + data['vertices'].shape[0])
            if self.val and 'labels' in data:
                infer_labels.append(data['labels'].squeeze().numpy())
        self.infer_offsets = infer_offsets
        if self.val and len(infer_labels)>0:
            self.infer_labels = np.hstack(infer_labels)
            self.infer_labels = torch.tensor(self.infer_labels).cuda()
        
        dataset = LoadDataset(self.infer_files)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size = self.batch_size,
            collate_fn = self.inferCollate,
            num_workers = self.num_workers,
            shuffle = self.shuffle,
            pin_memory = True
            )   

    def inferCollate(self, tbl):
        # datas = [(torch.load(self.infer_files[i]),i) for i in tbl]
        # return datas
        return tbl
            
    
    def inferAfter(self, batch):
        
        coords_v_b = []         
        colors_v_b = [] 
        vertices_v_b = []
        reindex_v_b = []         
        
        point_ids = []
        num=0

        # Process in batch    
        for idx, (data,i) in enumerate(batch):
            
            # vertices
            vertices_ori = data['vertices'].cuda() 
            colors = data['colors'].cuda()    
        
            # Affine linear transformation
            trans_m = np.eye(3)
            if self.flip_x:
                trans_m[0][0] *= np.random.randint(0, 2) * 2 - 1

            trans_m *= self.scale
            theta = np.random.rand() * 2 * math.pi
            if self.rotate:
                trans_m = np.matmul(trans_m, [[math.cos(theta), math.sin(theta), 0], [-math.sin(theta), math.cos(theta), 0], [0, 0, 1]])
            
            trans_m = trans_m.astype(np.float32)
            
            # vertices_ori = np.matmul(vertices_ori, trans_m)
            vertices_ori = torch.matmul(vertices_ori, torch.tensor(trans_m,device=vertices_ori.device))

            # Random placement in the receptive field
            vertices_min = torch.min(vertices_ori, dim=0)[0].cpu().numpy()
            vertices_max = torch.max(vertices_ori, dim=0)[0].cpu().numpy()
            offset = -vertices_min + np.clip(self.full_scale - vertices_max + vertices_min - 0.001, 0, None) * np.random.rand(3) \
                + np.clip(self.full_scale - vertices_max + vertices_min + 0.001, None, 0) * np.random.rand(3)
            
            vertices_ori += torch.tensor(offset,device=vertices_ori.device)

            pointidx = torch.arange(0,vertices_ori.size(0),device=vertices_ori.device)

            # Voxelization
            coords_v = vertices_ori.int()

            # Remove duplicate items
            _, unique_idxs,unique_reidx = VF.unique(coords_v, dim=0, return_index=True,return_inverse=True)
            coords_v = coords_v[unique_idxs]
            colors_v = colors[unique_idxs]
            vertices_v = vertices_ori[unique_idxs]

            # Put into containers
            coords_v_b += [torch.cat([coords_v, torch.full(size=(coords_v.shape[0], 1),fill_value=idx,device=coords_v.device,dtype=torch.int)], 1)]
            colors_v_b += [colors_v]
            vertices_v_b += [vertices_v]
            reindex_v_b += [unique_reidx+num]
            num+=len(coords_v)
                
            point_ids += [pointidx + self.infer_offsets[i]]


        # Construct batches
        coords_v_b = torch.cat(coords_v_b, 0)
        colors_v_b = torch.cat(colors_v_b, 0)
        vertices_v_b = torch.cat(vertices_v_b, 0)
        point_ids = torch.cat(point_ids, 0)
        reindex_v_b = torch.cat(reindex_v_b,0)        
        return {'coords_v_b': coords_v_b, 'colors_v_b': colors_v_b, 'point_ids': point_ids,"reindex_v_b":reindex_v_b,"vertices_v_b":vertices_v_b}
    