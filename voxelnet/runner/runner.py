import os
import torch
import torch.nn.functional as F
import time
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP

from voxelnet.utils.general import check_file,clean, current_time, init_seeds,search_ckpt,build_file
from voxelnet.utils.config import get_cfg,save_cfg
from voxelnet.utils.registry import MODELS,SCHEDULERS,OPTIMS,DATASETS,HOOKS,build_from_cfg
from voxelnet.utils.metrics import confusion_matrix,get_iou,CLASS_LABELS


class Runner:
    def __init__(self):
        cfg = get_cfg()
        self.cfg = cfg

        self.work_dir = cfg.work_dir
        if cfg.clean and cfg.rank<=0:
            clean(self.work_dir)

        self.checkpoint_interval = cfg.checkpoint_interval
        self.eval_interval = cfg.eval_interval
        self.log_interval = cfg.log_interval
        self.resume_path = cfg.resume_path
        self.pretrain_path = cfg.pretrain_path

        self.model = build_from_cfg(cfg.model,MODELS).cuda()
        if self.cfg.rank>=0:
            self.model = DDP(self.model)

       
        if self.cfg.rank <= 0:
            self.val_dataset = build_from_cfg(cfg.dataset.val,DATASETS)
            self.val_data_loader = self.val_dataset.infer_data_loader()
        
        self.logger = build_from_cfg(self.cfg.logger,HOOKS,work_dir=self.work_dir,rank=self.cfg.rank)
        self.logger.print('Model Parameters: '+str(sum([x.nelement() for x in self.model.parameters()])))
        if self.cfg.rank <= 0:
            save_file = build_file(self.work_dir,prefix="config.yaml")
            save_cfg(save_file)
            self.logger.print(f"Save config to {save_file}")
        
        self.epoch = 0
        self.iter = 0
        
        assert cfg.max_epoch is not None or cfg.max_iter is not None,"Must set max epoch or max iter in config"
        self.max_epoch = cfg.max_epoch
        self.max_iter = cfg.max_iter
        
        self.start_time = -1

        if check_file(self.pretrain_path):
            self.load(self.pretrain_path,model_only=True)

        if self.resume_path is None and not cfg.clean:
            self.resume_path = search_ckpt(self.work_dir)
        if check_file(self.resume_path):
            self.resume()


    @torch.no_grad()
    def val(self,):
        self.logger.print("Validating")
        self.model.eval()
        val_reps = 8 if self.epoch == self.max_epoch else 1 
        if self.cfg.val_reps is not None:
            val_reps = self.cfg.val_reps

        size = 20
        
        store = torch.zeros(self.val_dataset.infer_offsets[-1], size).cuda()
        start = time.time()
        for rep in range(val_reps):
            for i, batch in enumerate(self.val_data_loader):

                batch = self.val_dataset.inferAfter(batch)
                # Forward
                out_euc = self.model(batch['colors_v_b'], batch['coords_v_b'],batch['vertices_v_b'])
                predictions = out_euc[batch['reindex_v_b']]
                store.index_add_(0, batch['point_ids'], predictions)
            self.logger.print(f'Val Rep: {rep}, time: {time.time()-start} s')
            self.evaluate(store.max(1)[1], self.val_dataset.infer_labels)
    
    @torch.no_grad()
    def test(self,):
        self.logger.print("Testing")
        self.model.eval()
        val_reps = self.cfg.val_reps if self.cfg.val_reps is not None else 1
        
        store = torch.zeros(self.val_dataset.infer_offsets[-1], 20).cuda()
        start = time.time()
        for rep in range(val_reps):
            for i, batch in enumerate(self.val_data_loader):

                batch = self.val_dataset.inferAfter(batch)
                # Forward
                out_euc = self.model(batch['colors_v_b'], batch['coords_v_b'],batch['vertices_v_b'])
                predictions = out_euc[batch['reindex_v_b']]

                store.index_add_(0, batch['point_ids'], predictions)
            self.logger.print(f'Test Rep: {rep}, time: {time.time()-start} s')


        inverse_mapper = np.array([1,2,3,4,5,6,7,8,9,10,11,12,14,16,24,28,33,34,36,39])

        predictions = store.max(1)[1].cpu().numpy()

        save_dir = os.path.join(self.work_dir, 'test_results')
        os.makedirs(save_dir, exist_ok=True)

        for idx, test_file in enumerate(self.val_dataset.infer_files):

            pred = predictions[self.val_dataset.infer_offsets[idx] : self.val_dataset.infer_offsets[idx + 1]]

            ori_pred = np.array([inverse_mapper[i] for i in pred])
            ori_pred = ori_pred.astype(np.int32)

            test_name = os.path.join(save_dir, test_file[-15:-3] + '.txt')

            np.savetxt(test_name, ori_pred, fmt='%d', encoding='utf8')
        
    def evaluate(self,pred_ids, gt_ids):
        class_labels = CLASS_LABELS
        N_classes = len(class_labels)

        self.logger.print(f'Evaluating {gt_ids.shape[0]} points...')
        assert pred_ids.shape == gt_ids.shape, (pred_ids.shape, gt_ids.shape)
        idxs = (gt_ids >= 0)

        # confusion = np.bincount(pred_ids[idxs] * 20 + gt_ids[idxs], minlength=400).reshape((20, 20))
        confusion = torch.bincount(pred_ids[idxs] * N_classes + gt_ids[idxs], minlength=N_classes*N_classes).reshape((N_classes, N_classes)).cpu().numpy()
        confusion = confusion.astype(np.ulonglong)

        # confusion = confusion_matrix(pred_ids, gt_ids)
        self.logger.print("calculate ious")
        class_ious = {}
        mean_iou = 0

        

        for i in range(N_classes):
            label_name = class_labels[i]
            class_ious[label_name] = get_iou(i, confusion)
            mean_iou += class_ious[label_name][0] / N_classes

        self.logger.print('classes          IoU')
        self.logger.print('----------------------------')
        class_ious_data = {}
        accs = []
        for i in range(N_classes):
            label_name = class_labels[i]
            self.logger.print('{0:<14s}: {1:>5.3f}   ({2:>6d}/{3:<6d})'.format(label_name, class_ious[label_name][0], class_ious[label_name][1], class_ious[label_name][2]))
            accs.append(class_ious[label_name][3])

            class_ious_data[f'val_class_ious/{label_name}']=class_ious[label_name][0]
        
        class_ious_data['val/miou']=mean_iou
        class_ious_data['val/macc'] = np.mean(accs)
        class_ious_data['val/oacc'] = np.sum(np.diag(confusion)) / np.sum(confusion)
        self.logger.log(class_ious_data,global_step=self.iter)

    def save(self,name):
        save_data = {
            "meta":{
                "epoch": self.epoch,
                "iter": self.iter,
                "max_epoch": self.max_epoch,
                "save_time":current_time(),
            },
            "model":self.model.state_dict(),
        }

        save_file = build_file(self.work_dir,prefix=f"checkpoints/ckpt_{name}.pt")
        torch.save(save_data,save_file)
        self.logger.print(f"Save checkpoint to {save_file}")

    def load(self, load_path, model_only=False):
        resume_data = torch.load(load_path)

        if (not model_only):
            meta = resume_data.get("meta",dict())
            self.epoch = meta.get("epoch",self.epoch)
            self.iter = meta.get("iter",self.iter)
            self.max_epoch = meta.get("max_epoch",self.max_epoch)
        if ("model" in resume_data):
            state_dict = resume_data["model"]
            for key in list(state_dict.keys()):
                if key.startswith("module."):
                    new_key = key.replace("module.", "")
                    state_dict[new_key] = state_dict[key]
                    del state_dict[key]
            self.model.load_state_dict(state_dict)
        elif ("state_dict" in resume_data):
            state_dict = resume_data["state_dict"]
            for key in list(state_dict.keys()):
                if key.startswith("module."):
                    new_key = key.replace("module.", "")
                    state_dict[new_key] = state_dict[key]
                    del state_dict[key]
            self.model.load_state_dict(state_dict)
        else:
            self.model.load_state_dict(resume_data)
        self.logger.print(f"Loading model parameters from {load_path}")

    def resume(self):
        self.load(self.resume_path)