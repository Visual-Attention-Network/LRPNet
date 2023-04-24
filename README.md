# Long Range Pooling for 3D Large-Scale Scene Understanding (CVPR 2023)

The repository contains official Pytorch implementations for **LRPNet**. 

For Jittor user, https://github.com/li-xl/LRPNet is a jittor version. 

The paper is in [Here](https://arxiv.org/pdf/2301.06962).


## Installation
```
pip install numpy torch tensorboardX open3d
pip install git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0
```

## Models
We release our trained models and training logs in "work_dirs".

## Evaluation
Like VMNet, we repeat val/test for 8 times.

To evaluate the model, run:

```bash
bash run.sh 0 configs/scannet_largenet_f10_scale_val.py --task=val
# for test 
bash run.sh 0 configs/scannet_largenet_f10_scale_trainval_test.py --task=test
```

## Citation
If you find our repo useful for your research, please consider citing our paper:

```
@article{li2023long,
  title={Long Range Pooling for 3D Large-Scale Scene Understanding},
  author={Li, Xiang-Li and Guo, Meng-Hao and Mu, Tai-Jiang and Martin, Ralph R and Hu, Shi-Min},
  journal={arXiv preprint arXiv:2301.06962},
  year={2023}
}
```

## LICENSE

This repo is under the Apache-2.0 license. For commercial use, please contact the authors.