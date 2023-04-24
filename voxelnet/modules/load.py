import os 
import glob
from torch.utils.cpp_extension import load as load_cpp_ext

abs_path = os.path.split(os.path.abspath(__file__))[0]
def search_sources(dpath):
    path = os.path.join(dpath,"csrc")
    sources = list(glob.glob(path+"/*/*.cpp"))+list(glob.glob(path+"/*/*.cu"))
    sources += list(glob.glob(path+"/*.cpp"))
    return sources

voxel_module = load_cpp_ext('voxel_module', sources=search_sources(abs_path),verbose=True)
