import os.path as osp
import sys
#用来初始化路径的，也就是之后的路径会join（path，*）
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

# Add lib to PYTHONPATH
lib_path = osp.join(this_dir, '..', 'lib')
add_path(lib_path)

coco_path = osp.join(this_dir, '..', 'data', 'coco', 'PythonAPI')
add_path(coco_path)
