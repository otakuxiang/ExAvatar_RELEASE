import os
import os.path as osp
from glob import glob
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, dest='root_path')
    args = parser.parse_args()
    assert args.root_path, "Please set root_path."
    return args

# get path
args = parse_args()
root_path = args.root_path
if root_path[-1] == '/':
    subject_id = root_path.split('/')[-2]
else:
    subject_id = root_path.split('/')[-1]



# DECA (get initial FLAME parameters)
os.chdir('./DECA')
cmd = 'python run_deca.py --root_path ' + root_path
print(cmd)
os.system(cmd)
os.chdir('..')

# Hand4Whole (get initial SMPLX parameters)
os.chdir('./Hand4Whole_RELEASE/demo')
cmd = 'python run_hand4whole.py --gpu 0 --root_path ' + root_path
print(cmd)
os.system(cmd)
os.chdir('../../')

# mmpose (get 2D whole-body keypoints)
os.chdir('./mmpose')
cmd = 'python run_mmpose.py --root_path ' + root_path
print(cmd)
os.system(cmd)
os.chdir('..')

# fit SMPLX
os.chdir('../main')
cmd = 'python fit.py --subject_id ' + subject_id
print(cmd)
os.system(cmd)
os.chdir('../tools')
cmd = 'mv ' + osp.join('..', 'output', 'result', subject_id, '*') + ' ' + osp.join(root_path, '.')
print(cmd)
os.system(cmd)

# unwrap textures of FLAME
os.chdir('../main')
cmd = 'python unwrap.py --subject_id ' + subject_id
print(cmd)
os.system(cmd)
os.chdir('../tools')
cmd = 'mv ' + osp.join('..', 'output', 'result', subject_id, 'unwrapped_textures', '*') + ' ' + osp.join(root_path, 'smplx_optimized', '.')
os.system(cmd)

# smooth SMPLX
# cmd = 'python smooth_smplx_params.py --root_path ' + root_path
# print(cmd)
# os.system(cmd)
