import sys
import numpy as np
import torch
from torch.nn import functional as F
import os.path as osp
from config import cfg
from utils.smplx import smplx
import pickle

from smplx.utils import Struct
from pytorch3d.structures import Meshes
from pytorch3d.ops import SubdivideMeshes
from pytorch3d.io import load_obj
from smplx.lbs import batch_rigid_transform
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
import math
import kaolin as kal
import open3d as o3d

class SMPLX(object):
    def __init__(self):
        self.shape_param_dim = 100
        self.expr_param_dim = 50
        self.layer_arg = {'create_global_orient': False, 'create_body_pose': False, 'create_left_hand_pose': False, 'create_right_hand_pose': False, 'create_jaw_pose': False, 'create_leye_pose': False, 'create_reye_pose': False, 'create_betas': False, 'create_expression': False, 'create_transl': False}
        self.layer = {gender: smplx.create(cfg.human_model_path, 'smplx', gender=gender, num_betas=self.shape_param_dim, num_expression_coeffs=self.expr_param_dim, use_pca=False, use_face_contour=True, **self.layer_arg) for gender in ['neutral', 'male', 'female']}
        self.face_vertex_idx = np.load(osp.join(cfg.human_model_path, 'smplx', 'SMPL-X__FLAME_vertex_ids.npy'))
        self.layer = {gender: self.get_expr_from_flame(self.layer[gender]) for gender in ['neutral', 'male', 'female']}
        self.vertex_num = 10475
        self.face_orig = self.layer['neutral'].faces.astype(np.int64)
        self.is_cavity, self.face = self.add_cavity()
        with open(osp.join(cfg.human_model_path, 'smplx', 'MANO_SMPLX_vertex_ids.pkl'), 'rb') as f:
            hand_vertex_idx = pickle.load(f, encoding='latin1')
        self.rhand_vertex_idx = hand_vertex_idx['right_hand']
        self.lhand_vertex_idx = hand_vertex_idx['left_hand']
        self.expr_vertex_idx = self.get_expr_vertex_idx()

        smplx_path = osp.join(cfg.human_model_path, 'smplx', 'SMPLX_NEUTRAL.npz')
        model_data = np.load(smplx_path, allow_pickle=True)
        data_struct = Struct(**model_data)

        self._uv = torch.Tensor(data_struct.vt).contiguous()
        self._uv_idx = torch.Tensor(data_struct.ft.astype(np.int32)).contiguous().int()
        
        # self.uv_verts,two,three = load_obj(osp.join(cfg.human_model_path, 'smplx', 'smplx_uv.obj'))
        # SMPLX joint set
        self.joint_num = 55 # 22 (body joints) + 3 (face joints) + 30 (hand joints)
        self.joints_name = \
        ('Pelvis', 'L_Hip', 'R_Hip', 'Spine_1', 'L_Knee', 'R_Knee', 'Spine_2', 'L_Ankle', 'R_Ankle', 'Spine_3', 'L_Foot', 'R_Foot', 'Neck', 'L_Collar', 'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', # body joints
        'Jaw', 'L_Eye', 'R_Eye', # face joints
        'L_Index_1', 'L_Index_2', 'L_Index_3', 'L_Middle_1', 'L_Middle_2', 'L_Middle_3', 'L_Pinky_1', 'L_Pinky_2', 'L_Pinky_3', 'L_Ring_1', 'L_Ring_2', 'L_Ring_3', 'L_Thumb_1', 'L_Thumb_2', 'L_Thumb_3', # left hand joints
        'R_Index_1', 'R_Index_2', 'R_Index_3', 'R_Middle_1', 'R_Middle_2', 'R_Middle_3', 'R_Pinky_1', 'R_Pinky_2', 'R_Pinky_3', 'R_Ring_1', 'R_Ring_2', 'R_Ring_3', 'R_Thumb_1', 'R_Thumb_2', 'R_Thumb_3' # right hand joints
        )
        self.root_joint_idx = self.joints_name.index('Pelvis')
        self.joint_part = \
        {'body': range(self.joints_name.index('Pelvis'), self.joints_name.index('R_Wrist')+1),
        'face': range(self.joints_name.index('Jaw'), self.joints_name.index('R_Eye')+1),
        'lhand': range(self.joints_name.index('L_Index_1'), self.joints_name.index('L_Thumb_3')+1),
        'rhand': range(self.joints_name.index('R_Index_1'), self.joints_name.index('R_Thumb_3')+1)}
        self.neutral_body_pose = torch.zeros((len(self.joint_part['body'])-1,3)) # 大 pose in axis-angle representation (body pose without root joint)
        self.neutral_body_pose[0] = torch.FloatTensor([0, 0, 1])
        self.neutral_body_pose[1] = torch.FloatTensor([0, 0, -1])
        self.neutral_jaw_pose = torch.FloatTensor([1/3, 0, 0])
        
        # subdivider
        self.subdivider_list = self.get_subdivider(2)
        self.face_upsampled = self.subdivider_list[-1]._subdivided_faces.cpu().numpy()
        self.vertex_num_upsampled = int(np.max(self.face_upsampled)+1)
        self.last_uv = None
        self.last_verts = None
        self.last_cords_3d = None

    def match_uv_with_mesh_and_subdivide(self, subdivide_num, feat_list=None):
        '''
            match the uv map with the smplx mesh, duplicate mesh vertices wich not in the uv map
        '''
        # compute the mapping between the smplx mesh and the uv map
        mapping = {}
        for face_idx, face in enumerate(self.face_orig):
            for i, v_idx in enumerate(face):
                uv_idx = self._uv_idx[face_idx,i]
                if mapping.get(v_idx.item()) is None:
                    mapping[v_idx.item()] = [uv_idx.item()]
                else:
                    if uv_idx.item() not in mapping[v_idx.item()]:
                        mapping[v_idx.item()].append(uv_idx.item())
        
        # face_uv = []
        # for v_idx in self.face_vertex_idx:
        #     for uv_idx in mapping[v_idx]:
        #         face_uv.append(self._uv[uv_idx])
        # face_uv = torch.stack(face_uv).float().cuda()
        # print("range of face_uv:",face_uv.min().item(), face_uv.max().item())
            
        
        ### add cavity faces in uv_face_idx
        cavity_faces = self.face[-6:]
        new_cavity_uv_faces = []
        for face in cavity_faces:
            face_vertices = []
            for v_idx in face:
                face_vertices.append(mapping[v_idx.item()][0])
            new_cavity_uv_faces.append(face_vertices)
        self._uv_idx = np.concatenate((self._uv_idx, np.array(new_cavity_uv_faces).astype(np.int32)), axis=0)
        
        ### duplicates vertices not in the uv map
        if feat_list is not None:
            feat_dims = [x.shape[1] for x in feat_list]
            feats = torch.cat(feat_list,1)
        new_vertices = []
        new_uvs = []
        new_faces = []
        new_feats = []
        vert_map = {}
        for face_idx, face in enumerate(self.face):
            face_vertices = []
            for i, v_idx in enumerate(face):
                uv_idx = self._uv_idx[face_idx,i]
                if (v_idx.item(), uv_idx.item()) not in vert_map:
                    vert_map[(v_idx.item(), uv_idx.item())] = len(new_vertices)
                    new_vertices.append(self.layer['neutral'].v_template[v_idx.item()])
                    new_uvs.append(self._uv[uv_idx.item()])
                    if feat_list is not None:
                        new_feats.append(feats[v_idx.item()])
                face_vertices.append(vert_map[(v_idx.item(), uv_idx.item())])
            new_faces.append(face_vertices)
        new_vertices = torch.stack(new_vertices).float().cuda()
        new_uvs = torch.stack(new_uvs).float().cuda()
        new_faces = torch.tensor(new_faces).long().cuda()
        new_feats = torch.stack(new_feats).float().cuda()     
        # subdivide the uv map
        mesh = Meshes(new_vertices[None,:,:], new_faces[None,:,:])


        subdivider_list = [SubdivideMeshes(mesh)]

        for i in range(subdivide_num-1):
            mesh = subdivider_list[-1](mesh)
            subdivider_list.append(SubdivideMeshes(mesh))
            
        mesh = Meshes(new_vertices[None,:,:], new_faces[None,:,:])

        feats = torch.cat([new_uvs, new_feats],dim=-1)     
        for subdivider in subdivider_list:
            mesh, feats = subdivider(mesh, feats)
        # vert_upsampled = mesh.verts_list()[0]
        face_upsampled = subdivider_list[-1]._subdivided_faces
        feats = feats[0]
        new_uvs = feats[:,:2]
        feat_list = torch.split(feats[:,2:], feat_dims, dim=1)

        
        # new_face_verts_upsampled = kal.ops.mesh.index_vertices_by_faces(vert_upsampled.unsqueeze(0), face_upsampled)
        # old_verts_upsampled = self.upsample_mesh(self.layer['neutral'].v_template.float().cuda())
        # old_face_verts_upsampled = kal.ops.mesh.index_vertices_by_faces(old_verts_upsampled.unsqueeze(0), self.subdivider_list[-1]._subdivided_faces)

        self.face_vertices_image = kal.ops.mesh.index_vertices_by_faces(new_uvs.unsqueeze(0), face_upsampled) * 2 - 1
        
        # self.get_3d_cords_from_uv(new_uvs, vert_upsampled)
        
        return new_uvs.cpu().numpy(), *feat_list
        
            
    def get_3d_cords_from_uv(self, uv_cords, verts_3d):
        '''
            args:
                uv_cords: (vertex_num, 2)
                verts_3d: (vertex_num, 3) the vertices of the smplx model
            return:
                cords_3d: (vertex_num, 3)
        '''
        face_vertices_3d = kal.ops.mesh.index_vertices_by_faces(verts_3d.unsqueeze(0), self.subdivider_list[-1]._subdivided_faces)
        face_vertices_z = face_vertices_3d[:,:,2]
        cord_map,face_index = kal.render.mesh.rasterize(512, 512, face_vertices_z, self.face_vertices_image, face_features = face_vertices_3d)
        # breakpoint()
        # import cv2 as cv
        # tm = cord_map[0].detach().cpu()
        # tm = (tm - tm.min()) / (tm.max() - tm.min())
        # cv.imshow('cord_map', tm.numpy())
        # cv.waitKey(0)
        # convert pytorch uv cords to opengl uv cords    
        # opengl_coords = torch.zeros_like(uv_cords)
        # opengl_coords[:, 0] = uv_cords[:, 0]  # Convert x
        # opengl_coords[:, 1] = (uv_cords[:, 1])   # Convert y        
        # face_idx = kal.render.mesh.texture_mapping(opengl_coords.unsqueeze(0),face_index.unsqueeze(0).float())
        cords_3d = kal.render.mesh.texture_mapping(uv_cords.unsqueeze(0),cord_map.permute(0,3,1,2))
        # get cord_3d from cord_map using grid_sample
        # cords_3d = torch.nn.functional.grid_sample(cord_map.permute(0,3,1,2), opengl_coords.reshape(1,-1,1,2), mode='bilinear', align_corners=True)
        # cords_3d = cords_3d.reshape(uv_cords.shape[0], 3)
        
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(cords_3d[0].detach().cpu().numpy())
        # pcd_l = o3d.geometry.PointCloud()
        # pcd_l.points = o3d.utility.Vector3dVector(verts_3d.detach().cpu().numpy())
        # o3d.visualization.draw_geometries([pcd])
        # breakpoint()
        return cords_3d[0]
    
    # def render_normal_uv(self, )
    
    def get_expr_from_flame(self, smplx_layer):
        flame_layer = smplx.create(cfg.human_model_path, 'flame', gender='neutral', num_betas=self.shape_param_dim, num_expression_coeffs=self.expr_param_dim)
        smplx_layer.expr_dirs[self.face_vertex_idx,:,:] = flame_layer.expr_dirs
        return smplx_layer
       
    def set_id_info(self, shape_param, face_offset, joint_offset, locator_offset):
        self.shape_param = shape_param
        self.face_offset = face_offset
        self.joint_offset = joint_offset
        self.locator_offset = locator_offset

    def get_joint_offset(self, joint_offset):
        weight = torch.ones((1,self.joint_num,1)).float().cuda()
        weight[:,self.root_joint_idx,:] = 0
        joint_offset = joint_offset * weight
        return joint_offset

    def get_subdivider(self, subdivide_num):
        vert = self.layer['neutral'].v_template.float().cuda()
        face = torch.LongTensor(self.face).cuda()
        mesh = Meshes(vert[None,:,:], face[None,:,:])

        subdivider_list = [SubdivideMeshes(mesh)]

        for i in range(subdivide_num-1):
            mesh = subdivider_list[-1](mesh)
            subdivider_list.append(SubdivideMeshes(mesh))

        return subdivider_list

    def upsample_mesh(self, vert, feat_list=None):
        face = torch.LongTensor(self.face).cuda()
        mesh = Meshes(vert[None,:,:], face[None,:,:])
        if feat_list is None:
            for subdivider in self.subdivider_list:
                mesh = subdivider(mesh)
            vert = mesh.verts_list()[0]
            return vert
        else:
            feat_dims = [x.shape[1] for x in feat_list]
            feats = torch.cat(feat_list,1)
            for subdivider in self.subdivider_list:
                mesh, feats = subdivider(mesh, feats)
            vert = mesh.verts_list()[0]
            feats = feats[0]
            feat_list = torch.split(feats, feat_dims, dim=1)
            return vert, *feat_list

    def add_cavity(self):
        lip_vertex_idx = [2844, 2855, 8977, 1740, 1730, 1789, 8953, 2892]
        is_cavity = np.zeros((self.vertex_num), dtype=np.float32)
        is_cavity[lip_vertex_idx] = 1.0

        cavity_face = [[0,1,7], [1,2,7], [2, 3,5], [3,4,5], [2,5,6], [2,6,7]]
        face_new = list(self.face_orig)
        for face in cavity_face:
            v1, v2, v3 = face
            face_new.append([lip_vertex_idx[v1], lip_vertex_idx[v2], lip_vertex_idx[v3]])
        face_new = np.array(face_new, dtype=np.int64)
        return is_cavity, face_new
 
    def get_expr_vertex_idx(self):
        # FLAME 2020 has all vertices of expr_vertex_idx. use FLAME 2019
        with open(osp.join(cfg.human_model_path, 'flame', '2019', 'generic_model.pkl'), 'rb') as f:
            flame_2019 = pickle.load(f, encoding='latin1')
        vertex_idxs = np.where((flame_2019['shapedirs'][:,:,300:300+self.expr_param_dim] != 0).sum((1,2)) > 0)[0] # FLAME.SHAPE_SPACE_DIM == 300

        # exclude neck and eyeball regions
        flame_joints_name = ('Neck', 'Head', 'Jaw', 'L_Eye', 'R_Eye')
        expr_vertex_idx = []
        flame_vertex_num = flame_2019['v_template'].shape[0]
        is_neck_eye = torch.zeros((flame_vertex_num)).float()
        is_neck_eye[flame_2019['weights'].argmax(1)==flame_joints_name.index('Neck')] = 1
        is_neck_eye[flame_2019['weights'].argmax(1)==flame_joints_name.index('L_Eye')] = 1
        is_neck_eye[flame_2019['weights'].argmax(1)==flame_joints_name.index('R_Eye')] = 1
        for idx in vertex_idxs:
            if is_neck_eye[idx]:
                continue
            expr_vertex_idx.append(idx)

        expr_vertex_idx = np.array(expr_vertex_idx)
        expr_vertex_idx = self.face_vertex_idx[expr_vertex_idx]

        return expr_vertex_idx
    
    def get_arm(self, mesh_neutral_pose, skinning_weight):
        normal = Meshes(verts=mesh_neutral_pose[None,:,:], faces=torch.LongTensor(self.face_upsampled).cuda()[None,:,:]).verts_normals_packed().reshape(self.vertex_num_upsampled,3).detach()
        part_label = skinning_weight.argmax(1)
        is_arm = 0
        for name in ('R_Shoulder', 'R_Elbow', 'L_Shoulder', 'L_Elbow'):
            is_arm = is_arm + (part_label == self.joints_name.index(name))
        is_arm = (is_arm > 0)
        is_upper_arm = is_arm * (normal[:,1] > math.cos(math.pi/3))
        is_lower_arm = is_arm * (normal[:,1] <= math.cos(math.pi/3))
        return is_upper_arm, is_lower_arm


smpl_x = SMPLX()
