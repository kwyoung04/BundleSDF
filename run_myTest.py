from bundlesdf import *
import argparse
import os,sys
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(code_dir)
from segmentation_utils import Segmenter

class CameraParam:
  def __init__ (self, yaml_file, device):
    self.fx: float = None
    self.fy: float = None
    self.cx: float = None
    self.cy: float = None

    self.width: int = None
    self.height: int = None

    self.distortions: np.array = None

    self.d2c_rotation: np.array = None
    self.d2c_translation: np.array = None

    self.device = device

    self._read_yml(yaml_file)

  def _read_yml(self, yaml_file):
    with open(yaml_file, 'r') as file:
      camera_intrinsics = yaml.load(file)

    if isinstance(camera_intrinsics, str):
      self.fx, self.cx = float(camera_intrinsics.split(' ')[0]), float(camera_intrinsics.split(' ')[2])
      self.fy, self.cy = float(camera_intrinsics.split(' ')[4]), float(camera_intrinsics.split(' ')[5])
      
    else:
      self.fx, _, self.cx = camera_intrinsics['intrinsics'][0]
      _, self.fy, self.cy = camera_intrinsics['intrinsics'][1]

      self.distortions = camera_intrinsics['distortions'][:]

      self.width, self.height = camera_intrinsics['resolution']

  def get_k_matrix(self):
    k = np.array([[self.fx, 0,       self.cx],
                  [0,       self.fy, self.cy],
                  [0,       0,       1]])
    return k
  
  def set_image_scale(self, w, h):
    w_sale = w / self.width
    h_sale = h / self.height
    
    self.fx = self.fx * w_sale
    self.fy = self.fy * h_sale
    self.cx = self.cx * w_sale
    self.cy = self.cy * h_sale

    self.width = w
    self.height = h
  
  def c2d_transform(self):
    color_pixel_coords = np.array([[x, y, 1] for y in range(self.width) for x in range(self.height)])

    c2d_R = np.linalg.inv(self.d2c_rotation)
    c2d_v = self.d2c_translation * -1
    
    transformed_coords = np.dot(c2d_R, color_pixel_coords.T).T + c2d_v
    #transformed_coords = transformed_coords[:, :2] / transformed_coords[:, 2][:, np.newaxis]
    
    transformed_coords = transformed_coords.astype(int)
    return transformed_coords, color_pixel_coords

    

def map_depth_to_color(u, v, depth_intrinsics, color_intrinsics):
    u_color = (u - depth_intrinsics['cx']) * (color_intrinsics['fx'] / depth_intrinsics['fx']) + color_intrinsics['cx']
    v_color = (v - depth_intrinsics['cy']) * (color_intrinsics['fy'] / depth_intrinsics['fy']) + color_intrinsics['cy']
    return int(round(u_color)), int(round(v_color))



def get_partial_time_from_filename(filename):
    parts = filename.split('_')
    parts = parts[1].split('.')
    if len(parts) > 0:
        #return int(parts[3][:11])  
        #return int(parts[1][:])  
        return int(parts[0][:])  
    return None


def run_one_video(video_dir="/home/eric/github/data/slam/object/milk/data", 
                  out_folder="/home/eric/github/data/slam/object/milk/result", 
                  use_segmenter=False):
  set_seed(0)

  os.system(f'rm -rf {out_folder} && mkdir -p {out_folder}')

  cfg_bundletrack = yaml.load(open(f"{code_dir}/BundleTrack/config_ho3d.yml",'r'))
  cfg_bundletrack['SPDLOG'] = int(args.debug_level)
  cfg_bundletrack['depth_processing']["zfar"] = 3.5  # z 거리
  cfg_bundletrack['depth_processing']["percentile"] = 30 
  cfg_bundletrack['erode_mask'] = 3  # 침식
  cfg_bundletrack['debug_dir'] = out_folder+'/'
  cfg_bundletrack['bundle']['max_BA_frames'] = 15
  cfg_bundletrack['bundle']['max_optimized_feature_loss'] = 0.01

  cfg_bundletrack['bundle']['window_size'] = 1
  cfg_bundletrack['bundle']['robust_delta'] = 0.005

  cfg_bundletrack['feature_corres']['max_dist_neighbor'] = 300
  cfg_bundletrack['feature_corres']['max_normal_neighbor'] = 45
  cfg_bundletrack['feature_corres']['max_dist_no_neighbor'] = 0.02
  cfg_bundletrack['feature_corres']['max_normal_no_neighbor'] = 45
  cfg_bundletrack['feature_corres']['map_points'] = True
  cfg_bundletrack['feature_corres']['resize'] = 400
  cfg_bundletrack['feature_corres']['suppression_patch_size'] = 5
  cfg_bundletrack['feature_corres']['rematch_after_nerf'] = True
  cfg_bundletrack['keyframe']['min_rot'] = 10

  cfg_bundletrack['ransac']['max_iter'] = 2000
  cfg_bundletrack['ransac']['num_sample'] = 3

  cfg_bundletrack['ransac']['inlier_dist'] = 0.01
  cfg_bundletrack['ransac']['inlier_normal_angle'] = 20
  cfg_bundletrack['ransac']['max_trans_neighbor'] = 2
  cfg_bundletrack['ransac']['max_rot_deg_neighbor'] = 300
  cfg_bundletrack['ransac']['max_trans_no_neighbor'] = 0.01
  cfg_bundletrack['ransac']['max_rot_no_neighbor'] = 1
  cfg_bundletrack['p2p']['max_dist'] = 0.02
  cfg_bundletrack['p2p']['max_normal_angle'] = 45
  cfg_track_dir = f'{out_folder}/config_bundletrack.yml'
  yaml.dump(cfg_bundletrack, open(cfg_track_dir,'w'))

  cfg_nerf = yaml.load(open(f"{code_dir}/config.yml",'r'))
  cfg_nerf['continual'] = True
  cfg_nerf['trunc_start'] = 0.01
  cfg_nerf['trunc'] = 0.01
  cfg_nerf['mesh_resolution'] = 0.005
  cfg_nerf['down_scale_ratio'] = 1
  cfg_nerf['fs_sdf'] = 0.1
  cfg_nerf['far'] = cfg_bundletrack['depth_processing']["zfar"]
  cfg_nerf['datadir'] = f"{cfg_bundletrack['debug_dir']}nerf_with_bundletrack_online"
  cfg_nerf['notes'] = ''
  cfg_nerf['expname'] = 'nerf_with_bundletrack_online'
  cfg_nerf['save_dir'] = cfg_nerf['datadir']
  cfg_nerf_dir = f'{out_folder}/config_nerf.yml'
  yaml.dump(cfg_nerf, open(cfg_nerf_dir,'w'))


  use_segmenter = 0
  if use_segmenter:
    segmenter = Segmenter()

  ### 3. Approach
  tracker = BundleSdf(cfg_track_dir=cfg_track_dir, cfg_nerf_dir=cfg_nerf_dir, start_nerf_keyframes=5)

  #reader = YcbineoatReader(video_dir=video_dir, shorter_side=480)

  color_dir = os.path.join(video_dir, "Color")
  depth_dir = os.path.join(video_dir, "Depth")
  mask_dir = os.path.join(video_dir, "Mask")
  
  color_files = [f for f in os.listdir(color_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
  depth_files = [f for f in os.listdir(depth_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))]
  
  color_files.sort()
  depth_files.sort()

  color_times = {get_partial_time_from_filename(f): f for f in color_files}
  depth_times = {get_partial_time_from_filename(f): f for f in depth_files}
  
  intrinsic_path = os.path.join('/'.join(video_dir.split('/')[:-1]), 'CamParam.yaml')
  intrinsic_path = os.path.join('/'.join(video_dir.split('/')[:-2]), 'CamParam.yaml')
  #intrinsic_path = os.path.join('/'.join(video_dir.split('/')[:-2]), 'cam_K.txt')
  colorIn = CameraParam(intrinsic_path, 'Color')
  
  K = colorIn.get_k_matrix()
  color_list = []
  depth_list = []
  mask_list = []

  i = 0
  for color_time, color_file in color_times.items():      
    if color_time in depth_times:
      depth_file = depth_times[color_time]
    else:
      continue  

    color_path = os.path.join(color_dir, color_file)
    depth_path = os.path.join(depth_dir, depth_file)

    try:
      color = cv2.imread(color_path)
      depth = cv2.imread(depth_path, -1)
      depth = (depth / 1e3).astype(np.float32)
      #depth = (depth / 1).astype(np.float32)
           
      if color is None or depth is None:
        print(f"Failed to load image {color_file} or {depth_file}")
    except IOError as e:
      print(f"Failed to load image {color_file} or {depth_file}: {e}")

    H,W = depth.shape[:2]
    H0, W0 = color.shape[:2]


    if not use_segmenter:
      mask_path = os.path.join(mask_dir, color_file)
      mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

      mask = cv2.resize(mask, (W,H), interpolation=cv2.INTER_AREA)


    color = cv2.resize(color, (W,H), interpolation=cv2.INTER_AREA)
    depth = cv2.resize(depth, (W,H), interpolation=cv2.INTER_AREA)

    pose_in_model = np.eye(4)

    tracker.run(color, depth, K, str(i), mask=mask, occ_mask=None, pose_in_model=pose_in_model)

    # color_list.append(color)
    # depth_list.append(depth)
    # mask_list.append(mask)

    i += 1

  tracker.on_finish()





def run_one_video_global_nerf(out_folder='/home/bowen/debug/bundlesdf_scan_coffee_415'):
  set_seed(0)

  out_folder += '/'   #!NOTE there has to be a / in the end

  cfg_bundletrack = yaml.load(open(f"{out_folder}/config_bundletrack.yml",'r'))
  cfg_bundletrack['debug_dir'] = out_folder
  cfg_track_dir = f"{out_folder}/config_bundletrack.yml"
  yaml.dump(cfg_bundletrack, open(cfg_track_dir,'w'))

  cfg_nerf = yaml.load(open(f"{out_folder}/config_nerf.yml",'r'))
  cfg_nerf['n_step'] = 2000
  cfg_nerf['N_samples'] = 64
  cfg_nerf['N_samples_around_depth'] = 256
  cfg_nerf['first_frame_weight'] = 1
  cfg_nerf['down_scale_ratio'] = 1
  cfg_nerf['finest_res'] = 256
  cfg_nerf['num_levels'] = 16
  cfg_nerf['mesh_resolution'] = 0.002
  cfg_nerf['n_train_image'] = 500
  cfg_nerf['fs_sdf'] = 0.1
  cfg_nerf['frame_features'] = 2
  cfg_nerf['rgb_weight'] = 100

  cfg_nerf['i_img'] = np.inf
  cfg_nerf['i_mesh'] = cfg_nerf['i_img']
  cfg_nerf['i_nerf_normals'] = cfg_nerf['i_img']
  cfg_nerf['i_save_ray'] = cfg_nerf['i_img']

  cfg_nerf['datadir'] = f"{out_folder}nerf_with_bundletrack_online"
  cfg_nerf['save_dir'] = copy.deepcopy(cfg_nerf['datadir'])

  os.makedirs(cfg_nerf['datadir'],exist_ok=True)

  cfg_nerf_dir = f"{cfg_nerf['datadir']}/config.yml"
  yaml.dump(cfg_nerf, open(cfg_nerf_dir,'w'))

  # reader = YcbineoatReader(video_dir=args.video_dir, downscale=1)

  tracker = BundleSdf(cfg_track_dir=cfg_track_dir, cfg_nerf_dir=cfg_nerf_dir, start_nerf_keyframes=5)
  tracker.cfg_nerf = cfg_nerf
  tracker.run_global_nerf(reader=None, get_texture=True, tex_res=512)
  tracker.on_finish()

  print(f"Done")


def draw_pose():
  K = np.loadtxt(f'{args.out_folder}/cam_K.txt').reshape(3,3)
  color_files = sorted(glob.glob(f'{args.out_folder}/color/*'))
  mesh = trimesh.load(f'{args.out_folder}/textured_mesh.obj')
  to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
  bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)
  out_dir = f'{args.out_folder}/pose_vis'
  os.makedirs(out_dir, exist_ok=True)
  logging.info(f"Saving to {out_dir}")
  for color_file in color_files:
    color = imageio.imread(color_file)
    pose = np.loadtxt(color_file.replace('.png','.txt').replace('color','ob_in_cam'))
    pose = pose@np.linalg.inv(to_origin)
    vis = draw_posed_3d_box(K, color, ob_in_cam=pose, bbox=bbox, line_color=(255,255,0))
    id_str = os.path.basename(color_file).replace('.png','')
    imageio.imwrite(f'{out_dir}/{id_str}.png', vis)

if __name__=="__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--video_dir', type=str, default="/home/eric/github/data/slam/object/zivid/240530_object/")
  parser.add_argument('--out_folder', type=str, default="/home/eric/github/data/slam/object/zivid/240530_object/result")
  #parser.add_argument('--video_dir', type=str, default="/home/eric/github/data/slam/object/HD_hull/")
  #parser.add_argument('--out_folder', type=str, default="/home/eric/github/data/slam/object/HD_hull/result")
  parser.add_argument('--use_segmenter', type=int, default=1)
  parser.add_argument('--stride', type=int, default=1, help='interval of frames to run; 1 means using every frame')
  parser.add_argument('--debug_level', type=int, default=100, help='higher means more logging')
  args = parser.parse_args()

  
  run_one_video(video_dir=args.video_dir, out_folder=args.out_folder, use_segmenter=args.use_segmenter)