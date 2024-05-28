from bundlesdf import *
import argparse
import os,sys
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(code_dir)
from segmentation_utils import Segmenter

class CameraIntrinsics:
  def __init__ (self, yaml_file, device):
    self.fx: float = None
    self.fy: float = None
    self.cx: float = None
    self.cy: float = None

    self.width: int = None
    self.height: int = None

    self.device = device

    self._read_yml(yaml_file)

  def _read_yml(self, yaml_file):
    with open(yaml_file, 'r') as file:
      camera_intrinsics = yaml.load(file)

    self.fx = camera_intrinsics[self.device]['Intrinsic']['fx']
    self.fy = camera_intrinsics[self.device]['Intrinsic']['fy']
    self.cx = camera_intrinsics[self.device]['Intrinsic']['cx']
    self.cy = camera_intrinsics[self.device]['Intrinsic']['cy']

    self.width = camera_intrinsics[self.device]['Intrinsic']['width']
    self.height = camera_intrinsics[self.device]['Intrinsic']['height']

  def get_k_matrix(self):
    k = np.array([[self.fx, 0,       self.cx],
                  [0,       self.fy, self.cy],
                  [0,       0,       1]])
    return k

def map_depth_to_color(u, v, depth_intrinsics, color_intrinsics):
    u_color = (u - depth_intrinsics['cx']) * (color_intrinsics['fx'] / depth_intrinsics['fx']) + color_intrinsics['cx']
    v_color = (v - depth_intrinsics['cy']) * (color_intrinsics['fy'] / depth_intrinsics['fy']) + color_intrinsics['cy']
    return int(round(u_color)), int(round(v_color))



def get_partial_time_from_filename(filename):
    parts = filename.split('_')
    if len(parts) > 4:
        return int(parts[3][:11])  
    return None


def run_one_video(video_dir="/home/eric/github/data/slam/object/milk/data", 
                  out_folder="/home/eric/github/data/slam/object/milk/result", 
                  use_segmenter=False):
  set_seed(0)

  os.system(f'rm -rf {out_folder} && mkdir -p {out_folder}')

  cfg_bundletrack = yaml.load(open(f"{code_dir}/BundleTrack/config_ho3d.yml",'r'))
  cfg_bundletrack['SPDLOG'] = int(args.debug_level)
  cfg_bundletrack['depth_processing']["zfar"] = 1
  cfg_bundletrack['depth_processing']["percentile"] = 95
  cfg_bundletrack['erode_mask'] = 3
  cfg_bundletrack['debug_dir'] = out_folder+'/'
  cfg_bundletrack['bundle']['max_BA_frames'] = 10
  cfg_bundletrack['bundle']['max_optimized_feature_loss'] = 0.03
  cfg_bundletrack['feature_corres']['max_dist_neighbor'] = 0.02
  cfg_bundletrack['feature_corres']['max_normal_neighbor'] = 30
  cfg_bundletrack['feature_corres']['max_dist_no_neighbor'] = 0.01
  cfg_bundletrack['feature_corres']['max_normal_no_neighbor'] = 20
  cfg_bundletrack['feature_corres']['map_points'] = True
  cfg_bundletrack['feature_corres']['resize'] = 400
  cfg_bundletrack['feature_corres']['rematch_after_nerf'] = True
  cfg_bundletrack['keyframe']['min_rot'] = 5
  cfg_bundletrack['ransac']['inlier_dist'] = 0.01
  cfg_bundletrack['ransac']['inlier_normal_angle'] = 20
  cfg_bundletrack['ransac']['max_trans_neighbor'] = 0.02
  cfg_bundletrack['ransac']['max_rot_deg_neighbor'] = 30
  cfg_bundletrack['ransac']['max_trans_no_neighbor'] = 0.01
  cfg_bundletrack['ransac']['max_rot_no_neighbor'] = 10
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
  cfg_nerf['datadir'] = f"{cfg_bundletrack['debug_dir']}/nerf_with_bundletrack_online"
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
  depth_files = [f for f in os.listdir(depth_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
  
  color_files.sort()
  depth_files.sort()

  color_times = {get_partial_time_from_filename(f): f for f in color_files}
  depth_times = {get_partial_time_from_filename(f): f for f in depth_files}
  
  intrinsic_path = os.path.join('/'.join(video_dir.split('/')[:-2]), 'CameraIntrinsics.yaml')
  colorIn = CameraIntrinsics(intrinsic_path, 'Color')
  
  scale_x = 640/colorIn.width
  scale_y = 576/colorIn.height

  colorIn.fx = colorIn.fx * scale_x
  colorIn.fy = colorIn.fy * scale_y
  colorIn.cx = colorIn.cx * scale_x
  colorIn.cy = colorIn.cy * scale_y

  K = colorIn.get_k_matrix()
  
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
      depth = cv2.imread(depth_path,-1)
      depth = (depth / 1e3).astype(np.float32)
      
      depth_normalized = cv2.normalize(depth, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
      
      if color is None or depth is None:
        print(f"Failed to load image {color_file} or {depth_file}")
    except IOError as e:
      print(f"Failed to load image {color_file} or {depth_file}: {e}")

    H,W = depth.shape[:2]
   

    if not use_segmenter:
      mask_path = os.path.join(mask_dir, color_file)
      mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
      mask = cv2.resize(mask, (W,H), interpolation=cv2.INTER_AREA)

    color = cv2.resize(color, (W,H), interpolation=cv2.INTER_AREA)
    depth = cv2.resize(depth, (W,H), interpolation=cv2.INTER_AREA)


    pose_in_model = np.eye(4)
    
    tracker.run(color, depth, K, str(i), mask=mask, occ_mask=None, pose_in_model=pose_in_model)

    i += 1


  ####################################################################
  for i in range(0,len(reader.color_files),args.stride):
    color_file = reader.color_files[i]
    color = cv2.imread(color_file)
    H0, W0 = color.shape[:2]
    depth = reader.get_depth(i)
    H,W = depth.shape[:2]
    color = cv2.resize(color, (W,H), interpolation=cv2.INTER_NEAREST)
    depth = cv2.resize(depth, (W,H), interpolation=cv2.INTER_NEAREST)

    if i==0:
      mask = reader.get_mask(0)
      mask = cv2.resize(mask, (W,H), interpolation=cv2.INTER_NEAREST)
      if use_segmenter:
        mask = segmenter.run(color_file.replace('rgb','masks'))
    else:
      if use_segmenter:
        mask = segmenter.run(color_file.replace('rgb','masks'))
      else:
        mask = reader.get_mask(i)
        mask = cv2.resize(mask, (W,H), interpolation=cv2.INTER_NEAREST)

    if cfg_bundletrack['erode_mask']>0:
      kernel = np.ones((cfg_bundletrack['erode_mask'], cfg_bundletrack['erode_mask']), np.uint8)
      mask = cv2.erode(mask.astype(np.uint8), kernel)

    id_str = reader.id_strs[i]
    pose_in_model = np.eye(4)

    K = reader.K.copy()

    tracker.run(color, depth, K, id_str, mask=mask, occ_mask=None, pose_in_model=pose_in_model)

  tracker.on_finish()

  #run_one_video_global_nerf(out_folder=out_folder)






if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', type=str, default="/home/eric/github/data/slam/object/orrbec/240523_object/")
    #parser.add_argument('--video_dir', type=str, default="/home/eric/github/data/slam/object/milk/data")
    parser.add_argument('--out_folder', type=str, default="/home/eric/github/data/slam/object/orrbec/240523_object/result")
    parser.add_argument('--use_segmenter', type=int, default=1)
    parser.add_argument('--stride', type=int, default=1, help='interval of frames to run; 1 means using every frame')
    parser.add_argument('--debug_level', type=int, default=2, help='higher means more logging')
    args = parser.parse_args()

    run_one_video(video_dir=args.video_dir, out_folder=args.out_folder, use_segmenter=args.use_segmenter)
