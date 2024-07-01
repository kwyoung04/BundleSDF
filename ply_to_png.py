import os
import open3d as o3d
import numpy as np
import cv2
import xml.etree.ElementTree as ET
import re
import argparse

def sorted_nicely(l):
    """ 
    Sort the given list in the way that humans expect.
    """
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

def parse_matrix(file_path):
    """ Parse a 4x4 matrix from a text file """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    matrix = []
    for line in lines:
        row = list(map(float, line.strip().split()))
        matrix.append(row)

    return np.array(matrix)

def project_points(point, K):
    """ Project 3D points to 2D using camera matrix K """
    X, Y, Z = point
    
    # Project to 2D
    point_2d_homogeneous = K @ np.array([X, Y, Z])
    u, v = point_2d_homogeneous[0] / point_2d_homogeneous[2], point_2d_homogeneous[1] / point_2d_homogeneous[2]
    
    return int(u), int(v)

def annotate_image(color_img, matrix, K):
    """ Annotate image with projected 2D coordinates from 3D matrix """
    # Extract rotation (R) and translation (t) from the 4x4 matrix
    R = matrix[:3, :3]
    t = matrix[:3, 3]

    # Define the origin and the direction vectors in 3D
    origin = np.array([0, 0, 0, 1])  # Homogeneous coordinates
    x_direction = np.array([0.05, 0, 0, 1])
    y_direction = np.array([0, 0.05, 0, 1])
    z_direction = np.array([0, 0, 0.05, 1])

    # Transform the points using the transformation matrix
    transformed_origin = matrix @ origin
    transformed_x_direction = matrix @ x_direction
    transformed_y_direction = matrix @ y_direction
    transformed_z_direction = matrix @ z_direction

    # Project the transformed points to 2D
    u, v = project_points(transformed_origin[:3], K)
    u_x, v_x = project_points(transformed_x_direction[:3], K)
    u_y, v_y = project_points(transformed_y_direction[:3], K)
    u_z, v_z = project_points(transformed_z_direction[:3], K)

    # Draw a circle at the projected 2D coordinates
    cv2.circle(color_img, (u, v), 5, (0, 255, 0), -1)  # Green circle

    # Draw lines indicating the direction
    cv2.arrowedLine(color_img, (u, v), (u_x, v_x), (0, 0, 255), 2)  # Red arrow for x-direction
    cv2.arrowedLine(color_img, (u, v), (u_y, v_y), (255, 0, 0), 2)  # Blue arrow for y-direction
    cv2.arrowedLine(color_img, (u, v), (u_z, v_z), (0, 255, 255), 2)  # Yellow arrow for z-direction

    return color_img

def annotate_images_with_ob_in_cam(base_path, intrinsics):
    color_folder = os.path.join(base_path, 'Color')
    ob_in_cam_folder = os.path.join(base_path, 'result', 'ob_in_cam')

    color_files = sorted_nicely(os.listdir(color_folder))
    ob_in_cam_files = sorted_nicely(os.listdir(ob_in_cam_folder))

    for color_file, ob_in_cam_file in zip(color_files, ob_in_cam_files):
        color_path = os.path.join(color_folder, color_file)
        ob_in_cam_path = os.path.join(ob_in_cam_folder, ob_in_cam_file)

        color_img = cv2.imread(color_path)

        matrix = parse_matrix(ob_in_cam_path)

        annotated_img = annotate_image(color_img, matrix, intrinsics)

        result_path = os.path.join(base_path, 'Annotated', color_file)
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        cv2.imwrite(result_path, annotated_img)

def create_mask_from_annotation(base_path):
    xml_folder = os.path.join(base_path, 'Mask')
    image_folder = os.path.join(base_path, 'Color')
    save_folder = os.path.join(base_path, 'Mask_Annotated')

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    for xml_file in os.listdir(xml_folder):
        if xml_file.endswith('.xml'):
            tree = ET.parse(os.path.join(xml_folder, xml_file))
            root = tree.getroot()

            # Get image file name
            image_filename = root.find('filename').text

            # Load image to get the size
            image_path = os.path.join(image_folder, image_filename)
            image = cv2.imread(image_path)
            height, width = image.shape[:2]

            # Create an empty mask
            mask = np.zeros((height, width), dtype=np.uint8)

            for obj in root.findall('object'):
                polygon = obj.find('polygon')
                points = []
                for pt in polygon.findall('pt'):
                    x = int(float(pt.find('x').text))
                    y = int(float(pt.find('y').text))
                    points.append([x, y])
                points = np.array(points, dtype=np.int32)
                
                # Fill the polygon with 1
                cv2.fillPoly(mask, [points], 255)

            # Save the mask image
            mask_filename = os.path.splitext(image_filename)[0] + '_mask.png'
            save_path = os.path.join(save_folder, mask_filename)
            cv2.imwrite(save_path, mask)
            print(f"저장 완료: {save_path}")

def ply_to_depth(base_path, intrinsics, distortions):
    input_folder = os.path.join(base_path, 'Depth')
    output_folder = input_folder

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(input_folder):
        if file_name.endswith(".ply"):
            ply_path = os.path.join(input_folder, file_name)
            pcd = o3d.io.read_point_cloud(ply_path)
            points_3d = np.asarray(pcd.points, dtype=np.float32)

            points_3d = points_3d[~np.all(points_3d == 0, axis=1)]
            points_3d = points_3d
            
            camera_matrix = np.array(intrinsics, dtype=np.float32)
            distortions = np.array(distortions, dtype=np.float32)
            
            rvec = np.zeros((3, 1), dtype=np.float32)
            tvec = np.zeros((3, 1), dtype=np.float32)
            points_2d, _ = cv2.projectPoints(points_3d, rvec, tvec, camera_matrix, distortions)

            print("height: ", height)
            print("width: ", width)
            depth_image = np.zeros((height, width), dtype=np.float32)

            for i, point in enumerate(points_2d.squeeze()):
                x, y = point.astype(int)
                if 0 <= x < width and 0 <= y < height:
                    depth_image[y, x] = points_3d[i, 2]  
                    
            # normalized_depth_image = depth_image / np.max(depth_image)
            
            output_path = os.path.join(output_folder, os.path.basename(ply_path).replace('.ply', '.png'))
            cv2.imwrite(output_path, (depth_image).astype(np.uint16))  
            print(f"저장 완료: {output_path}")


def image_full_mask(base_path):
    color_folder = os.path.join(base_path, 'Color')
    mask_folder = os.path.join(base_path, 'Mask')

    if not os.path.exists(mask_folder):
        os.makedirs(mask_folder)

    for file in os.listdir(color_folder):
        if file.endswith(('.png', '.jpg', '.jpeg')):
            color_img = cv2.imread(os.path.join(color_folder, file))

            mask_img = np.ones(color_img.shape, dtype=np.uint8) * 255

            cv2.imwrite(os.path.join(mask_folder, file), mask_img)





if __name__ == "__main__":
    base_path = "/home/eric/github/data/slam/object/HD_hull"
    func = 1
    zivid = True
    
    if zivid:
        intrinsics = [
            [1783.2022705078125, 0, 848.0558325503455],
            [0, 1783.1019287109375, 581.9340314479339],
            [0, 0, 1]
        ]
        distortions = [-0.08865971863269806, 0.12055657804012299, 0.0004125814011786133, 0.0002607700298540294, -0.045955438166856766]
        width, height = (1120, 800)  # 이미지 해상도
    else:
        intrinsics = [
            [615.3791504, 0, 314.7796326],
            [0, 615.3792114, 236.6492004],
            [0, 0, 1]
        ]
        distortions = [0, 0, 0, 0, 0]
        width, height = (640, 480)  # 이미지 해상도

    if func == 1:
        print("run ply_to_depth")
        ply_to_depth(base_path, intrinsics, distortions)

    elif func == 2:
        print("run create_mask_from_annotation")
        create_mask_from_annotation(base_path)

    elif func == 3:
        print("run annotate_images_with_ob_in_cam")
        annotate_images_with_ob_in_cam(base_path, intrinsics)

    elif func == 4:
        print("run image_full_mask")
        image_full_mask(base_path)


    print("End")
