import torch
import numpy as np
import open3d as o3d
import argparse
import os
import glob


def convert_single_file(pth_path, ply_path, data_type, vert_key, color_key, face_key):
    """
    Convert a single .pth file to .ply file.
    This version has been modified to handle cases where coordinates and colors exist as independent keys in the dictionary.
    """
    try:
        data = torch.load(pth_path, map_location=torch.device('cpu'))
    except Exception as e:
        print(f"    - Error loading .pth file: {e}")
        raise

    if data_type == 'pointcloud':
        pcd = o3d.geometry.PointCloud()

        # --- New logic: handle dictionary format point cloud ---
        if isinstance(data, dict):
            # 1. Extract coordinates (required)
            if vert_key not in data:
                raise KeyError(f"Coordinate key '{vert_key}' not found in dictionary. Available keys: {list(data.keys())}")

            points = data[vert_key]
            if isinstance(points, torch.Tensor):
                points = points.cpu().numpy()

            pcd.points = o3d.utility.Vector3dVector(points[:, :3].astype(np.float64))
            print(f"    - Loaded {len(points)} points from key '{vert_key}'.")

            # 2. Extract colors (optional)
            if color_key in data:
                colors = data[color_key]
                if isinstance(colors, torch.Tensor):
                    colors = colors.cpu().numpy()

                if colors.shape[0] != points.shape[0]:
                    print(
                        f"    - Warning: Number of points ({points.shape[0]}) and colors ({colors.shape[0]}) do not match. Colors will be ignored.")
                else:
                    # If color values are in range 0-255, normalize to 0-1
                    if np.max(colors) > 1.0:
                        colors = colors / 255.0
                    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3].astype(np.float64))
                    print(f"    - Loaded colors from key '{color_key}'.")
            else:
                print(f"    - Warning: Color key '{color_key}' not found in dictionary.")

        # --- Old logic: handle original tensor format point cloud (as fallback) ---
        elif isinstance(data, torch.Tensor):
            points = data.cpu().numpy()
            pcd.points = o3d.utility.Vector3dVector(points[:, :3].astype(np.float64))
            if points.shape[1] >= 6:
                colors = points[:, 3:6]
                if np.max(colors) > 1.0:
                    colors = colors / 255.0
                pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
        else:
            raise TypeError(f"Unsupported data type: {type(data)}. Data should be a dictionary or tensor.")

        o3d.io.write_point_cloud(ply_path, pcd)

    elif data_type == 'mesh':
        # Mesh processing logic remains unchanged
        if not isinstance(data, dict):
            raise ValueError("For mesh type, data must be a dictionary.")
        if vert_key not in data or face_key not in data:
            raise KeyError(f"Required keys for mesh ('{vert_key}', '{face_key}') not found in dictionary. Available keys: {list(data.keys())}")

        vertices = data[vert_key].cpu().numpy()
        faces = data[face_key].cpu().numpy()

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices.astype(np.float64))
        mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
        mesh.compute_vertex_normals()

        o3d.io.write_triangle_mesh(ply_path, mesh)
    else:
        raise ValueError(f"Unknown data type '{data_type}'. Please use 'pointcloud' or 'mesh'.")


def batch_convert(input_dir, output_dir, data_type, vert_key, color_key, face_key):
    if not os.path.isdir(input_dir):
        print(f"Error: Input folder '{input_dir}' does not exist.")
        return

    os.makedirs(output_dir, exist_ok=True)
    print(f"Output files will be saved to: '{output_dir}'")

    pth_files = glob.glob(os.path.join(input_dir, '*.pth'))
    if not pth_files:
        print(f"No .pth files found in '{input_dir}'.")
        return

    print(f"Found {len(pth_files)} .pth files, starting conversion...")
    success_count, fail_count = 0, 0

    for pth_path in pth_files:
        base_filename = os.path.basename(pth_path)
        ply_filename = os.path.splitext(base_filename)[0] + '.ply'
        ply_path = os.path.join(output_dir, ply_filename)

        print(f"\n[Processing] '{base_filename}'")
        try:
            convert_single_file(pth_path, ply_path, data_type, vert_key, color_key, face_key)
            print(f"  -> [Success] Saved as '{ply_path}'")
            success_count += 1
        except Exception as e:
            print(f"  -> [Failed] Error during conversion: {e}")
            fail_count += 1

    print("\n" + "=" * 50)
    print("Batch conversion completed!")
    print(f"Success: {success_count}")
    print(f"Failed: {fail_count}")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch convert .pth 3D files in a folder to .ply files.")
    parser.add_argument("input_dir", type=str, help="Input folder path containing .pth files.")
    parser.add_argument("output_dir", type=str, help="Output folder path for saving .ply files.")
    parser.add_argument("--type", type=str, required=True, choices=['pointcloud', 'mesh'],
                        help="3D data type in all .pth files: 'pointcloud' or 'mesh'.")
    # Modified default values to match your data format
    parser.add_argument("--vert_key", type=str, default="coord",
                        help="Dictionary key name for vertex/point cloud coordinate data. Default: 'coord'")
    parser.add_argument("--color_key", type=str, default="color",
                        help="Dictionary key name for color data (optional). Default: 'color'")
    parser.add_argument("--face_key", type=str, default="faces",
                        help="Dictionary key name for face data (only for 'mesh' type). Default: 'faces'")

    args = parser.parse_args()

    batch_convert(args.input_dir, args.output_dir, args.type, args.vert_key, args.color_key, args.face_key)