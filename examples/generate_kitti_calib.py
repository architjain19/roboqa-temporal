"""

################################################################

File: examples/generate_kitti_calib.py
Created: 2025-12-08
Created by: Archit Jain (architj@uw.edu)
Last Modified: 2025-12-08
Last Modified by: Archit Jain (architj@uw.edu)

#################################################################

Copyright: RoboQA-Temporal Authors
License: MIT License

################################################################

Generate synthetic KITTI calibration files for Camera-LiDAR fusion quality assessment.
This script creates `calib_velo_to_cam.txt` and `calib_cam_to_cam.txt` files
with typical KITTI calibration parameters in the specified dataset directory.

################################################################

"""


import sys
from pathlib import Path


def generate_velo_to_cam(output_path: Path):
    """Generate calib_velo_to_cam.txt with typical KITTI parameters."""
    content = """calib_time: 2011-09-26
R: 7.533745e-03 -9.999714e-01 -6.166020e-04 1.480249e-02 7.280733e-04 -9.998902e-01 9.998621e-01 7.523790e-03 1.480755e-02
T: -4.069766e-03 -7.631618e-02 -2.717806e-01
delta_f: 0.000000e+00 0.000000e+00
delta_c: 0.000000e+00 0.000000e+00
"""
    with open(output_path, 'w') as f:
        f.write(content)
    print(f"Created: {output_path}")


def generate_cam_to_cam(output_path: Path):
    """Generate calib_cam_to_cam.txt with typical KITTI parameters."""
    content = """calib_time: 2011-09-26
corner_dist: 9.950000e-02
S_00: 1.392000e+03 5.120000e+02
K_00: 9.842439e+02 0.000000e+00 6.900000e+02 0.000000e+00 9.808141e+02 2.331966e+02 0.000000e+00 0.000000e+00 1.000000e+00
D_00: -3.691481e-01 1.968681e-01 1.353473e-03 5.677587e-04 -6.770705e-02
R_00: 1.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00
T_00: 0.000000e+00 0.000000e+00 0.000000e+00
S_rect_00: 1.241000e+03 3.760000e+02
R_rect_00: 9.999239e-01 9.837760e-03 -7.445048e-03 -9.869795e-03 9.999421e-01 -4.278459e-03 7.402527e-03 4.351614e-03 9.999631e-01
P_rect_00: 7.188560e+02 0.000000e+00 6.071928e+02 0.000000e+00 0.000000e+00 7.188560e+02 1.852157e+02 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00
S_01: 1.392000e+03 5.120000e+02
K_01: 9.895267e+02 0.000000e+00 7.020000e+02 0.000000e+00 9.878386e+02 2.455590e+02 0.000000e+00 0.000000e+00 1.000000e+00
D_01: -3.639558e-01 1.788651e-01 6.029694e-04 -3.922424e-04 -5.382460e-02
R_01: 9.995448e-01 1.699833e-02 -2.431313e-02 -1.704540e-02 9.998531e-01 -1.624653e-03 2.427893e-02 1.958859e-03 9.997028e-01
T_01: -5.370000e-01 4.822061e-03 -1.252488e-02
S_rect_01: 1.241000e+03 3.760000e+02
R_rect_01: 9.999586e-01 7.036084e-03 -5.326380e-03 -7.059319e-03 9.999713e-01 -2.128940e-03 5.299185e-03 2.181472e-03 9.999835e-01
P_rect_01: 7.188560e+02 0.000000e+00 6.071928e+02 -3.861448e+02 0.000000e+00 7.188560e+02 1.852157e+02 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00
S_02: 1.392000e+03 5.120000e+02
K_02: 9.597910e+02 0.000000e+00 6.960217e+02 0.000000e+00 9.569251e+02 2.241806e+02 0.000000e+00 0.000000e+00 1.000000e+00
D_02: -3.691481e-01 1.968681e-01 1.353473e-03 5.677587e-04 -6.770705e-02
R_02: 9.999758e-01 -5.267463e-03 -4.552439e-03 5.251945e-03 9.999804e-01 -3.413835e-03 4.570332e-03 3.389843e-03 9.999838e-01
T_02: 5.956621e-02 2.900141e-04 2.577209e-03
S_rect_02: 1.241000e+03 3.760000e+02
R_rect_02: 9.999890e-01 3.580760e-03 3.244596e-03 -3.586963e-03 9.999928e-01 1.379594e-03 -3.237731e-03 -1.394165e-03 9.999942e-01
P_rect_02: 7.215377e+02 0.000000e+00 6.095593e+02 4.485728e+01 0.000000e+00 7.215377e+02 1.728540e+02 2.163791e-01 0.000000e+00 0.000000e+00 1.000000e+00 2.745884e-03
S_03: 1.392000e+03 5.120000e+02
K_03: 9.037596e+02 0.000000e+00 6.957519e+02 0.000000e+00 9.019653e+02 2.242509e+02 0.000000e+00 0.000000e+00 1.000000e+00
D_03: -3.639558e-01 1.788651e-01 6.029694e-04 -3.922424e-04 -5.382460e-02
R_03: 9.995115e-01 1.887176e-02 -2.474958e-02 -1.889593e-02 9.998178e-01 -2.017895e-03 2.472431e-02 2.384835e-03 9.996923e-01
T_03: -4.731050e-01 5.551470e-03 -5.250882e-03
S_rect_03: 1.241000e+03 3.760000e+02
R_rect_03: 9.999165e-01 1.024166e-02 -7.874936e-03 -1.029742e-02 9.999424e-01 -3.278144e-03 7.846117e-03 3.382896e-03 9.999604e-01
P_rect_03: 7.215377e+02 0.000000e+00 6.095593e+02 -3.395242e+02 0.000000e+00 7.215377e+02 1.728540e+02 2.199936e+00 0.000000e+00 0.000000e+00 1.000000e+00 2.729905e-03
"""
    with open(output_path, 'w') as f:
        f.write(content)
    print(f"Created: {output_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 generate_kitti_calib.py <dataset_path>")
        print("Example: python3 generate_kitti_calib.py dataset/2011_09_26_drive_0005_sync/")
        sys.exit(1)
    
    dataset_path = Path(sys.argv[1])
    
    if not dataset_path.exists():
        print(f"Error: Dataset path not found: {dataset_path}")
        sys.exit(1)
    
    print(f"Generating calibration files for: {dataset_path}")
    print()
    
    # Generate calibration files
    velo_to_cam = dataset_path / "calib_velo_to_cam.txt"
    cam_to_cam = dataset_path / "calib_cam_to_cam.txt"
    
    if velo_to_cam.exists():
        print(f"Warning: {velo_to_cam} already exists, skipping...")
    else:
        generate_velo_to_cam(velo_to_cam)
    
    if cam_to_cam.exists():
        print(f"Warning: {cam_to_cam} already exists, skipping...")
    else:
        generate_cam_to_cam(cam_to_cam)
    
    print("\nCalibration files generated successfully!")
    print("\nNote: These are synthetic calibration files based on typical KITTI camera parameters. For accurate results, use actual calibration files from your dataset or download official KITTI calibration files from: http://www.cvlibs.net/datasets/kitti/")


if __name__ == "__main__":
    main()
