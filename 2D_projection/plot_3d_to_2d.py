import json
import argparse
import numpy as np
import cv2
from pathlib import Path
SCRIPT_DIR = Path(__file__).resolve().parent


def load_fisheye_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    size = data.get('size', [1280, 1024])
    intrinsic = np.array(data['intrinsic'], dtype=np.float64)
    affine = np.array(data.get('affine', [1.0, 0.0, 0.0]), dtype=np.float64)
    polW2C = np.array(data.get('polynomialW2C', []), dtype=np.float64)  # low->high order
    return {
        'W': int(size[0]),
        'H': int(size[1]),
        'intrinsic': intrinsic,
        'cx': float(intrinsic[0][2]),
        'cy': float(intrinsic[1][2]),
        'fx': float(intrinsic[0][0]),
        'fy': float(intrinsic[1][1]),
        'affine': affine,     # [c, d, e]
        'polW2C': polW2C
    }


def project_fisheye(pts_3d, fisheye_data):
    """Use the proper fisheye projection like in the reference code"""
    point3d = pts_3d.copy()
    point3d[:, 2] = point3d[:, 2] * -1  # Flip Z as in the reference code
    point3d = point3d.T  # Transpose to 3xN
    
    intrinsic = fisheye_data['intrinsic']
    polW2C = fisheye_data['polW2C'] 
    xc, yc = intrinsic[0, 2], intrinsic[1, 2]  # Image center
    
    point2d_list = []
    
    for i in range(point3d.shape[1]):  # For each point
        x, y, z = point3d[0, i], point3d[1, i], point3d[2, i]
        
        norm = np.sqrt(x*x + y*y)
        
        if norm != 0:
            theta = np.arctan(z / norm)
            invnorm = 1.0 / norm
            t = theta
            rho = polW2C[0]
            t_i = 1.0
            
            for j in range(1, len(polW2C)):
                t_i *= t
                rho += t_i * polW2C[j]
            
            u = x * invnorm * rho + xc
            v = y * invnorm * rho + yc
        else:
            u, v = xc, yc
            
        point2d_list.append([u, v])
    
    return np.array(point2d_list)


def project_ocam(pts_cam_m, intrinsic, polW2C, affine, img_size, flip_x=False):
    coeffs = polW2C.tolist()
    cx, cy = intrinsic[0][2], intrinsic[1][2]
    c, d, e = affine.tolist()

    X, Y, Z = pts_cam_m[:, 0], pts_cam_m[:, 1], pts_cam_m[:, 2]
    rx = -X if flip_x else X
    ry, rz = Y, Z

    # Normalize ray
    norm = np.sqrt(rx**2 + ry**2 + rz**2) + 1e-12
    rx, ry, rz = rx/norm, ry/norm, rz/norm

    rho = np.sqrt(rx * rx + rz * rz) + 1e-12
    theta = np.arctan2(rho, np.maximum(1e-12, ry))

    r = np.polyval(coeffs, theta)

    ux = rx / rho
    uz = rz / rho

    x_ = ux * r
    y_ = uz * r

    u = x_ + c * y_ + cx
    v = d * x_ + e * y_ + cy

    uv = np.stack([u, v], axis=1)
    return uv



def overlay_joints(image_path, uv, color=(0, 0, 255), radius=10, draw_idx=True, outline=True):
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    for idx, (u, v) in enumerate(uv.astype(int)):
        if outline:
            cv2.circle(img, (int(u), int(v)), radius+2, (255, 255, 255), -1, lineType=cv2.LINE_AA)
        cv2.circle(img, (int(u), int(v)), radius, color, -1, lineType=cv2.LINE_AA)
        if draw_idx:
            cv2.putText(img, str(idx), (int(u)+6, int(v)-6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255,255,255), 1, cv2.LINE_AA)
    return img
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fisheye_json', type=str, default=str(SCRIPT_DIR / 'fisheye.json'))
    ap.add_argument('--image', type=str, default='/data/My_Backup/Dataset/SceneEgo_train/train/diogo1/imgs/img_000379.jpg')
    ap.add_argument('--out', type=str, default='overlay.png')
    ap.add_argument('--model', type=str, default='pinhole', choices=['ocam', 'pinhole'])
    # ap.add_argument('--flip_x', action='store_true', help="Flip X axis before projection")
    args = ap.parse_args()

    cam = load_fisheye_json(args.fisheye_json)
    joints = np.array([[ 0.02589113,  0.35482132,  0.12483309],
       [ 0.19347556,  0.37019998,  0.13230804],
       [ 0.30226275,  0.4982058 ,  0.39474738],
       [ 0.30361623,  0.4991027 ,  0.66687554],
       [-0.13716699,  0.4286215 ,  0.15331464],
       [-0.20969634,  0.49873042,  0.44813457],
       [-0.1906363 ,  0.37877142,  0.6916565 ],
       [ 0.15666716,  0.55354095,  0.7033041 ],
       [ 0.12252908,  0.5070812 ,  1.1030318 ],
       [ 0.08527222,  0.5202438 ,  1.5376955 ],
       [ 0.12399783,  0.3804176 ,  1.6065158 ],
       [-0.02526866,  0.5959057 ,  0.7049719 ],
       [ 0.07297195,  0.8311841 ,  1.0181795 ],
       [ 0.21608864,  1.1341499 ,  1.2978636 ],
       [ 0.1529281 ,  1.0923258 ,  1.4394573 ]], dtype=np.float32)
    
    # Add the ego camera matrix from your ground truth data
    ego_camera_matrix = np.array([[ 0.98676174, -0.00513333,  0.16209538,  2.20140959],
       [ 0.16173584, -0.04249372, -0.98591876,  1.18792072],
       [ 0.01194908,  0.99908355, -0.04110094, -1.75328518],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])

    if args.model == 'pinhole':
        # Use proper fisheye projection instead of pinhole
        uv = project_fisheye(joints, cam)
    else:
        uv = project_ocam(joints,
                          intrinsic=cam['intrinsic'],
                          polW2C=cam['polW2C'],
                          affine=cam['affine'],
                          img_size=(cam['W'], cam['H']))
        print(uv)

    W, H = cam['W'], cam['H']
    finite_mask = np.isfinite(uv).all(axis=1)
    in_bounds_mask = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    print(f"Projected points: total={len(uv)}, finite={finite_mask.sum()}, in_bounds={in_bounds_mask.sum()}")

    img = overlay_joints(args.image, uv, color=(0, 0, 255), radius=6, draw_idx=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    print(f"Saved {out_path} ({img.shape[1]}x{img.shape[0]})")


if __name__ == '__main__':
    main()
