from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
from PIL import Image
import cv2
import numpy as np
import torch
from torchvision.transforms import Compose
encoder = 'vits' # can also be 'vitb' or 'vitl'
depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_{:}14'.format(encoder)).eval()
transform = Compose([
    Resize(
        width=518,
        height=518,
        resize_target=False,
        keep_aspect_ratio=True,
        ensure_multiple_of=14,
        resize_method='lower_bound',
        image_interpolation_method=cv2.INTER_CUBIC,
    ),
    NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    PrepareForNet(),
])
############################################
def project_image_to_target_view(image, depth_map, src_transform, target_transform, camera_angle_x, image_width,
                                 image_height):
    """
    Project an image from source camera to target camera view using depth map with PyTorch GPU acceleration.

    Args:
        image (np.ndarray): Original image (H x W x 3).
        depth_map (np.ndarray): Depth map corresponding to the image (H x W).
        src_transform (np.ndarray): Source camera transform matrix (4x4).
        target_transform (np.ndarray): Target camera transform matrix (4x4).
        camera_angle_x (float): Horizontal field of view of the camera in radians.
        image_width (int): Width of the image.
        image_height (int): Height of the image.

    Returns:
        np.ndarray: Reprojected image in the target camera view.
    """

    # Convert inputs to torch tensors
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Compute intrinsic matrix from horizontal FoV
    f_x = image_width / (2 * np.tan(camera_angle_x / 2))
    f_y = f_x * (image_height / image_width)  # Maintain aspect ratio
    c_x = image_width / 2
    c_y = image_height / 2
    K = torch.tensor([[f_x, 0, c_x], [0, f_y, c_y], [0, 0, 1]], dtype=torch.float32, device=device)

    # Transform matrices
    R_src = torch.tensor(src_transform[:3, :3], dtype=torch.float32, device=device)
    T_src = torch.tensor(src_transform[:3, 3], dtype=torch.float32, device=device).view(3, 1)
    R_tgt = torch.tensor(target_transform[:3, :3], dtype=torch.float32, device=device)
    T_tgt = torch.tensor(target_transform[:3, 3], dtype=torch.float32, device=device).view(3, 1)

    # Convert image and depth map to torch tensors
    image_tensor = torch.tensor(image, dtype=torch.float32, device=device)
    depth_tensor = torch.tensor(depth_map, dtype=torch.float32, device=device)
    valid_depth_mask = (depth_tensor > 1e-3)

    # Create meshgrid for pixel coordinates
    x = torch.arange(image_width, device=device)
    y = torch.arange(image_height, device=device)
    x, y = torch.meshgrid(x, y, indexing='xy')  # Pixel coordinates
    ones = torch.ones_like(x, device=device)
    pixels_homogeneous = torch.stack((x, y, ones), dim=0).view(3, -1)  # (3, H*W)

    # Convert pixel coordinates to camera coordinates in the source view
    depth = depth_tensor.view(-1)  # (H*W,)
    cam_coords_src = torch.linalg.inv(K) @ (pixels_homogeneous * depth)  # (3, H*W)

    # Transform to world coordinates
    world_coords = R_src @ cam_coords_src + T_src

    # Transform to target camera coordinates
    cam_coords_tgt = torch.linalg.inv(R_tgt) @ (world_coords - T_tgt)  # (3, H*W)

    # Project to target image plane
    pixels_tgt_homogeneous = K @ cam_coords_tgt  # (3, H*W)
    pixels_tgt = pixels_tgt_homogeneous[:2] / pixels_tgt_homogeneous[2]  # Normalize by z (2, H*W)

    # Round to nearest pixel and clip to image boundaries
    u_tgt = torch.round(pixels_tgt[0]).long()
    v_tgt = torch.round(pixels_tgt[1]).long()
    valid_mask = valid_depth_mask.view(-1) & (u_tgt >= 0) & (u_tgt < image_width) & (v_tgt >= 0) & (v_tgt < image_height)

    # Initialize the reprojected image
    reprojected_image = torch.full_like(image_tensor, fill_value=255)  # Assume white background

    # Map valid pixels
    u_src = x.reshape(-1)[valid_mask]
    v_src = y.reshape(-1)[valid_mask]

    u_tgt_valid = u_tgt[valid_mask]
    v_tgt_valid = v_tgt[valid_mask]
    reprojected_image[v_tgt_valid, u_tgt_valid] = image_tensor[v_src, u_src]

    # Convert back to numpy array
    return reprojected_image.cpu().numpy()

# Example usage
if __name__ == "__main__":
    # img_path = '/cluster/work/cvl/jiezcao/jiameng/layered-GS/ckpt/hotdog_far/test/ours_30000_far/gt/r_0.png'  # renders/r_0.png
    # img_path = '/cluster/work/cvl/jiezcao/jiameng/3D-Gaussian/nerf_synthetic/hotdog/test/r_0.png'
    img_path = 'scribble.jpg'
    image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) / 255.0
    image = transform({'image': image})['image']
    image_th = torch.from_numpy(image).unsqueeze(0)

    # depth shape: 1xHxW
    # depth_map = np.load("depth_map.npy")  # Depth map[H,W]
    depth_map = depth_anything(image_th)
    depth_map = torch.clamp(depth_map, min=1e-3)  # min_depth equals 0.001
    depth_map = depth_map.squeeze(0).detach().numpy() # (518,518)

    # Load img, depth
    image = cv2.imread(img_path)  # Original image
    image_height, image_width = image.shape[:2]


    # resize from (518,518)
    depth_map = cv2.resize(depth_map, (image_width, image_height), interpolation=cv2.INTER_LINEAR)
    # save_depth
    depth_map_normalized = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map)) * 255
    depth_map_normalized = depth_map_normalized.astype(np.uint8)
    depth_map_normalized = cv2.applyColorMap(depth_map_normalized, cv2.COLORMAP_INFERNO)
    # cv2.imwrite('depth_map_resized.png', depth_map_normalized)
    cv2.imwrite('depth_scibble_resized.png', depth_map_normalized)


    # Camera parameters from JSON
    camera_angle_x = 0.6911112070083618
    # image_width, image_height  = 800, 800
    src_transform = np.array([
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, -0.7341099977493286, 0.6790306568145752, 2.737260103225708],
        [0.0, 0.6790306568145752, 0.7341099381446838, 2.959291696548462],
        [0.0, 0.0, 0.0, 1.0]
    ])# np.array()
    target_transform = np.array([
        [-0.9980267286300659, 0.04609514772891998, -0.042636688798666, -0.17187398672103882],
        [-0.06279052048921585, -0.7326614260673523, 0.6776907444000244, 2.731858730316162],
        [-3.7252898543727042e-09, 0.6790306568145752, 0.7341099381446838, 2.959291696548462],
        [0.0, 0.0, 0.0, 1.0]
    ])

    # Perform reprojection
    reprojected_image = project_image_to_target_view(image, depth_map, src_transform, target_transform, camera_angle_x, image_width, image_height)

    # Save or visualize the result
    # cv2.imwrite("reprojected_image.png", reprojected_image)
    cv2.imwrite("reprojected_scribble.png", reprojected_image)

