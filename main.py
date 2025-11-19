"""
Full-integrated Eye-in-Hand & Eye-to-Hand ArUco detection + pose estimation + UR5 IK mover.

Requirements:
- pybullet
- opencv-contrib-python (must include cv2.aruco)
- numpy

Place this file next to your 'assets' folder and utils.py (used for set_joint_positions).
"""

import os
import math
import time
import numpy as np
import cv2
import pybullet as p
import pybullet_data as pd

# ---------- Adjust these if your project uses different layout ----------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(CURRENT_DIR, "assets")

# ---------- Basic helpers (replace if you have in utils) ----------
def set_joint_positions(body, joint_indices, positions):
    """Helper to set joint target positions immediately (position control)."""
    for j, pos in zip(joint_indices, positions):
        p.resetJointState(body, j, targetValue=pos)
        p.setJointMotorControl2(body, j, p.POSITION_CONTROL, targetPosition=pos, force=200)

# Rotation helpers
def Rx(theta): return np.array([[1,0,0],[0,math.cos(theta), -math.sin(theta)],[0,math.sin(theta),math.cos(theta)]])
def Ry(theta): return np.array([[math.cos(theta),0,math.sin(theta)],[0,1,0],[-math.sin(theta),0,math.cos(theta)]])
def Rz(theta): return np.array([[math.cos(theta),-math.sin(theta),0],[math.sin(theta),math.cos(theta),0],[0,0,1]])

# ---------------------- PyBullet init ----------------------
def init_simulation(gui=True):
    client = p.connect(p.GUI if gui else p.DIRECT)
    p.resetSimulation()
    p.setAdditionalSearchPath(pd.getDataPath())
    p.setGravity(0, 0, -9.8)
    p.setTimeStep(1.0 / 240.0)
    p.loadURDF("plane.urdf")
    p.resetDebugVisualizerCamera(cameraDistance=2.0, cameraYaw=180, cameraPitch=-40, cameraTargetPosition=[0.5,0,0])
    return client

# ---------------------- Load robot & marker box ----------------------
def load_robot_and_box(robot_urdf, simple_box_urdf):
    # Load UR5 robot
    arm_id = p.loadURDF(robot_urdf, basePosition=[0,0,0], useFixedBase=True)

    # We'll load an AR-marker textured box for detection
    AR_BOX_URDF = os.path.join(ASSETS_DIR, "ar_marker_box.urdf")
    ARUCO_TEXTURE = os.path.join(ASSETS_DIR, "texture", "ar_marker_box.png")
    # If your project used a different path, replace AR_BOX_URDF above

    # load the ar marker box (non-fixed so we can inspect transform if needed)
    box_id = p.loadURDF(AR_BOX_URDF, basePosition=[0.5, 0, 0.05], useFixedBase=False)
    if os.path.exists(ARUCO_TEXTURE):
        tex = p.loadTexture(ARUCO_TEXTURE)
        p.changeVisualShape(box_id, -1, textureUniqueId=tex)

    # Initialize UR5 joints roughly (assume first 6 are arm joints)
    num_joints = p.getNumJoints(arm_id)
    ur5_joint_indices = list(range(min(6, num_joints)))
    init_conf = [0, -math.pi/2, math.pi/2, -math.pi/2, -math.pi/2, 0][:len(ur5_joint_indices)]
    set_joint_positions(arm_id, ur5_joint_indices, init_conf)

    return arm_id, box_id, ur5_joint_indices, init_conf

# ---------------------- Camera / projection utils ----------------------
def setup_camera_parameters(width=320, height=240, fov=60, near=0.05, far=5.0):
    aspect = width / height
    proj = p.computeProjectionMatrixFOV(fov, aspect, near, far)
    fov_rad = np.deg2rad(fov)
    f = (height / 2) / math.tan(fov_rad / 2)
    return {"width": width, "height": height, "fov": fov, "near": near, "far": far, "f": f, "projection": proj}

def pybullet_mat_to_np(mat_list):
    return np.array(mat_list, dtype=np.float64).reshape((4,4))

def depth_buffer_to_world(u, v, depth_buffer, view_matrix, proj_matrix, width, height):
    """
    Unproject pixel (u,v) using depth buffer to world coordinates using inverse(P*V).
    depth_buffer assumed shaped (H,W) with values in [0,1].
    """
    u = int(np.clip(u, 0, width-1))
    v = int(np.clip(v, 0, height-1))
    d = float(depth_buffer[v, u])

    # convert to NDC
    z_ndc = 2.0 * d - 1.0
    x_ndc = 2.0 * (u / (width - 1)) - 1.0
    y_ndc = 1.0 - 2.0 * (v / (height - 1))

    clip = np.array([x_ndc, y_ndc, z_ndc, 1.0], dtype=np.float64)

    V = pybullet_mat_to_np(view_matrix)
    P = pybullet_mat_to_np(proj_matrix)

    PV = P.dot(V)
    try:
        invPV = np.linalg.inv(PV)
    except np.linalg.LinAlgError:
        return None

    world_h = invPV.dot(clip)
    if abs(world_h[3]) < 1e-6:
        return None
    world = world_h[:3] / world_h[3]
    return world

# ------------- ArUco detection (NEW cv2 API) and pose estimation -------------
# New API usage: getPredefinedDictionary + ArucoDetector
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
ARUCO_PARAMS = cv2.aruco.DetectorParameters()
ARUCO_DETECTOR = cv2.aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)

def rotation_matrix_from_axes(x_axis, y_axis, z_axis):
    """Form rotation matrix with columns [x_axis, y_axis, z_axis] (orthonormal)."""
    R = np.vstack([x_axis, y_axis, z_axis]).T  # 3x3
    return R

def rotmat_to_quaternion(R):
    """Convert 3x3 rotation matrix to quaternion [x,y,z,w]."""
    # Using standard numerical stable algorithm
    m = R
    tr = m[0,0] + m[1,1] + m[2,2]
    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * S
        qx = (m[2,1] - m[1,2]) / S
        qy = (m[0,2] - m[2,0]) / S
        qz = (m[1,0] - m[0,1]) / S
    elif (m[0,0] > m[1,1]) and (m[0,0] > m[2,2]):
        S = math.sqrt(1.0 + m[0,0] - m[1,1] - m[2,2]) * 2.0
        qw = (m[2,1] - m[1,2]) / S
        qx = 0.25 * S
        qy = (m[0,1] + m[1,0]) / S
        qz = (m[0,2] + m[2,0]) / S
    elif m[1,1] > m[2,2]:
        S = math.sqrt(1.0 + m[1,1] - m[0,0] - m[2,2]) * 2.0
        qw = (m[0,2] - m[2,0]) / S
        qx = (m[0,1] + m[1,0]) / S
        qy = 0.25 * S
        qz = (m[1,2] + m[2,1]) / S
    else:
        S = math.sqrt(1.0 + m[2,2] - m[0,0] - m[1,1]) * 2.0
        qw = (m[1,0] - m[0,1]) / S
        qx = (m[0,2] + m[2,0]) / S
        qy = (m[1,2] + m[2,1]) / S
        qz = 0.25 * S
    return [qx, qy, qz, qw]

def detect_aruco_and_estimate_pose(rgb_img, depth_buf, view_matrix, proj_matrix, cam_params, debug=False):
    """
    Detect ArUco marker, unproject corners to world, estimate marker pose (position + quaternion).
    Returns (pos, quat, corners_world) or (None, None, None).
    """
    gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
    corners, ids, rejected = ARUCO_DETECTOR.detectMarkers(gray)

    if ids is None or len(corners) == 0:
        if debug:
            print("[detect] No markers")
        return None, None, None

    # Use first detected marker
    c = corners[0].reshape((4,2))  # order usually [top-left, top-right, bottom-right, bottom-left] (image coords)
    width = cam_params["width"]
    height = cam_params["height"]

    # Unproject each corner using depth buffer
    corners_world = []
    for (u_f, v_f) in c:
        u = int(np.clip(round(u_f), 0, width-1))
        v = int(np.clip(round(v_f), 0, height-1))
        world_pt = depth_buffer_to_world(u, v, depth_buf, view_matrix, proj_matrix, width, height)
        if world_pt is None:
            # fallback: try using center pixel depth
            cx = int(np.clip(round(np.mean(c[:,0])), 0, width-1))
            cy = int(np.clip(round(np.mean(c[:,1])), 0, height-1))
            world_pt = depth_buffer_to_world(cx, cy, depth_buf, view_matrix, proj_matrix, width, height)
            if world_pt is None:
                if debug: print("Unprojection failed for corner and fallback. Aborting.")
                return None, None, None
        corners_world.append(np.array(world_pt))

    corners_world = np.array(corners_world)  # shape (4,3)

    # Position: centroid of corners
    centroid = np.mean(corners_world, axis=0)

    # Compute approximate marker axes:
    # x-axis -> vector from corner0 to corner1 (top-left -> top-right)
    v01 = corners_world[1] - corners_world[0]
    v03 = corners_world[3] - corners_world[0]

    # Normalize and orthonormalize: x, y, z
    x_axis = v01 / (np.linalg.norm(v01) + 1e-9)
    y_temp = v03 / (np.linalg.norm(v03) + 1e-9)
    z_axis = np.cross(x_axis, y_temp)
    z_axis /= (np.linalg.norm(z_axis) + 1e-9)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= (np.linalg.norm(y_axis) + 1e-9)

    R = rotation_matrix_from_axes(x_axis, y_axis, z_axis)
    quat = rotmat_to_quaternion(R)  # x,y,z,w

    if debug:
        vis = rgb_img.copy()
        cv2.polylines(vis, [c.astype(np.int32)], True, (0,255,0), 2)
        cx = int(np.mean(c[:,0])); cy = int(np.mean(c[:,1]))
        cv2.circle(vis, (cx, cy), 3, (255,0,0), -1)
        cv2.putText(vis, f"pos: {centroid.round(3)}", (5,15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        cv2.imshow("ARUCO Debug", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

    return centroid.tolist(), quat, corners_world

# ---------------------- UR5 motion helper ----------------------
def move_ur5_to_pose(arm_id, joint_indices, ee_link_index, target_pos, target_quat, max_iters=100, sleep=1/240.0):
    """
    Compute IK and move UR5 towards (target_pos, target_quat).
    """
    # PyBullet IK (simple)
    ik_sol = p.calculateInverseKinematics(arm_id, ee_link_index, target_pos, targetOrientation=target_quat,
                                          maxNumIterations=200, residualThreshold=1e-4)
    # ik_sol returns a list of joint angles (for all revolute joints). We map to our joint_indices.
    # set joint motors
    for idx, j in enumerate(joint_indices):
        if idx < len(ik_sol):
            p.setJointMotorControl2(arm_id, j, p.POSITION_CONTROL, targetPosition=ik_sol[idx], force=200)

    # step simulation a bit to let the robot reach
    for _ in range(60):
        p.stepSimulation()
        time.sleep(sleep)

# ---------------------- Eye-in-hand routine ----------------------
def run_eye_in_hand():
    arm_id, box_id, ur5_joints, init_conf = load_robot_and_box(os.path.join(ASSETS_DIR,"ur5.urdf"),
                                                               os.path.join(ASSETS_DIR,"simple_box.urdf"))
    cam_params = setup_camera_parameters(width=320, height=240, fov=60)
    print("Running Eye-in-Hand. Press Ctrl+C to exit.")

    # choose camera & ee link indices:
    # CAMERA_LINK_INDEX should be the link where camera is attached (example used earlier was 6)
    # EE link we'll choose as the last link of the URDF (common)
    CAMERA_LINK_INDEX = 6 if p.getNumJoints(arm_id) > 6 else (p.getNumJoints(arm_id)-1)
    EE_LINK_INDEX = p.getNumJoints(arm_id) - 1

    try:
        while True:
            # get camera link pose (world)
            link_state = p.getLinkState(arm_id, CAMERA_LINK_INDEX)
            cam_pos = link_state[0]
            cam_orn = link_state[1]  # quaternion

            # target point in front of camera: use link orientation to compute forward vector
            R_mat = np.array(p.getMatrixFromQuaternion(cam_orn)).reshape((3,3))
            forward = R_mat[:,2]  # z-axis of link frame
            target_pt = np.array(cam_pos) + forward * 0.5

            view_matrix = p.computeViewMatrix(cam_pos, target_pt.tolist(), [0, -1, 0])
            proj_matrix = cam_params["projection"]

            w = cam_params["width"]; h = cam_params["height"]
            _, _, rgb_raw, depth_buf, _ = p.getCameraImage(w, h, view_matrix, proj_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)
            rgb = np.reshape(rgb_raw, (h, w, 4))[:, :, :3].astype(np.uint8)  # RGB order
            depth = np.reshape(depth_buf, (h, w))

            pos, quat, corners_world = detect_aruco_and_estimate_pose(rgb, depth, view_matrix, proj_matrix, cam_params, debug=True)

            if pos is not None:
                print("Detected marker (eye-in-hand) pos:", pos, "quat:", quat)
                # Move end-effector to hover 10cm above detected marker
                target_hover = [pos[0], pos[1], pos[2] + 0.10]
                # For orientation, use same quaternion found for the marker, or keep end-effector facing downwards:
                # Here we set orientation to marker's orientation.
                move_ur5_to_pose(arm_id, ur5_joints, EE_LINK_INDEX, target_hover, quat)
            else:
                # no detection; idle or rotate a bit
                pass

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("Exiting Eye-in-Hand.")

    cv2.destroyAllWindows()
    p.disconnect()

# ---------------------- Eye-to-hand routine ----------------------
def run_eye_to_hand():
    arm_id, box_id, ur5_joints, init_conf = load_robot_and_box(os.path.join(ASSETS_DIR,"ur5.urdf"),
                                                               os.path.join(ASSETS_DIR,"simple_box.urdf"))
    cam_params = setup_camera_parameters(width=320, height=240, fov=60)
    print("Running Eye-to-Hand. Press Ctrl+C to exit.")

    # Load a simple static camera URDF if you have one (else compute view from world pose)
    # Here we will define a fixed camera pose manually:
    cam_x, cam_y, cam_z = 0.0, -0.5, 1.2
    # look at workspace center
    look_at = [0.5, 0.0, 0.0]
    up_vector = [0, -1, 0]
    EE_LINK_INDEX = p.getNumJoints(arm_id) - 1

    try:
        while True:
            view_matrix = p.computeViewMatrix([cam_x, cam_y, cam_z], look_at, up_vector)
            proj_matrix = cam_params["projection"]

            w = cam_params["width"]; h = cam_params["height"]
            _, _, rgb_raw, depth_buf, _ = p.getCameraImage(w, h, view_matrix, proj_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)
            rgb = np.reshape(rgb_raw, (h, w, 4))[:, :, :3].astype(np.uint8)
            depth = np.reshape(depth_buf, (h, w))

            pos, quat, corners_world = detect_aruco_and_estimate_pose(rgb, depth, view_matrix, proj_matrix, cam_params, debug=True)
            if pos is not None:
                print("Detected marker (eye-to-hand) pos:", pos, "quat:", quat)
                target_hover = [pos[0], pos[1], pos[2] + 0.10]
                move_ur5_to_pose(arm_id, ur5_joints, EE_LINK_INDEX, target_hover, quat)
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("Exiting Eye-to-Hand.")

    cv2.destroyAllWindows()
    p.disconnect()

# ---------------------- Main ----------------------
def get_user_choice():
    print("\nSelect camera configuration:")
    print("1 - Eye-in-Hand")
    print("2 - Eye-to-Hand")
    while True:
        c = input("choice (1/2): ").strip()
        if c == '1': return 'eye_in_hand'
        if c == '2': return 'eye_to_hand'
        print("Invalid. enter 1 or 2.")

def main():
    init_simulation(gui=True)
    choice = get_user_choice()
    if choice == 'eye_in_hand':
        run_eye_in_hand()
    else:
        run_eye_to_hand()

if __name__ == "__main__":
    main()
