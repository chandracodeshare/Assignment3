import pybullet
import cv2
import numpy as np
import math

CLIENT = 0

# ----------------------- PyBullet Joint Utilities -----------------------

def set_joint_position(body, joint, value):
    pybullet.resetJointState(body, joint, value, physicsClientId=CLIENT)

def set_joint_positions(body, joints, values):
    assert len(joints) == len(values)
    for joint, value in zip(joints, values):
        set_joint_position(body, joint, value)

# ----------------------- Color Object Detection -----------------------

def detect_color_obj_pose(target_rgb, rgb_img, depth_img):
    """Detect object of given color and return its center pixel + depth"""
    hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    target_hsv = cv2.cvtColor(np.uint8([[target_rgb]]), cv2.COLOR_RGB2HSV)[0][0]
    lower = np.array([target_hsv[0]-10, 50, 50])
    upper = np.array([target_hsv[0]+10, 255, 255])

    mask = cv2.inRange(hsv_img, lower, upper)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    max_contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(max_contour)
    if M["m00"] == 0:
        return None

    cx, cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
    depth = depth_img[cy, cx]
    return [cx, cy, depth]


# ----------------------- Rotation Matrices -----------------------

def Rx(theta):
    return np.array([[1, 0, 0],
                     [0, math.cos(theta), -math.sin(theta)],
                     [0, math.sin(theta), math.cos(theta)]])

def Ry(theta):
    return np.array([[math.cos(theta), 0, math.sin(theta)],
                     [0, 1, 0],
                     [-math.sin(theta), 0, math.cos(theta)]])

def Rz(theta):
    return np.array([[math.cos(theta), -math.sin(theta), 0],
                     [math.sin(theta), math.cos(theta), 0],
                     [0, 0, 1]])
                     
                     
# ------------------- ArUco Detection Function -------------------
def detect_aruco_markers(frame, camera_matrix, dist_coeffs, marker_size=0.10):
    """
    Detects all ArUco markers in the frame and returns a list of detections and a visualization frame.
    Each detection: (marker_id, tvec, rvec, (cx, cy))
    """
    frame_out = frame.copy()
    detections = []

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Dictionary setup
    if hasattr(cv2.aruco, "getPredefinedDictionary"):
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    else:
        aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)

    # Detector parameters (compatibility)
    if hasattr(cv2.aruco, "DetectorParameters_create"):
        parameters = cv2.aruco.DetectorParameters_create()
    else:
        parameters = cv2.aruco.DetectorParameters()

    # Detection
    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None and len(ids) > 0:
        for i, marker_id in enumerate(ids.flatten()):
            cv2.aruco.drawDetectedMarkers(frame_out, [corners[i]], np.array([[marker_id]]))
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                [corners[i]], marker_size, camera_matrix, dist_coeffs
            )
            rvec, tvec = rvecs[0][0], tvecs[0][0]

            # Get corners
            pts = corners[i][0]
            top_left = tuple(map(int, pts[0]))
            top_right = tuple(map(int, pts[1]))
            bottom_right = tuple(map(int, pts[2]))
            bottom_left = tuple(map(int, pts[3]))

            # Compute center
            cx, cy = float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))

            # Print information
            print(f"\nMarker ID: {marker_id}")
            print(f"  Center:       ({cx:.1f}, {cy:.1f})")
            print(f"  Top Left:     {top_left}")
            print(f"  Top Right:    {top_right}")
            print(f"  Bottom Right: {bottom_right}")
            print(f"  Bottom Left:  {bottom_left}")

            # Draw corner markers
            cv2.circle(frame_out, top_left, 4, (255, 0, 0), -1)
            cv2.circle(frame_out, top_right, 4, (0, 255, 0), -1)
            cv2.circle(frame_out, bottom_right, 4, (0, 0, 255), -1)
            cv2.circle(frame_out, bottom_left, 4, (255, 255, 0), -1)

            # Crosshair and axes
            cv2.line(frame_out, (int(cx - 50), int(cy)), (int(cx + 50), int(cy)), (0, 255, 0), 2)
            cv2.line(frame_out, (int(cx), int(cy - 50)), (int(cx), int(cy + 50)), (0, 255, 0), 2)
            cv2.drawFrameAxes(frame_out, camera_matrix, dist_coeffs, rvec, tvec, marker_size * 0.5)
            cv2.putText(frame_out, f"ID {marker_id} z={tvec[2]:.2f}m",
                        (int(cx - 40), int(cy - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            detections.append((int(marker_id), np.array(tvec), np.array(rvec), (cx, cy)))

    # Draw image center
    image_height, image_width = frame.shape[:2]
    cv2.drawMarker(frame_out, (image_width // 2, image_height // 2), (0, 0, 255),
                   markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

    return detections, frame_out
