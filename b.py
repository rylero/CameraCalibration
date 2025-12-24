
    def save_sample_code(self):
        """Generate and save sample Python code using the calibration data"""
        if not self.calibration_result:
            return

        result = self.calibration_result

        # Generate sample code based on camera model
        if result['model'] == 'fisheye':
            sample_code = f"""#!/usr/bin/env python3
\"\"\"
Sample code for using fisheye camera calibration
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Camera Model: {result['model']}
RMS Error: {result['rms_error']:.6f} pixels
\"\"\"

import cv2
import numpy as np

# Camera calibration parameters
camera_matrix = np.array({result['camera_matrix']})
dist_coeffs = np.array({result['distortion_coefficients']})
image_size = {tuple(result['image_size'])}

def undistort_image(img):
    \"\"\"Undistort a fisheye image\"\"\"
    h, w = img.shape[:2]

    # Calculate new camera matrix for fisheye
    new_camera_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        camera_matrix, dist_coeffs, (w, h), np.eye(3), balance=0.0
    )

    # Generate undistortion maps
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        camera_matrix, dist_coeffs, np.eye(3), new_camera_matrix, (w, h), cv2.CV_16SC2
    )

    # Apply undistortion
    undistorted = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)
    return undistorted

def undistort_points(points):
    \"\"\"Undistort image points (Nx2 array)\"\"\"
    points = points.reshape(-1, 1, 2).astype(np.float32)
    undistorted = cv2.fisheye.undistortPoints(
        points, camera_matrix, dist_coeffs, P=camera_matrix
    )
    return undistorted.reshape(-1, 2)

# Example usage
if __name__ == "__main__":
    # Load an image
    img = cv2.imread('your_image.jpg')

    if img is not None:
        # Undistort the image
        undistorted_img = undistort_image(img)

        # Display results
        cv2.imshow('Original', img)
        cv2.imshow('Undistorted', undistorted_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Save undistorted image
        cv2.imwrite('undistorted_output.jpg', undistorted_img)

    # Example: Undistort specific points
    # distorted_points = np.array([[320, 240], [640, 480]], dtype=np.float32)
    # undistorted_points = undistort_points(distorted_points)
    # print("Undistorted points:", undistorted_points)
"""
        else:
            sample_code = f"""#!/usr/bin/env python3
\"\"\"
Sample code for using camera calibration
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Camera Model: {result['model']}
RMS Error: {result['rms_error']:.6f} pixels
\"\"\"

import cv2
import numpy as np

# Camera calibration parameters
camera_matrix = np.array({result['camera_matrix']})
dist_coeffs = np.array({result['distortion_coefficients']})
image_size = {tuple(result['image_size'])}

def undistort_image(img, alpha=1.0):
    \"\"\"
    Undistort an image

    Args:
        img: Input distorted image
        alpha: Free scaling parameter (0-1)
               0 = all pixels valid but cropped
               1 = all source pixels retained but with black borders
    \"\"\"
    h, w = img.shape[:2]

    # Get optimal new camera matrix
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), alpha, (w, h)
    )

    # Undistort
    undistorted = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)

    # Crop to region of interest if alpha=0
    if alpha == 0:
        x, y, w, h = roi
        undistorted = undistorted[y:y+h, x:x+w]

    return undistorted

def undistort_image_remap(img, alpha=1.0):
    \"\"\"Undistort using remap (more efficient for multiple frames)\"\"\"
    h, w = img.shape[:2]

    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), alpha, (w, h)
    )

    # Generate undistortion maps (do this once, reuse for all frames)
    mapx, mapy = cv2.initUndistortRectifyMap(
        camera_matrix, dist_coeffs, None, new_camera_matrix, (w, h), cv2.CV_32FC1
    )

    # Apply remapping
    undistorted = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

    if alpha == 0:
        x, y, w, h = roi
        undistorted = undistorted[y:y+h, x:x+w]

    return undistorted

def undistort_points(points):
    \"\"\"Undistort image points (Nx2 array)\"\"\"
    points = points.reshape(-1, 1, 2).astype(np.float32)
    undistorted = cv2.undistortPoints(points, camera_matrix, dist_coeffs, P=camera_matrix)
    return undistorted.reshape(-1, 2)

def project_3d_to_2d(object_points, rvec, tvec):
    \"\"\"Project 3D points to 2D image coordinates\"\"\"
    image_points, _ = cv2.projectPoints(
        object_points, rvec, tvec, camera_matrix, dist_coeffs
    )
    return image_points.reshape(-1, 2)

# Example usage
if __name__ == "__main__":
    # Load an image
    img = cv2.imread('your_image.jpg')

    if img is not None:
        # Undistort the image (alpha=1 keeps all pixels, alpha=0 crops to valid region)
        undistorted_img = undistort_image(img, alpha=1.0)

        # Display results
        cv2.imshow('Original', img)
        cv2.imshow('Undistorted', undistorted_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Save undistorted image
        cv2.imwrite('undistorted_output.jpg', undistorted_img)

    # Example: Undistort specific points
    # distorted_points = np.array([[320, 240], [640, 480]], dtype=np.float32)
    # undistorted_points = undistort_points(distorted_points)
    # print("Undistorted points:", undistorted_points)

    # Example: Video undistortion
    # cap = cv2.VideoCapture(0)
    # while True:
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
    #     undistorted_frame = undistort_image(frame)
    #     cv2.imshow('Undistorted Video', undistorted_frame)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    # cap.release()
    # cv2.destroyAllWindows()
"""