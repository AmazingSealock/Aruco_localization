import cv2
import numpy as np
import os

# 棋盘格参数
chessboard_size = (11, 8)  # 内角点数
square_size = 15.0  # 每格边长 (mm)

# 世界坐标系下的角点位置
objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

# 储存对应点
objpoints = []  # 3D点
imgpoints = []  # 2D点

# 图像路径
image_dir = 'calibration_images'
image_paths = [f"{image_dir}/{i}.jpg" for i in range(1, 48)]

# 可视化输出目录
output_dir = "detected_corners"
os.makedirs(output_dir, exist_ok=True)

for idx, fname in enumerate(image_paths):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        # 亚像素优化
        corners2 = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )
        objpoints.append(objp)
        imgpoints.append(corners2)

        # 显示并保存角点图
        cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)
        print(f"[{idx+1}] 角点检测成功: {fname}")
    else:
        print(f"[{idx+1}] ❌ 检测失败: {fname}")

    # 保存图像
    cv2.imwrite(f"{output_dir}/corners_{idx+1}.jpg", img)

# 相机标定
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

# 输出结果
print("\n==== 相机标定完成 ====")
print("📷 相机内参矩阵 (Camera Matrix):\n", camera_matrix)
print("🎯 畸变系数 (Distortion Coefficients):\n", dist_coeffs.ravel())

# 保存参数
np.savez('camera_calibration_result.npz', camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)

# 重投影误差
total_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    total_error += error

print("📏 平均重投影误差: ", total_error / len(objpoints))
