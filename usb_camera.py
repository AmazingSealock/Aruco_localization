import cv2
import cv2.aruco
import numpy as np
import time
from math import atan2, asin, sqrt, degrees
from PIL import Image, ImageDraw, ImageFont

# ======= 相机内参与畸变参数（替换成你标定所得的实际参数） =======
camera_matrix = np.array(
    [[1.71550426e+03, 0.00000000e+00, 9.90898248e+02],
     [0.00000000e+00, 1.71559943e+03, 3.94319799e+02],
     [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
    dtype=np.float32
)
dist_coeffs = np.array(
    [2.43887172e-01, -1.80385992e+00, 8.76212694e-04, -9.01268237e-04, 2.86093478e+00],
    dtype=np.float32
)

# ArUco 字典映射
ARUCO_DICT = {
    0: "外卖柜",
    1: "仓库区",
    2: "装货区",
    3: "停车区"
}

# 中文字体
FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
font = ImageFont.truetype(FONT_PATH, 32)

def draw_chinese(img, text, pos, color=(0,255,0)):
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    draw.text(pos, text, font=font, fill=color)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

def draw_axes(img, rvec, tvec, length=0.05):
    # 定义三个轴在标记坐标系中的终点
    axes = np.float32([[length,0,0], [0,length,0], [0,0,length]]).reshape(-1,3)
    # 原点在标记中心
    origin = np.float32([[0,0,0]]).reshape(-1,3)
    # 投影到图像平面
    imgpts, _ = cv2.projectPoints(np.vstack((origin, axes)),
                                  rvec, tvec,
                                  camera_matrix, dist_coeffs)
    o = tuple(imgpts[0].ravel().astype(int))
    x = tuple(imgpts[1].ravel().astype(int))
    y = tuple(imgpts[2].ravel().astype(int))
    z = tuple(imgpts[3].ravel().astype(int))
    # 画三条线：X 红，Y 绿，Z 蓝
    cv2.line(img, o, x, (0,0,255), 5)
    cv2.line(img, o, y, (0,255,0), 5)
    cv2.line(img, o, z, (255,0,0), 5)

def rvec_to_euler(rvec):
    """把 rvec 转成 roll, pitch, yaw（单位：度）"""
    R, _ = cv2.Rodrigues(rvec)
    # 避免 gimbal lock
    sy = sqrt(R[0,0]*R[0,0] + R[1,0]*R[1,0])
    singular = sy < 1e-6
    if not singular:
        roll  = atan2(R[2,1], R[2,2])
        pitch = atan2(-R[2,0], sy)
        yaw   = atan2(R[1,0], R[0,0])
    else:
        # 在奇异情况下，忽略 yaw
        roll  = atan2(-R[1,2], R[1,1])
        pitch = atan2(-R[2,0], sy)
        yaw   = 0
    return degrees(roll), degrees(pitch), degrees(yaw)

def input_target_ids():
    s = input("请输入要追踪的 ArUco ID（逗号分隔，如 0,2,3）：")
    try:
        ids = [int(x.strip()) for x in s.split(',') if x.strip()!='']
        print(f"当前追踪 ID 列表：{ids}")
        return ids
    except:
        print("输入格式错误，请重试。")
        return input_target_ids()


def main():

    # 第一次输入
    target_ids = input_target_ids()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    cv2.namedWindow('ArUco 轴线演示', cv2.WINDOW_NORMAL)
    print("按 'q' 键退出")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 去畸变 + 灰度
        und = cv2.undistort(frame, camera_matrix, dist_coeffs)
        gray = cv2.cvtColor(und, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = detector.detectMarkers(gray)
        disp = und.copy()

        info_lines = []

        if ids is not None and len(ids) > 0 and corners is not None and len(corners) > 0:
            # 估计姿态（边长 0.085m，根据实际修改）
            for idx, marker_id in enumerate(ids.flatten()):
                # 索引检查
                if idx >= len(corners) or corners[idx] is None or corners[idx].size == 0:
                    continue
                # 只处理目标 ID
                if marker_id not in target_ids:
                    continue

                corner = corners[idx]
                # 单 marker 姿态估计前，确保 corner 形状正确
                if corner.ndim != 3 or corner.shape[1:] != (4,2):
                    continue

                # 姿态估计
                try:
                    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                        [corner], 0.085, camera_matrix, dist_coeffs)
                except cv2.error:
                    continue

                rvec, tvec = rvecs[0].reshape(3,1), tvecs[0].reshape(3,1)

                # 画 marker 边框
                cv2.polylines(disp, [corner.astype(int)], True, (0,255,0), 5)
                # 画三维轴
                draw_axes(disp, rvecs, tvecs, length=0.03)
                
                mid = corner[0].mean(axis=0).astype(int)
                # marker_id = int(ids[idx][0])
                
                # 中文标签
                label = f"ID:{marker_id} {ARUCO_DICT.get(marker_id,'未知')}"
                disp = draw_chinese(disp, label, (mid[0]-50, mid[1]-50))

                # 准备显示位姿信息
                t = tvec.flatten()
                roll, pitch, yaw = rvec_to_euler(rvecs)
                dist = np.linalg.norm(t)
                info = (f"ID{marker_id} tx={t[0]:.2f} ty={t[1]:.2f} tz={t[2]:.2f} d={dist:.2f}m | "
                        f"roll={roll:.1f}° pitch={pitch:.1f}° yaw={yaw:.1f}°")
                info_lines.append(info)

        # 在左上角显示所有检测到的位姿信息
        for idx, line in enumerate(info_lines):
            disp = draw_chinese(disp, line, (10, 10 + idx*30))

        cv2.imshow('ArUco 轴线演示', disp)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            # 重新输入追踪 ID
            target_ids = input_target_ids()
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
