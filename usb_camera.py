import cv2
import numpy as np

# ======= 相机内参与畸变参数（替换成你标定所得的实际参数） =======
camera_matrix = np.array( [[1.71550426e+03, 0.00000000e+00, 9.90898248e+02],
 [0.00000000e+00, 1.71559943e+03, 3.94319799e+02],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], dtype=np.float32)  # 替换 fx, fy, cx, cy
dist_coeffs = np.array([2.43887172e-01, -1.80385992e+00,  8.76212694e-04, -9.01268237e-04, 2.86093478e+00], dtype=np.float32)  # 替换为真实值

# ======= ArUco字典与ID对应含义 =======
ARUCO_DICT = {
    0: "外卖柜",
    1: "停车区",
    2: "装货区",
    3: "停车区"
}
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
parameters = cv2.aruco.DetectorParameters_create()

# ======= 打开摄像头 =======
cap = cv2.VideoCapture(0)  # 你也可以换成视频文件路径

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 检测ArUco标记
    corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None:
        # 绘制检测到的方框
        cv2.aruco.drawDetectedMarkers(frame, corners, ids, borderColor=(0, 255, 0))

        # 位姿估计
        marker_length = 0.1  # ArUco边长，单位：米（你需要替换成你的实际值）
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coeffs)

        for i, id_ in enumerate(ids.flatten()):
            rvec, tvec = rvecs[i][0], tvecs[i][0]

            # 绘制坐标轴
            cv2.aruco.drawAxis(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.03)

            # 显示含义文字
            meaning = ARUCO_DICT.get(id_, f"未知ID:{id_}")
            c = corners[i][0]
            corner_center = tuple(np.mean(c, axis=0).astype(int))
            cv2.putText(frame, meaning, corner_center, cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)

            # 控制台输出定位信息
            print(f"ID {id_} ({meaning}) --> tvec: {tvec}, rvec: {rvec}")

    cv2.imshow("ArUco Localization", frame)
    key = cv2.waitKey(1)
    if key == 27:  # 按下 ESC 退出
        break

cap.release()
cv2.destroyAllWindows()
