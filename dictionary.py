import cv2
import cv2.aruco as aruco
import numpy as np
import time

# 定义ArUco标记ID与区域名称的映射关系
ARUCO_DICT = {
    0: "外卖柜",
    1: "停车区",
    2: "装货区",
    3: "停车区"
}

# 配置中文字体
# FONT_PATH = r"C:/Windows/Fonts/simhei.ttf"  # Windows系统默认黑体
# 如果是Linux/Mac系统，可以使用系统中的中文字体路径
FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"

def detect_aruco(frame, dictionary=aruco.DICT_4X4_50):
    """检测图像中的ArUco标记"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.Dictionary_get(dictionary)
    parameters = aruco.DetectorParameters_create()
    
    # 检测标记
    corners, ids, rejectedImgPoints = aruco.detectMarkers(
        gray, aruco_dict, parameters=parameters)
    
    return corners, ids

def draw_detection_results(frame, corners, ids):
    """在图像上绘制检测结果"""
    if ids is not None and len(ids) > 0:
        # 绘制标记边框
        aruco.drawDetectedMarkers(frame, corners, ids)
        
        # 遍历每个检测到的标记
        for i, corner in enumerate(corners):
            marker_id = ids[i][0]
            if marker_id in ARUCO_DICT:
                marker_name = ARUCO_DICT[marker_id]
                
                # 获取标记的四个角点坐标
                pts = corner[0].astype(np.int32)
                
                # 计算标记中心坐标
                center_x = int((pts[0][0] + pts[1][0] + pts[2][0] + pts[3][0]) / 4)
                center_y = int((pts[0][1] + pts[1][1] + pts[2][1] + pts[3][1]) / 4)
                
                # 计算标记面积（用于判断距离）
                area = cv2.contourArea(pts)
                
                # 在标记中心上方显示名称
                try:
                    # 使用PIL添加中文文本
                    from PIL import Image, ImageDraw, ImageFont
                    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(pil_img)
                    font = ImageFont.truetype(FONT_PATH, 20)
                    
                    # 显示标记名称和ID
                    text = f"{marker_name} (ID:{marker_id})"
                    draw.text((center_x - 50, center_y - 30), text, 
                             font=font, fill=(0, 0, 255))
                    
                    # 显示中心坐标和面积
                    pos_text = f"坐标: ({center_x}, {center_y})"
                    area_text = f"面积: {area:.1f}"
                    draw.text((center_x - 50, center_y - 10), pos_text, 
                             font=font, fill=(0, 0, 255))
                    draw.text((center_x - 50, center_y + 10), area_text, 
                             font=font, fill=(0, 0, 255))
                    
                    frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                except Exception as e:
                    # 如果PIL无法使用，回退到英文显示
                    cv2.putText(frame, f"ID:{marker_id}", 
                               (center_x - 20, center_y - 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    return frame

def main():
    """主函数：初始化摄像头并进行实时检测"""
    # 打开摄像头（0表示默认摄像头）
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("无法打开摄像头")
        return
    
    # 设置摄像头分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("按下ESC键退出程序")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法获取图像")
            break
        
        # 检测ArUco标记
        corners, ids = detect_aruco(frame)
        
        # 绘制检测结果
        frame = draw_detection_results(frame, corners, ids)
        
        # 显示帧率
        fps = cap.get(cv2.CAP_PROP_FPS)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 显示图像
        cv2.imshow('ArUco标记检测', frame)
        
        # 按ESC键退出
        key = cv2.waitKey(1)
        if key == 27:  # ESC键
            break
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()    
    