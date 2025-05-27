import cv2
import os
from datetime import datetime

def main():
    # 创建保存图像的目录（如果不存在）
    save_dir = "captured_frames"
    os.makedirs(save_dir, exist_ok=True)

    # 打开默认摄像头（通常为0）
    cap = cv2.VideoCapture(0)
    
    # 设置摄像头分辨率为1080P
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    # 检查摄像头是否成功打开
    if not cap.isOpened():
        print("无法打开摄像头")
        exit()

    print("摄像头已启动")
    print("按 's' 键保存当前帧")
    print("按 'q' 键退出程序")

    frame_count = 0

    while True:
        # 读取一帧视频
        ret, frame = cap.read()

        # 检查是否成功读取帧
        if not ret:
            print("无法获取帧，退出...")
            break

        # 显示当前帧
        cv2.imshow('Camera Feed', frame)

        # 处理键盘输入
        key = cv2.waitKey(1) & 0xFF

        # 按 's' 键保存当前帧
        if key == ord('s'):
            # 生成带时间戳的文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{save_dir}/frame_{timestamp}.jpg"

            # 保存图像
            cv2.imwrite(filename, frame)
            frame_count += 1
            print(f"已保存图像: {filename}")
            print(f"总共保存了 {frame_count} 张图像")

        # 按 'q' 键退出循环
        elif key == ord('q'):
            break

    # 释放摄像头并关闭所有窗口
    cap.release()
    cv2.destroyAllWindows()

    print(f"程序已退出，共保存了 {frame_count} 张图像")

if __name__ == "__main__":
    main()