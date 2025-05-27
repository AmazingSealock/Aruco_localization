import cv2
import cv2.aruco as aruco
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import os

# 定义ArUco标记ID与区域名称的映射关系
ARUCO_DICT = {
    0: "外卖柜",
    1: "停车区",
    2: "装货区",
    3: "停车区"
}

# 配置中文字体
FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"

def create_output_directory(directory="aruco_markers"):
    """创建输出目录"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def add_text_to_image(image, text, position, font_size=20):
    """在图像上添加文本"""
    try:
        pil_img = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_img)
        font = ImageFont.truetype(FONT_PATH, font_size)
        draw.text(position, text, font=font, fill=(0, 0, 0))
        return np.array(pil_img)
    except Exception as e:
        print(f"无法添加中文标签: {e}")
        return image  # 返回未修改的图像

def generate_aruco_markers(dictionary=aruco.DICT_5X5_100, size=400, border=1):
    """生成ArUco标记图像并保存"""
    create_output_directory()
    aruco_dict = aruco.getPredefinedDictionary(dictionary)

    for marker_id, marker_name in ARUCO_DICT.items():
        marker_image = aruco.generateImageMarker(aruco_dict, marker_id, size)
        marker_with_border = cv2.copyMakeBorder(marker_image, border, border, border, border,
                                                 cv2.BORDER_CONSTANT, value=[255, 255, 255])
        marker_rgb = cv2.cvtColor(marker_with_border, cv2.COLOR_GRAY2RGB)

        text = f"{marker_name} (ID:{marker_id})"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_x = (marker_rgb.shape[1] - text_size[0]) // 2
        text_y = marker_rgb.shape[0] + 10

        result_img = add_text_to_image(marker_rgb, text, (text_x, text_y), font_size=20)

        filename = f"aruco_markers/{marker_name}_ID_{marker_id}.png"
        cv2.imwrite(filename, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
        print(f"已生成标记: {filename}")

        plt.figure(figsize=(5, 5))
        plt.imshow(result_img)
        plt.axis('off')
        plt.title(f"{marker_name} (ID:{marker_id})")
        plt.show()

def generate_marker_grid(dictionary=aruco.DICT_5X5_100, size=400, cols=2):
    """生成包含多个标记的网格图像"""
    aruco_dict = aruco.getPredefinedDictionary(dictionary)
    rows = (len(ARUCO_DICT) + cols - 1) // cols
    grid_size = size + 100
    canvas = np.ones((rows * grid_size, cols * grid_size, 3), dtype=np.uint8) * 255

    for i, (marker_id, marker_name) in enumerate(ARUCO_DICT.items()):
        row, col = divmod(i, cols)
        marker_image = aruco.generateImageMarker(aruco_dict, marker_id, size)
        marker_rgb = cv2.cvtColor(marker_image, cv2.COLOR_GRAY2RGB)

        x = col * grid_size + (grid_size - size) // 2
        y = row * grid_size + (grid_size - size) // 2
        canvas[y:y+size, x:x+size] = marker_rgb

        text = f"{marker_name} (ID:{marker_id})"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_x = x + (size - text_size[0]) // 2
        text_y = y + size + 10

        canvas = add_text_to_image(canvas, text, (text_x, text_y), font_size=24)

    cv2.imwrite("aruco_markers_grid_5x5_100.png", canvas)
    print("已生成标记网格: aruco_markers_grid.png")

    plt.figure(figsize=(10, 10))
    plt.imshow(canvas)
    plt.axis('off')
    plt.title("ArUco标记网格")
    plt.show()

if __name__ == "__main__":
    generate_aruco_markers()
    generate_marker_grid()