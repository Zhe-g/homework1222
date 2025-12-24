import os

# 🔥 🔥 🔥 必须在导入任何 Paddle 相关库之前设置环境变量
os.environ['PADDLE_DISABLE_ONE_DNN'] = '1'
os.environ['FLAGS_use_mkldnn'] = '0'
os.environ['CPU_NUM'] = '1'  # 限制CPU线程数，避免冲突

# 现在才导入 Paddle 相关库
from paddleocr import PaddleOCR
import cv2
from ultralytics import YOLO
import detect_tools as tools
from PIL import ImageFont
import numpy as np
from pathlib import Path


def get_license_result(ocr, image):
    """
    image: 输入的车牌截取照片
    输出: 车牌号与置信度
    """
    try:
        # 确保输入是numpy数组
        if isinstance(image, np.ndarray):
            result = ocr.ocr(image, cls=True)
        else:
            result = ocr.ocr(np.array(image), cls=True)

        if result and result[0]:
            license_name, conf = result[0][0][1]
            if '·' in license_name:
                license_name = license_name.replace('·', '')
            return license_name, conf
        else:
            return None, None
    except Exception as e:
        print(f"OCR处理错误: {e}")
        return None, None


def initialize_models():
    """初始化模型，处理路径和错误"""
    # 检查模型文件是否存在
    model_path = r'E:\programingCodeFile\DeepLearninng\homework1222\result\yolov8n_carPlate\weights\best.pt'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"YOLO模型文件不存在: {model_path}")

    # 检查Paddle模型目录
    cls_model_dir = 'paddleModels/whl/cls/ch_ppocr_mobile_v2.0_cls_infer'
    rec_model_dir = 'paddleModels/whl/rec/ch/ch_PP-OCRv4_rec_infer'

    # 如果本地模型不存在，使用默认模型（会自动下载）
    if not os.path.exists(cls_model_dir):
        print(f"警告: 分类模型目录不存在，使用默认模型: {cls_model_dir}")
        cls_model_dir = None
    if not os.path.exists(rec_model_dir):
        print(f"警告: 识别模型目录不存在，使用默认模型: {rec_model_dir}")
        rec_model_dir = None

    # 加载OCR模型 - 使用更稳定的参数
    ocr = PaddleOCR(
        use_angle_cls=False,
        lang="ch",
        det=False,  # 只用识别，不用检测
        cls_model_dir=cls_model_dir,
        rec_model_dir=rec_model_dir,
        show_log=False,  # 减少日志输出
        use_gpu=False,  # 强制CPU，避免GPU冲突
        enable_mkldnn=False  # 明确禁用MKLDNN
    )

    # 加载YOLO模型
    model = YOLO(model_path, task='detect')

    return ocr, model


def load_font():
    """安全加载字体文件"""
    font_paths = [
        "Font/platech.ttf",  # 原始路径
        "C:/Windows/Fonts/simhei.ttf",  # Windows系统字体
        "C:/Windows/Fonts/msyh.ttc",  # 微软雅黑
    ]

    for font_path in font_paths:
        if os.path.exists(font_path):
            return ImageFont.truetype(font_path, 50, encoding="utf-8")

    # 如果都找不到，使用默认字体
    print("警告: 未找到指定字体，使用默认字体")
    return ImageFont.load_default()


def process_single_image(image_path, ocr, model, fontC):
    """处理单张图片"""
    # 读取图片
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"错误: 无法读取图片: {image_path}")
        return None

    print(f"正在处理图片: {image_path}")

    try:
        # YOLOv8检测车牌
        results = model(image, conf=0.25, iou=0.7)[0]

        # 获取检测框坐标
        boxes = results.boxes
        if boxes is not None and len(boxes) > 0:
            location_list = boxes.xyxy.cpu().numpy().astype(int).tolist()

            # 截取每个车牌区域
            license_imgs = []
            for x1, y1, x2, y2 in location_list:
                # 添加边界检查，防止越界
                h, w = image.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                if x2 > x1 and y2 > y1:  # 确保有效区域
                    cropImg = image[y1:y2, x1:x2]
                    license_imgs.append(cropImg)

            # 车牌识别
            license_results = []
            conf_list = []
            for i, each_img in enumerate(license_imgs):
                print(f"正在识别车牌区域 {i + 1}/{len(license_imgs)}...")
                license_num, conf = get_license_result(ocr, each_img)
                if license_num and license_num.strip():
                    license_results.append(license_num)
                    conf_list.append(conf)
                    print(f"识别结果: {license_num}, 置信度: {conf:.2f}")
                else:
                    license_results.append('无法识别')
                    conf_list.append(0)
                    print("车牌无法识别")

            # 在原图上绘制结果
            for text, box in zip(license_results, location_list):
                image = tools.drawRectBox(image, box, text, fontC)

        else:
            print("未检测到车牌")

        # 显示结果
        display_image = cv2.resize(image, dsize=None, fx=0.8, fy=0.8, interpolation=cv2.INTER_LINEAR)
        cv2.imshow(f"车牌检测结果 - {image_path.name}", display_image)
        cv2.waitKey(0)  # 等待按键
        cv2.destroyAllWindows()

        # 保存结果图片
        output_path = image_path.parent / f"result_{image_path.name}"
        cv2.imwrite(str(output_path), image)
        print(f"结果已保存到: {output_path}")

        return license_results

    except Exception as e:
        print(f"处理图片时出错: {e}")
        import traceback
        traceback.print_exc()  # 打印详细错误信息
        return None


def main():
    """主函数"""
    print("正在初始化模型...")

    try:
        # 初始化模型
        ocr, model = initialize_models()
        print("模型初始化完成")

        # 加载字体
        print("正在加载字体...")
        fontC = load_font()
        print("字体加载完成")

        # 处理单张图片
        image_path = input("请输入图片路径: ").strip().strip('"\'')
        if not os.path.exists(image_path):
            print(f"错误: 图片文件不存在: {image_path}")
            return

        results = process_single_image(Path(image_path), ocr, model, fontC)
        print(f"处理完成，识别结果: {results}")

    except Exception as e:
        print(f"程序运行错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()