import os
import warnings
import threading

# 🔥 🔥 🔥 必须在导入任何库之前设置环境变量，避免冲突
os.environ['PADDLE_DISABLE_ONE_DNN'] = '1'
os.environ['FLAGS_use_mkldnn'] = '0'
os.environ['CPU_NUM'] = '1'  # 限制CPU线程数，避免冲突
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # 使用GPU
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # 避免OpenMP冲突
os.environ['OMP_NUM_THREADS'] = '1'  # 限制OpenMP线程数

# 禁用不必要的警告
warnings.filterwarnings('ignore')

# 现在才导入 Paddle 相关库
from paddleocr import PaddleOCR
import cv2
import torch  # 在导入ultralytics前先导入torch
from ultralytics import YOLO
import detect_tools as tools
from PIL import ImageFont
import numpy as np
from pathlib import Path

# 强制使用CPU
torch.set_num_threads(1)

# 全局线程锁，保护多线程使用模型
model_lock = threading.Lock()


def get_license_result(ocr, image):
    """
    image: 输入的车牌截取照片
    输出: 车牌号与置信度
    """
    try:
        # 确保输入是numpy数组
        if isinstance(image, np.ndarray):
            result = ocr.ocr(image)
        else:
            result = ocr.ocr(np.array(image))

        # 处理新版本格式: [{ 'rec_texts': ['皖AD15208'], 'rec_scores': [0.995...], ... }]
        if isinstance(result, list) and len(result) > 0:
            item = result[0]
            if isinstance(item, dict) and 'rec_texts' in item:
                texts = item.get('rec_texts', [])
                if texts:
                    license_name = texts[0]
                    conf = item.get('rec_scores', [0.0])[0] if 'rec_scores' in item else 0.0
                    if '·' in license_name:
                        license_name = license_name.replace('·', '')
                    return license_name, conf

        # 处理字典格式
        if isinstance(result, dict) and 'rec_texts' in result:
            texts = result.get('rec_texts', [])
            if texts:
                license_name = texts[0]
                conf = result.get('rec_scores', [0.0])[0] if 'rec_scores' in result else 0.0
                if '·' in license_name:
                    license_name = license_name.replace('·', '')
                return license_name, conf

        # 处理旧版本列表格式
        if isinstance(result, list) and result[0]:
            item = result[0]
            if isinstance(item, list) and len(item) > 0:
                first_result = item[0]
                if isinstance(first_result, (list, tuple)) and len(first_result) >= 2:
                    text_conf = first_result[1] if isinstance(first_result[0], (list, tuple)) else first_result
                    license_name, conf = text_conf[0], text_conf[1]
                    if '·' in license_name:
                        license_name = license_name.replace('·', '')
                    return license_name, conf
        return None, None
    except Exception as e:
        print(f"OCR处理错误: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def initialize_models():
    """初始化模型，处理路径和错误"""
    # 检查模型文件是否存在
    model_path = r'E:\programingCodeFile\DeepLearninng\homework1222\result\yolov8n_carPlate\weights\best_all.pt'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"YOLO模型文件不存在: {model_path}")

    # 检查Paddle模型目录是否包含完整的配置文件
    cls_model_dir = 'paddleModels/whl/cls/ch_ppocr_mobile_v2.0_cls_infer'
    rec_model_dir = 'paddleModels/whl/rec/ch/ch_PP-OCRv4_rec_infer'

    # 检查是否有 inference.yml 配置文件（PaddleOCR 新版本需要）
    cls_has_config = os.path.exists(os.path.join(cls_model_dir, 'inference.yml'))
    rec_has_config = os.path.exists(os.path.join(rec_model_dir, 'inference.yml'))

    # 如果缺少配置文件，使用默认模型（会自动下载完整模型）
    if not cls_has_config:
        print(f"警告: 分类模型缺少inference.yml配置文件，使用默认模型")
        cls_model_dir = None
    if not rec_has_config:
        print(f"警告: 识别模型缺少inference.yml配置文件，使用默认模型")
        rec_model_dir = None

    # 加载OCR模型 - 使用更稳定的参数
    # 注意：为了兼容现有的模型文件（缺少inference.yml），我们使用旧的参数名
    # 虽然会有DeprecationWarning，但能正常加载.pdmodel文件
    ocr = PaddleOCR(
        use_angle_cls=False, # Reverted to old name to support legacy models
        lang="ch",
        # det=False,  # Removed as it causes error in init
        cls_model_dir=cls_model_dir, # Reverted to old name
        rec_model_dir=rec_model_dir, # Reverted to old name
        # show_log=False,  # Removed as it causes error in init
        device="gpu",  # 启用GPU加速
        # enable_mkldnn=False  # Removed as it causes error in init
    )

    # 加载YOLO模型
    model = YOLO(model_path, task='detect')
    # 显式指定使用GPU
    model.to('cuda:0')

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


def process_image_content(image, ocr, model, fontC):
    """
    处理图像内容：检测 -> 识别 -> 绘制
    返回: (处理后的图像, 识别结果列表)
    """
    license_results = []
    try:
        # 使用线程锁保护模型调用
        with model_lock:
            # YOLOv8检测车牌
            results = model(image, conf=0.25, iou=0.7, device='cuda:0')[0] # 使用gpu进行识别

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
            for i, each_img in enumerate(license_imgs):
                # 使用线程锁保护OCR调用
                with model_lock:
                    license_num, conf = get_license_result(ocr, each_img)
                if license_num and license_num.strip():
                    license_results.append(f"{license_num} ({conf:.2f})")
                else:
                    license_results.append('无法识别')

            # 在原图上绘制结果
            # 注意：这里license_results包含置信度字符串，绘制时可能只需要车牌号
            # 这里直接绘制完整字符串，或者只绘制车牌号
            draw_texts = [res.split(' ')[0] if '(' in res else res for res in license_results]

            for text, box in zip(draw_texts, location_list):
                image = tools.drawRectBox(image, box, text, fontC)

        return image, license_results

    except Exception as e:
        print(f"处理图片内容时出错: {e}")
        import traceback
        traceback.print_exc()
        return image, []


def process_single_image(image_path, ocr, model, fontC):
    """处理单张图片"""
    # 读取图片
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"错误: 无法读取图片: {image_path}")
        return None

    print(f"正在处理图片: {image_path}")
    
    processed_image, results = process_image_content(image, ocr, model, fontC)
    
    if not results:
        print("未检测到车牌或识别失败")
    else:
        for res in results:
            print(f"识别结果: {res}")

    # 显示结果
    display_image = cv2.resize(processed_image, dsize=None, fx=0.8, fy=0.8, interpolation=cv2.INTER_LINEAR)
    cv2.imshow(f"车牌检测结果 - {image_path.name}", display_image)
    cv2.waitKey(0)  # 等待按键
    cv2.destroyAllWindows()

    # 保存结果图片
    output_path = image_path.parent / f"result_{image_path.name}"
    cv2.imwrite(str(output_path), processed_image)
    print(f"结果已保存到: {output_path}")

    return results


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