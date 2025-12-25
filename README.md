# 车牌识别系统 (基于YOLOv8 + PaddleOCR)

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8.0-red.svg)](https://pytorch.org/)
[![PaddlePaddle](https://img.shields.io/badge/PaddlePaddle-3.0.0-orange.svg)](https://www.paddlepaddle.org.cn/)
[![License](https://img.shields.io/badge/License-Academic-green.svg)](LICENSE)

## 📋 项目简介

这是一个基于深度学习的车牌识别系统，使用**YOLOv8**进行车牌检测，结合**PaddleOCR**进行车牌字符识别，能够精准识别包括新能源车牌在内的多种类型车牌。系统支持GPU加速，推理速度达到实时处理水平（20-70 FPS）。

### ✨ 主要特性
- 🚀 **高性能识别**：支持GPU加速，推理速度提升23-67倍
- 📱 **多模式运行**：提供图形界面(GUI)和命令行两种运行方式
- 🔄 **实时处理**：支持视频流和图像文件的实时处理
- 🎯 **高精度识别**：基于CCPD2020数据集训练，识别准确率高
- 🔧 **易于部署**：提供完整的依赖管理，一键安装运行
- 📊 **可视化结果**：实时显示识别结果和置信度

---

## 🎯 深度学习期末大作业

### 硬件环境
- **显卡**: NVIDIA GeForce RTX 4060 Laptop
- **处理器**: Intel(R) Core(TM) i7-12700H
- **内存**: 建议16GB以上

### 软件环境
- **操作系统**: Windows 11
- **CUDA**: 12.6
- **cuDNN**: 8.9.7
- **Python**: 3.9+
- **Conda**: 推荐使用Anaconda或Miniconda

### 核心依赖库
详见 `requirements.txt`，主要包括：

| 依赖库 | 版本 | 用途 |
|--------|------|------|
| `torch` | 2.8.0+cu126 | PyTorch深度学习框架 |
| `ultralytics` | 8.3.133 | YOLOv8模型库 |
| `paddlepaddle-gpu` | 3.2.2 | PaddlePaddle GPU版本 |
| `paddleocr` | 3.0.2 | PaddleOCR文字识别 |
| `opencv-python` | 4.10.0.84 | 图像处理 |
| `PyQt5` | 5.15.11 | 图形界面 |
| `numpy` | 1.26.4 | 数值计算 |
| `pillow` | 11.0.0 | 图像处理库 |

---

## 🏗️ 技术架构

### 核心思路
采用**两级检测识别架构**：
1. **第一级 - 车牌检测**: 使用训练好的YOLOv8n模型在整幅图像中定位车牌区域
2. **第二级 - 字符识别**: 将检测到的车牌区域裁剪后，使用PaddleOCR识别具体车牌号码

### 数据集
模型训练数据集来自 **CCPD2020**（Chinese City Parking Dataset），包含超过30万张车牌图像，涵盖多种天气、光照、角度和车牌类型。

### 文件结构
```
homework1222/
├── run.py                      # 命令行模式主程序
├── gui.py                      # 图形化界面主程序
├── detect_tools.py             # 检测与绘制工具函数
├── Train.py                    # 模型训练脚本
├── test.py                     # 模型测试脚本
├── txtmaker.py                 # 标注文件（yolotxt）转换器
├── dataset.yaml                # 数据集配置文件
├── yolov8n.pt                  # YOLOv8预训练模型
├── best_all.pt                 # 训练好的最佳模型（全类型车牌）
│
├── paddleModels/               # PaddleOCR本地模型
│   └── whl/
│       ├── cls/ch_ppocr_mobile_v2.0_cls_infer/   # 文字方向分类模型
│       └── rec/ch_PP-OCRv4_rec_infer/            # 文字识别模型
│
├── result/                     # 训练结果与测试输出
│   ├── yolov8n_carPlate/       # 训练输出目录
│   │   └── weights/
│   │       └── best.pt         # 当前使用的模型文件
│   └── *.jpg, *.png            # 检测结果图片
│
├── Font/                       # 中文字体文件
│   └── platech.ttf             # 车牌专用字体
│
├── output/                     # 输出目录
├── requirements.txt            # GPU版依赖
├── requirements_cpu.txt        # CPU版依赖
└── README.md                   # 项目说明文档
```
- 注：CCPD2020数据集的标签是图片名，故需使用txtmaker.py转换为yolo的txt格式
---

## 🚀 快速开始

### 1. 环境准备

#### 创建Conda环境
```bash
# 创建Python 3.9环境
conda create -n carplate python=3.9

# 激活环境
conda activate carplate
```

#### 安装依赖（GPU版本 - 推荐）
```bash
# 方法一：直接安装requirements.txt（推荐）
pip install -r requirements.txt

# 方法二：分步安装（如遇到冲突）
# 1. 安装PyTorch (CUDA 12.6)
pip install torch==2.8.0+cu126 torchvision==0.23.0+cu126 -f https://download.pytorch.org/whl/torch_stable.html

# 2. 安装PaddlePaddle GPU版本
pip install paddlepaddle-gpu==3.2.2

# 3. 安装其他核心依赖
pip install ultralytics==8.3.133 paddleocr==3.0.2 opencv-python==4.10.0.84 PyQt5==5.15.11
```

#### 安装依赖（CPU版本）
如果没有GPU，使用CPU版本：
```bash
pip install -r requirements_cpu.txt
```

**性能对比**:
- **CPU版本**: 无车牌3FPS，有车牌0.3FPS
- **GPU版本**: 无车牌70FPS，有车牌20FPS

### 2. 模型准备

#### YOLO模型
- 默认使用 `result/yolov8n_carPlate/weights/best.pt`
- 识别所有类型车牌（含新能源），将 `best_all.pt` 复制到上述位置
- 如果只有 `yolov8n.pt`，可运行 `Train.py` 进行训练

#### PaddleOCR模型
- 首次运行会自动下载模型（约300MB）
- 或使用 `paddleModels/` 目录中的本地模型

### 3. 运行程序

#### 图形化界面模式（推荐）
```bash
python gui.py
```
- 支持图片上传、视频摄像头实时识别
- 实时显示识别结果和置信度

#### 命令行模式
```bash
python run.py
```
- 按提示输入图片路径
- 识别结果在终端输出并显示在弹窗

------------------------

## ⚙️ GPU更新

### ⚡ 性能对比 (RTX 4060)

| 场景 | CPU版本 | GPU版本 | 性能描述 |
|------|---------|---------|----------|
| 无车牌（仅YOLO检测） | ~3 FPS | ~70 FPS |  实时检测，流畅运行 |
| 有车牌（YOLO+OCR） | ~0.3 FPS | ~20 FPS| 实时识别，满足应用需求 |

### GPU配置说明

代码中已启用GPU加速，关键配置如下：

**run.py (第9行)** - CUDA设备选择:
```python
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 使用第一块GPU
```

**run.py (第119行)** - PaddleOCR GPU:
```python
ocr = PaddleOCR(
    use_angle_cls=False,
    lang="ch",
    cls_model_dir=cls_model_dir,
    rec_model_dir=rec_model_dir,
    device="gpu",  # 启用GPU加速
)
```

**run.py (第126、158行)** - YOLOv8 GPU:
```python
model.to('cuda:0')  # 模型加载到GPU
results = model(image, conf=0.25, iou=0.7, device='cuda:0')[0]  # GPU推理
```

### 多线程GPU冲突处理
由于PyTorch和PaddlePaddle同时使用GPU可能产生冲突，代码中采用以下保护措施：

```python
# 1. 环境变量限制线程
os.environ['PADDLE_DISABLE_ONE_DNN'] = '1'
os.environ['CPU_NUM'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

# 2. 使用线程锁保护模型调用
model_lock = threading.Lock()

# 3. 在模型推理时加锁
with model_lock:
    results = model(image, conf=0.25, iou=0.7, device='cuda:0')[0]
    license_num, conf = get_license_result(ocr, each_img)
```

---

## 📝 模型训练

如需自行训练YOLO模型：

```bash
python Train.py
```

训练配置在 `dataset.yaml` 中，可根据需要调整：
- 数据集路径
- 类别数量（单类别：车牌）
- 训练轮数、批次大小等超参数

---

## 🔧 常见问题

### 1. WinError 127错误
**现象**: 运行时报错 `WinError 127: 缺少相关依赖`

**解决方案**: 删除对应conda环境中的nvidia文件夹后重新运行
```bash
# 路径示例（根据实际路径调整）
rm -rf ~/anaconda3/envs/carplate/Lib/site-packages/nvidia
```

### 2. CUDA版本不匹配
**现象**: 报错 `CUDA out of memory` 或版本冲突

**解决方案**: 检查CUDA版本兼容性
```bash
nvcc --version  # 查看CUDA版本
```
PaddlePaddle 3.0.0 官方支持CUDA 11.2/11.8/12.0，你的CUDA 12.6可能需要降级。


### 3. 中文字体显示异常
**现象**: 识别结果文字显示为方框

**解决方案**: 确保 `Font/platech.ttf` 字体文件存在，或代码会自动回退到系统字体（黑体、微软雅黑）



---

## 🔄 Git分支管理

主要分支：

- **main分支**: GPU版本（当前默认）
- **cpu分支**: CPU版本（适用于无GPU环境）

------------

## 📄 许可证

本项目为深度学习课程期末大作业，仅供学习交流使用。

---

## 🙏 致谢

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - 目标检测框架
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) - 文字识别引擎
- [CCPD数据集](https://github.com/detectRecog/CCPD) - 车牌识别数据集