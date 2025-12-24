from ultralytics import YOLO
import torch

"""
训练模型
"""

def main():
    # 检查是否有可用的CUDA设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

# 1. 加载预训练模型
#  yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt

    model = YOLO('yolov8n.pt')  # 加载预训练的 YOLO11n 模型

# 2. 指定数据集配置文件

    data_yaml_path = 'dataset.yaml' # 相对于此脚本的路径，或者使用绝对路径


    epochs = 300
    batch_size = 16
    img_size = (720, 1160) # 输入图片的尺寸
    project_name = 'result' # 训练结果保存的项目名称
    run_name = 'yolov8n_carPlate' # 本次训练的名称

# 4. 开始训练
    try:
        print(f"Starting training with model: yolov8n.pt")
        print(f"Dataset YAML: {data_yaml_path}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Image size: {img_size}")
        print(f"Device: {device}")

        results = model.train(
            data=data_yaml_path,
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            device=device,        # 使用 'cuda' 或 'cpu'，或者直接指定GPU索引如 '0', '0,1'
            project=project_name, # 结果将保存在 runs/detect/HelmetWorkwearDetection 目录下
            name=run_name,        # 本次运行的子目录名
            patience=15,          # 早停轮数，如果20轮内验证集损失没有改善则停止训练


            
            optimizer='AdamW',    # 优化器 'SGD', 'Adam', 'AdamW', 'NAdam', 'RAdam', 'RMSProp'
            workers=8,          # 数据加载的worker数量，根据CPU核心数调整
            cos_lr=True,          # 使用余弦退火学习率调度,
            lr0=0.001,               #初始学习率
            lrf=0.01,  # 最终学习率 (lr0 * lrf)
            amp=True,            # 启用自动混合精度(AMP) 训练，可减少内存使用量并加快训练速度，同时将对精度的影响降至最低。
            # hsv_h=0.025,
            degrees=5,  # 旋转角度
            translate=0.1,  # 平移范围
            scale=0.3,  # 缩放范围 (original intent was [0.5, 1.5], this sets a fixed scale or a range depending on YOLO version, check docs if behavior is not as expected)
            shear=2.0,  # 剪切强度
            flipud=0.0,  # 上下翻转概率
            fliplr=0.5,   # 左右翻转概率

            # 颜色增强 - 应对不同光照条件
            hsv_h=0.015,  # 色调变化 - 模拟不同灯光
            hsv_s=0.7,  # 饱和度变化 - 食物颜色差异
            hsv_v=0.4,  # 亮度变化 - 不同光照强度


            # val=True,           # 是否在训练过程中进行验证 (默认为True)
            # resume=False,       # 是否从上一个断点继续训练 (例如 'runs/detect/trainX/weights/last.pt')
            exist_ok=False,     # 如果项目/名称已存在，是否覆盖 (默认为False，会创建新目录如 train2, train3)
        )

        print("Training completed.")
        print(f"Results saved to: {results.save_dir}")
        print(f"Best model saved at: {results.save_dir}/weights/best.pt")

    except Exception as e:
        print(f"An error occurred during training: {e}")

#5. (可选) 评估模型
    print("Evaluating model...")
    metrics = model.val() # 在验证集上评估最佳模型
    print(f"Validation metrics: {metrics}")

# 6.  导出模型为其他格式 (例如 ONNX)
# print("Exporting model to ONNX format...")
# model.export(format='onnx') # 导出为 ONNX 格式，文件将保存在 best.onnx
# print("Model exported to ONNX.")

#    - 图片文件夹 (train/val)
#    - 标签文件夹 (train/val)，每个图片对应一个txt文件，每行格式: <class_id> <x_center_norm> <y_center_norm> <width_norm> <height_norm>
# 最佳模型通常保存在 runs/detect/<project_name>/<run_name>/weights/best_all.pt。
#  使用 TensorBoard 查看训练过程中的指标: tensorboard --logdir runs/detect/<project_name>/<run_name>


if __name__ == '__main__':
    main()
