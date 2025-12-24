#  homework1222

--------------------
## 深度学习期末大作业
*注：当前仅支持新能源绿色车牌*

### 硬件环境
- 显卡：NVIDIA GeForce RTX 4060 laptop
- cpu：Intel(R) Core(TM) i7-12700H

### 软件环境
- OS：win11
- cuda版本：12.6
- cudnn版本：8.9.7

其余见requirements.txt

###  过程
使用yolov8n模型识别车牌，将识别的车牌部分截出并使用paddle模型对截图中的车牌号进行识别。
模型数据集来自CCPD2020.

## 运行
使用conda创建环境并安装依赖，运行gui.py即可。
