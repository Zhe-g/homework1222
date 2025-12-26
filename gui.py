import sys
import cv2
import os
import time
import numpy as np
import warnings

# 🔥 🔥 🔥 必须在导入任何库之前设置环境变量，避免冲突
os.environ['PADDLE_DISABLE_ONE_DNN'] = '1'
os.environ['FLAGS_use_mkldnn'] = '0'
os.environ['CPU_NUM'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

# 禁用警告
warnings.filterwarnings('ignore')

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QTextEdit,
                             QFileDialog, QMessageBox, QGroupBox, QSplitter)
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal

# 导入后端逻辑
import run as backend

class ModelInitThread(QThread):
    """模型初始化线程"""
    finished = pyqtSignal(object, object, object) # ocr, model, font
    error = pyqtSignal(str)

    def run(self):
        try:
            ocr, model = backend.initialize_models()
            fontC = backend.load_font()
            self.finished.emit(ocr, model, fontC)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))

class VideoThread(QThread):
    """视频处理线程"""
    change_pixmap_signal = pyqtSignal(QImage)
    update_result_signal = pyqtSignal(list)
    video_finished = pyqtSignal()  # 视频结束信号

    def __init__(self, ocr, model, fontC):
        super().__init__()
        self.ocr = ocr
        self.model = model
        self.fontC = fontC
        self.source = None
        self._run_flag = False
        self._finished_emitted = False  # 防止重复发送信号

    def set_source(self, source):
        self.source = source

    def run(self):
        self._run_flag = True
        self._finished_emitted = False
        cap = None
        try:
            cap = cv2.VideoCapture(self.source)

            # 尝试设置摄像头分辨率
            if isinstance(self.source, int):
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

            while self._run_flag:
                ret, frame = cap.read()
                if ret:
                    try:
                        # 处理帧
                        start_time = time.time()
                        processed_img, results = backend.process_image_content(frame, self.ocr, self.model, self.fontC)
                        fps = 1.0 / (time.time() - start_time)

                        # 在图像上绘制FPS
                        cv2.putText(processed_img, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                        if results:
                            self.update_result_signal.emit(results)

                        # 转换为QImage - 使用copy确保数据独立
                        rgb_image = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB).copy()
                        h, w, ch = rgb_image.shape
                        bytes_per_line = ch * w
                        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                        self.change_pixmap_signal.emit(convert_to_Qt_format)
                    except Exception as e:
                        print(f"处理帧时出错: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
                else:
                    break
        except Exception as e:
            print(f"视频线程错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if cap is not None:
                cap.release()
            # 发送视频结束信号（只发送一次）
            if not self._finished_emitted:
                self._finished_emitted = True
                self.video_finished.emit()

    def stop(self):
        """停止线程"""
        self._run_flag = False
        # 使用超时等待，避免死锁
        if self.isRunning():
            # 非阻塞方式，最多等待3秒
            self.wait(3000)

class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.title = '车牌识别系统'
        self.left = 100
        self.top = 100
        self.width = 1200
        self.height = 800
        self.initUI()

        self.ocr = None
        self.model = None
        self.fontC = None
        self.video_thread = None
        self._is_running = False  # 跟踪视频是否正在运行

        # 启动模型加载
        self.status_label.setText("正在加载模型，请稍候...")
        self.disable_buttons()
        self.init_thread = ModelInitThread()
        self.init_thread.finished.connect(self.on_models_loaded)
        self.init_thread.error.connect(self.on_model_error)
        self.init_thread.start()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        # 主窗口部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 布局
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)

        # 左侧控制栏
        control_panel = QGroupBox("控制面板")
        control_layout = QVBoxLayout()
        control_panel.setLayout(control_layout)
        control_panel.setFixedWidth(250)

        self.btn_image = QPushButton('打开图片')
        self.btn_image.clicked.connect(self.open_image)
        self.btn_image.setMinimumHeight(40)
        
        self.btn_video = QPushButton('打开视频')
        self.btn_video.clicked.connect(self.open_video)
        self.btn_video.setMinimumHeight(40)
        
        self.btn_camera = QPushButton('打开摄像头')
        self.btn_camera.clicked.connect(self.open_camera)
        self.btn_camera.setMinimumHeight(40)
        
        self.btn_stop = QPushButton('停止视频/摄像头')
        self.btn_stop.clicked.connect(self.stop_video)
        self.btn_stop.setMinimumHeight(40)
        self.btn_stop.setEnabled(False)

        control_layout.addWidget(self.btn_image)
        control_layout.addWidget(self.btn_video)
        control_layout.addWidget(self.btn_camera)
        control_layout.addWidget(self.btn_stop)
        control_layout.addStretch()
        
        self.status_label = QLabel("准备就绪")
        self.status_label.setWordWrap(True)
        control_layout.addWidget(self.status_label)

        # 中间显示区域
        display_panel = QGroupBox("实时画面")
        display_layout = QVBoxLayout()
        display_panel.setLayout(display_layout)
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setText("等待图像输入...")
        self.image_label.setStyleSheet("background-color: #f0f0f0;")
        display_layout.addWidget(self.image_label)

        # 右侧结果栏
        result_panel = QGroupBox("识别结果")
        result_layout = QVBoxLayout()
        result_panel.setLayout(result_layout)
        result_panel.setFixedWidth(250)
        
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        result_layout.addWidget(self.result_text)

        # 添加到主布局
        main_layout.addWidget(control_panel)
        main_layout.addWidget(display_panel, 1) # stretch factor 1
        main_layout.addWidget(result_panel)

    def disable_buttons(self):
        self.btn_image.setEnabled(False)
        self.btn_video.setEnabled(False)
        self.btn_camera.setEnabled(False)

    def enable_buttons(self):
        self.btn_image.setEnabled(True)
        self.btn_video.setEnabled(True)
        self.btn_camera.setEnabled(True)

    def on_models_loaded(self, ocr, model, fontC):
        self.ocr = ocr
        self.model = model
        self.fontC = fontC
        self.status_label.setText("模型加载完成！")
        self.enable_buttons()

    def on_model_error(self, error_msg):
        self.status_label.setText(f"模型加载失败:\n{error_msg}")
        QMessageBox.critical(self, "错误", f"模型加载失败: {error_msg}")

    def update_image(self, qt_img):
        # 调整图片大小以适应标签，保持比例
        scaled_pixmap = QPixmap.fromImage(qt_img).scaled(
            self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)

    def update_result(self, results):
        current_text = self.result_text.toPlainText()
        new_text = "\n".join(results)
        # 简单去重，避免刷屏太快
        if new_text not in current_text: # 简单的逻辑，实际可能需要更好的日志管理
            self.result_text.append(f"[{time.strftime('%H:%M:%S')}]")
            self.result_text.append(new_text)
            self.result_text.append("-" * 20)
            # 滚动到底部
            self.result_text.moveCursor(self.result_text.textCursor().End)

    def stop_video(self):
        """停止视频线程"""
        if self.video_thread is not None:
            print("正在停止视频线程...")
            self._is_running = False
            self.video_thread.stop()

            # 断开所有信号连接
            try:
                if self.video_thread:
                    try:
                        self.video_thread.change_pixmap_signal.disconnect()
                    except:
                        pass
                    try:
                        self.video_thread.update_result_signal.disconnect()
                    except:
                        pass
                    try:
                        self.video_thread.video_finished.disconnect()
                    except:
                        pass
            except:
                pass

            # 删除线程对象
            self.video_thread = None
            self.btn_stop.setEnabled(False)
            self.enable_buttons()
            self.status_label.setText("视频已停止")

    def on_video_finished(self):
        """视频播放完成时的回调"""
        self._is_running = False
        self.btn_stop.setEnabled(False)
        self.enable_buttons()
        self.status_label.setText("视频播放完成")

    def open_image(self):
        self.stop_video()
        fname, _ = QFileDialog.getOpenFileName(self, '选择图片', 'e:\\programingCodeFile\\DeepLearninng\\homework1222', "Image files (*.jpg *.gif *.png *.jpeg)")
        if fname:
            self.status_label.setText(f"正在处理: {os.path.basename(fname)}")
            try:
                # 读取并处理
                image = cv2.imread(fname)
                if image is not None:
                    processed_img, results = backend.process_image_content(image, self.ocr, self.model, self.fontC)

                    # 显示结果 - 使用copy确保数据独立
                    rgb_image = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB).copy()
                    h, w, ch = rgb_image.shape
                    bytes_per_line = ch * w
                    qt_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    self.update_image(qt_img)

                    # 更新文本
                    self.result_text.append(f"图片: {os.path.basename(fname)}")
                    if results:
                        self.result_text.append("\n".join(results))
                    else:
                        self.result_text.append("未检测到车牌")
                    self.result_text.append("-" * 20)

                    self.status_label.setText("处理完成")
                else:
                    self.status_label.setText("无法读取图片")
            except Exception as e:
                print(f"处理图片时出错: {e}")
                import traceback
                traceback.print_exc()
                self.status_label.setText(f"处理图片时出错: {e}")

    def open_video(self):
        """打开视频文件"""
        self.stop_video()
        # 稍微延迟，确保旧线程完全停止
        QApplication.processEvents()

        fname, _ = QFileDialog.getOpenFileName(self, '选择视频', 'e:\\programingCodeFile\\DeepLearninng\\homework1222', "Video files (*.mp4 *.avi *.mkv)")
        if fname:
            # 创建新的视频线程
            self.video_thread = VideoThread(self.ocr, self.model, self.fontC)
            self.video_thread.change_pixmap_signal.connect(self.update_image)
            self.video_thread.update_result_signal.connect(self.update_result)
            self.video_thread.video_finished.connect(self.on_video_finished)

            self.video_thread.set_source(fname)
            self._is_running = True
            self.video_thread.start()
            self.disable_buttons()
            self.btn_stop.setEnabled(True)
            self.status_label.setText(f"正在播放视频: {os.path.basename(fname)}")

    def open_camera(self):
        """打开摄像头"""
        self.stop_video()
        # 稍微延迟，确保旧线程完全停止
        QApplication.processEvents()

        # 创建新的视频线程
        self.video_thread = VideoThread(self.ocr, self.model, self.fontC)
        self.video_thread.change_pixmap_signal.connect(self.update_image)
        self.video_thread.update_result_signal.connect(self.update_result)
        self.video_thread.video_finished.connect(self.on_video_finished)

        # 默认使用摄像头0
        self.video_thread.set_source(0)
        self._is_running = True
        self.video_thread.start()
        self.disable_buttons()
        self.btn_stop.setEnabled(True)
        self.status_label.setText("正在使用摄像头")

    def closeEvent(self, event):
        """窗口关闭事件"""
        print("正在关闭窗口...")
        self.stop_video()
        # 等待一段时间确保线程完全停止
        import time
        time.sleep(0.5)
        event.accept()

if __name__ == '__main__':
    # 适配高DPI屏幕
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    ex = App()
    ex.show()
    sys.exit(app.exec_())
