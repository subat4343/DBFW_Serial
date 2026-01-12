import sys
import cv2
import pytesseract
import re
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, 
    QHBoxLayout, QPushButton, QLabel, QLineEdit, QFormLayout,
    QGroupBox, QSlider, QCheckBox # QCheckBox を追加
)
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QTimer, pyqtSlot, QRect
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen

# --- Tesseract-OCRのパス設定 ---
try:
    pytesseract.pytesseract.tesseract_cmd = r'D:\OCR\tesseract.exe'
    pytesseract.get_tesseract_version()
    print("Tesseract-OCR 連携OK.")
except Exception as e:
    print(f"エラー: Tesseract-OCRが見つかりません。パス指定 (tesseract_cmd) を確認してください。")
    print(f"指定されたパス: {pytesseract.pytesseract.tesseract_cmd}")
    print(f"詳細: {e}")
    sys.exit(-1)

# -----------------------------------------------
# (CameraThread, CameraLabel は変更なし)
# -----------------------------------------------

class CameraThread(QThread):
    """ (変更なし) """
    frame_signal = pyqtSignal(QImage, object) 
    def run(self):
        self.running = True; cap = cv2.VideoCapture(0)
        if not cap.isOpened(): print("エラー: カメラ 0 を開けません。"); return
        while self.running:
            ret, frame = cap.read()
            if ret:
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); h, w, ch = rgb_image.shape
                qt_image = QImage(rgb_image.data, w, h, ch * w, QImage.Format.Format_RGB888)
                self.frame_signal.emit(qt_image.copy(), frame)
            self.msleep(30)
        cap.release()
    def stop(self): self.running = False; self.wait()

class CameraLabel(QLabel):
    """ (変更なし) ROI描画とマウス座標特定機能"""
    mouse_pos_signal = pyqtSignal(int, int) 
    def __init__(self, text):
        super().__init__(text); self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background-color: black; color: white;"); self.pixmap_size = None
        self.image_origin = None; self.roi_rect = QRect(); self.setMouseTracking(True) 
    def setPixmap(self, pixmap):
        super().setPixmap(pixmap); self.pixmap_size = pixmap.size()
        if self.pixmap_size:
            margin_x = (self.width() - self.pixmap_size.width()) // 2
            margin_y = (self.height() - self.pixmap_size.height()) // 2
            self.image_origin = (margin_x, margin_y)
        else: self.image_origin = None
    def set_roi(self, x1, y1, x2, y2): self.roi_rect = QRect(x1, y1, x2 - x1, y2 - y1); self.update()
    def mouseMoveEvent(self, event):
        if self.image_origin:
            ox, oy = self.image_origin; local_x = event.pos().x() - ox; local_y = event.pos().y() - oy
            if 0 <= local_x < self.pixmap_size.width() and 0 <= local_y < self.pixmap_size.height():
                self.mouse_pos_signal.emit(local_x, local_y)
            else: self.mouse_pos_signal.emit(-1, -1)
    def paintEvent(self, event):
        super().paintEvent(event) 
        if self.image_origin and not self.roi_rect.isNull():
            painter = QPainter(self); ox, oy = self.image_origin
            draw_rect = self.roi_rect.translated(ox, oy)
            pen = QPen(Qt.GlobalColor.red, 2); painter.setPen(pen)
            painter.drawRect(draw_rect); painter.end()

# -----------------------------------------------
# ▼▼▼ MainWindow を修正 ▼▼▼
# -----------------------------------------------

class MainWindow(QMainWindow):
    """
    OCR調整用GUIウィンドウ (v4: Manual Slider + Invert Option)
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQt OCR Tuner (v4: Manual Slider + Invert)")
        self.setGeometry(100, 100, 1000, 700) 

        self.current_cv_frame = None 
        self.processed_ocr_frame = None 
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout() 
        central_widget.setLayout(main_layout)

        # --- 1. 左側: (変更なし) ---
        left_layout = QVBoxLayout()
        main_layout.addLayout(left_layout, 7) 
        self.camera_label = CameraLabel("カメラ待機中...")
        self.camera_label.mouse_pos_signal.connect(self.update_camera_coords)
        left_layout.addWidget(self.camera_label, 3) 
        debug_layout = QHBoxLayout()
        self.gray_debug_label = QLabel("Grayscale")
        self.gray_debug_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.gray_debug_label.setStyleSheet("background-color: #333; color: white;")
        self.binary_debug_label = QLabel("Binary (Manual)")
        self.binary_debug_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.binary_debug_label.setStyleSheet("background-color: #333; color: white;")
        debug_layout.addWidget(self.gray_debug_label)
        debug_layout.addWidget(self.binary_debug_label)
        left_layout.addLayout(debug_layout, 1) 

        # --- 2. 右側: 設定と操作 ---
        right_layout = QVBoxLayout() 
        main_layout.addLayout(right_layout, 3) 
        
        # 2a. OCR設定 (ROI)
        ocr_group = QGroupBox("1. OCR設定 (ROI)")
        ocr_layout = QFormLayout()
        self.cam_coord_label = QLabel("---")
        self.cam_coord_label.setStyleSheet("font-weight: bold; color: green;")
        ocr_layout.addRow("カメラ内 座標:", self.cam_coord_label)
        self.roi_x1_input = QLineEdit("100"); self.roi_y1_input = QLineEdit("200")
        self.roi_x2_input = QLineEdit("600"); self.roi_y2_input = QLineEdit("250")
        roi_layout1 = QHBoxLayout(); roi_layout1.addWidget(self.roi_x1_input); roi_layout1.addWidget(self.roi_y1_input)
        ocr_layout.addRow("ROI 左上 (X1, Y1):", roi_layout1)
        roi_layout2 = QHBoxLayout(); roi_layout2.addWidget(self.roi_x2_input); roi_layout2.addWidget(self.roi_y2_input)
        ocr_layout.addRow("ROI 右下 (X2, Y2):", roi_layout2)
        ocr_group.setLayout(ocr_layout)
        right_layout.addWidget(ocr_group)
        
        # ▼▼▼ 修正: 手動スライダー(QSlider)を復活 + QCheckBox追加 ▼▼▼
        preprocess_group = QGroupBox("2. 前処理設定 (手動)")
        preprocess_layout = QFormLayout()
        
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(0, 255)
        self.threshold_slider.setValue(135) # ユーザー指定のデフォルト値
        self.threshold_slider_label = QLabel("135")
        self.threshold_slider.valueChanged.connect(lambda val: self.threshold_slider_label.setText(str(val)))
        
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(self.threshold_slider)
        slider_layout.addWidget(self.threshold_slider_label)
        preprocess_layout.addRow("二値化しきい値:", slider_layout)
        
        self.invert_checkbox = QCheckBox("白黒反転 (黒背景/白文字)")
        self.invert_checkbox.setToolTip("チェックなし: 白背景/黒文字 (INV)\nチェックあり: 黒背景/白文字 (BINARY)")
        preprocess_layout.addRow(self.invert_checkbox)

        preprocess_group.setLayout(preprocess_layout)
        right_layout.addWidget(preprocess_group)
        # ▲▲▲ 修正 ▲▲▲

        # 2c. 実行ボタン
        self.start_button = QPushButton("OCR実行 (比較テスト)")
        self.start_button.setStyleSheet("font-size: 18px; background-color: #007BFF; color: white; padding: 10px;")
        self.start_button.clicked.connect(self.run_ocr_test) 
        right_layout.addWidget(self.start_button)

        # 2d. OCR結果
        self.ocr_result_label = QLabel("OCR結果: ---")
        self.ocr_result_label.setStyleSheet("font-size: 16px; background-color: #EEE; padding: 5px;")
        self.ocr_result_label.setWordWrap(True) 
        right_layout.addWidget(self.ocr_result_label)
        
        right_layout.addStretch() 

        # --- スレッド・シグナル接続 ---
        self.init_threads()
        QTimer.singleShot(100, self.update_roi_drawing) 
        self.roi_x1_input.textChanged.connect(self.update_roi_and_preprocess)
        self.roi_y1_input.textChanged.connect(self.update_roi_and_preprocess)
        self.roi_x2_input.textChanged.connect(self.update_roi_and_preprocess)
        self.roi_y2_input.textChanged.connect(self.update_roi_and_preprocess)
        # スライダーとチェックボックスが変更されたら画像処理を更新
        self.threshold_slider.valueChanged.connect(self.update_preprocess_image) 
        self.invert_checkbox.stateChanged.connect(self.update_preprocess_image)

    def init_threads(self):
        self.camera_thread = CameraThread()
        self.camera_thread.frame_signal.connect(self.update_camera_feed)
        self.camera_thread.start()

    @pyqtSlot(QImage, object)
    def update_camera_feed(self, qt_image, cv_frame):
        self.current_cv_frame = cv_frame 
        scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
            self.camera_label.width(), self.camera_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
        self.camera_label.setPixmap(scaled_pixmap)
        self.update_preprocess_image()

    @pyqtSlot(int, int)
    def update_camera_coords(self, x, y):
        if x == -1: self.cam_coord_label.setText("---")
        else: self.cam_coord_label.setText(f"({x}, {y})")
            
    def update_roi_and_preprocess(self):
        self.update_roi_drawing()
        self.update_preprocess_image()

    def update_roi_drawing(self):
        try:
            x1=int(self.roi_x1_input.text()); y1=int(self.roi_y1_input.text())
            x2=int(self.roi_x2_input.text()); y2=int(self.roi_y2_input.text())
            self.camera_label.set_roi(x1, y1, x2, y2)
        except ValueError: self.camera_label.set_roi(0, 0, 0, 0) 

    # ▼▼▼ 修正: update_preprocess_image (手動スライダー + 反転オプション) ▼▼▼
    def update_preprocess_image(self):
        """
        現在のROI、スライダー、反転チェックに基づき、画像前処理を行う
        """
        if self.current_cv_frame is None: return
        try:
            x1 = int(self.roi_x1_input.text()); y1 = int(self.roi_y1_input.text())
            x2 = int(self.roi_x2_input.text()); y2 = int(self.roi_y2_input.text())
            
            # 1. スライダーとしきい値タイプを取得
            threshold_val = self.threshold_slider.value()
            if self.invert_checkbox.isChecked():
                # 黒背景 / 白文字
                threshold_type = cv2.THRESH_BINARY
            else:
                # 白背景 / 黒文字 (デフォルト)
                threshold_type = cv2.THRESH_BINARY_INV
            
            if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1: raise ValueError("ROI座標が不正です")
            
            # 2. ROI切り抜き
            roi_frame = self.current_cv_frame[y1:y2, x1:x2]
            
            # 3. グレースケール化
            gray_frame = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
            
            # 4. 二値化 (手動しきい値 + 反転オプション)
            _, binary_frame = cv2.threshold(gray_frame, threshold_val, 255, threshold_type)
            
            # 5. OCR実行用に、処理後の画像(binary_frame)を保持
            self.processed_ocr_frame = binary_frame

            # 6. デバッグビューに表示
            self.display_debug_image(gray_frame, self.gray_debug_label)
            self.display_debug_image(binary_frame, self.binary_debug_label)

        except (ValueError, cv2.error):
            self.gray_debug_label.setText("Grayscale (ROI invalid)")
            self.binary_debug_label.setText("Binary (ROI invalid)")
            self.processed_ocr_frame = None
    # ▲▲▲ 修正 ▲▲▲

    def display_debug_image(self, cv_img, label: QLabel):
        """ (変更なし) OpenCV画像をQLabelに表示するヘルパー """
        try:
            if cv_img is None: label.setText("---"); return
            h, w = cv_img.shape
            if h == 0 or w == 0: label.setText("---"); return
            qt_image = QImage(cv_img.data, w, h, w, QImage.Format.Format_Grayscale8)
            scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
                label.width(), label.height(),
                Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
            )
            label.setPixmap(scaled_pixmap)
        except Exception as e:
            label.setText(f"Debug View Error: {e}")

    def run_ocr_test(self):
        """ (変更なし) 4ブロック分割ロジックで比較テスト """
        thresh_val = self.threshold_slider.value()
        invert_str = "反転あり" if self.invert_checkbox.isChecked() else "反転なし"
        print(f"\n--- OCR比較テスト開始 (しきい値: {thresh_val}, {invert_str}) ---")
        
        if self.processed_ocr_frame is None:
            print("エラー: 処理対象の画像がありません (ROIを確認してください)")
            self.ocr_result_label.setText("OCR結果: エラー (ROIが不正)")
            return

        frame_to_ocr = self.processed_ocr_frame
        results = []

        # --- 方法A: psm 7 (1行) で全体を認識 ---
        try:
            print("  [方法A] psm 7 (1行) を実行中...")
            config_psm7 = '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            raw_text_A = pytesseract.image_to_string(frame_to_ocr, lang='eng', config=config_psm7)
            processed_A = re.sub(r'[^A-Z0-9]', '', raw_text_A.upper())
            final_A = processed_A[:16] 
            results.append(f"方法A (1行 psm7):\n  {final_A}")
            print(f"    -> RAW: '{raw_text_A.strip()}', Final: '{final_A}'")
        except Exception as e:
            results.append(f"方法A (1行 psm7): エラー {e}")

        # --- 方法B: 4ブロックに分割し、psm 8 (1単語) で認識 ---
        try:
            print("  [方法B] 4ブロック分割 (psm 8) を実行中...")
            config_psm8 = '--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            
            h, w = frame_to_ocr.shape
            if w < 4: raise ValueError("ROI幅が小さすぎます")
            
            strip_w = w // 4
            strips = [
                frame_to_ocr[:, 0:strip_w],
                frame_to_ocr[:, strip_w:strip_w*2],
                frame_to_ocr[:, strip_w*2:strip_w*3],
                frame_to_ocr[:, strip_w*3:w]
            ]
            
            result_blocks = []
            for i, strip in enumerate(strips):
                raw_text_B = pytesseract.image_to_string(strip, lang='eng', config=config_psm8)
                processed_B = re.sub(r'[^A-Z0-9]', '', raw_text_B.upper())
                result_blocks.append(processed_B[:4]) # 各ブロック4桁に
            
            final_B = "".join(result_blocks)
            results.append(f"方法B (4ブロック psm8):\n  {final_B}")
            print(f"    -> Final: '{final_B}'")
            
        except Exception as e:
            results.append(f"方法B (4ブロック psm8): エラー {e}")

        self.ocr_result_label.setText("\n".join(results))
        print("--- 比較テスト完了 ---")
        
    def closeEvent(self, event):
        self.camera_thread.stop()
        event.accept()

# --- プログラム実行 ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())