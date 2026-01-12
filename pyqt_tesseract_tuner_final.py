import sys
import cv2
import pytesseract
import re
import numpy as np
import time 
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, 
    QHBoxLayout, QPushButton, QLabel, QLineEdit, QFormLayout,
    QGroupBox, QSlider, QCheckBox, QComboBox 
)
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QTimer, pyqtSlot, QRect
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen

# --- Tesseract-OCRのパス設定 ---
try:
    pytesseract.pytesseract.tesseract_cmd = r'D:\OCR\tesseract.exe'
    # 利用可能なOEMモードとPSMモードを確認
    # print("Tesseract Info:", pytesseract.get_tesseract_version(), pytesseract.get_languages(config='')) 
except Exception as e:
    print(f"エラー: Tesseract-OCRが見つかりません。パス指定 (tesseract_cmd) を確認してください。")
    sys.exit(-1)

# (CameraThread, CameraLabel は変更なし)
class CameraThread(QThread):
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

class MainWindow(QMainWindow):
    """
    Tesseract微調整用GUI (OEM + Morphology)
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQt Tesseract Fine Tuner (OEM + Morphology)")
        self.setGeometry(100, 100, 1000, 700) 

        self.current_cv_frame = None 
        self.processed_ocr_frame = None 
        self.morph_kernel = np.ones((3,3), np.uint8)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(); central_widget.setLayout(main_layout)
        left_layout = QVBoxLayout(); main_layout.addLayout(left_layout, 7) 
        self.camera_label = CameraLabel("カメラ待機中...")
        self.camera_label.mouse_pos_signal.connect(self.update_camera_coords)
        left_layout.addWidget(self.camera_label, 3) 
        debug_layout = QHBoxLayout()
        self.gray_debug_label = QLabel("Grayscale")
        self.gray_debug_label.setAlignment(Qt.AlignmentFlag.AlignCenter); self.gray_debug_label.setStyleSheet("background-color: #333; color: white;")
        self.binary_debug_label = QLabel("Binary (Final)")
        self.binary_debug_label.setAlignment(Qt.AlignmentFlag.AlignCenter); self.binary_debug_label.setStyleSheet("background-color: #333; color: white;")
        debug_layout.addWidget(self.gray_debug_label); debug_layout.addWidget(self.binary_debug_label)
        left_layout.addLayout(debug_layout, 1) 

        right_layout = QVBoxLayout(); main_layout.addLayout(right_layout, 3) 
        
        ocr_group = QGroupBox("1. OCR設定 (ROI)"); ocr_layout = QFormLayout()
        self.cam_coord_label = QLabel("---"); self.cam_coord_label.setStyleSheet("font-weight: bold; color: green;")
        ocr_layout.addRow("カメラ内 座標:", self.cam_coord_label)
        self.roi_x1_input = QLineEdit("100"); self.roi_y1_input = QLineEdit("200")
        self.roi_x2_input = QLineEdit("600"); self.roi_y2_input = QLineEdit("250")
        roi_layout1 = QHBoxLayout(); roi_layout1.addWidget(self.roi_x1_input); roi_layout1.addWidget(self.roi_y1_input)
        ocr_layout.addRow("ROI 左上 (X1, Y1):", roi_layout1)
        roi_layout2 = QHBoxLayout(); roi_layout2.addWidget(self.roi_x2_input); roi_layout2.addWidget(self.roi_y2_input)
        ocr_layout.addRow("ROI 右下 (X2, Y2):", roi_layout2)
        ocr_group.setLayout(ocr_layout); right_layout.addWidget(ocr_group)
        
        preprocess_group = QGroupBox("2. 二値化設定 (手動)"); preprocess_layout = QFormLayout()
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal); self.threshold_slider.setRange(0, 255); self.threshold_slider.setValue(140) 
        self.threshold_slider_label = QLabel("140"); self.threshold_slider.valueChanged.connect(lambda val: self.threshold_slider_label.setText(str(val)))
        slider_layout = QHBoxLayout(); slider_layout.addWidget(self.threshold_slider); slider_layout.addWidget(self.threshold_slider_label)
        preprocess_layout.addRow("二値化しきい値:", slider_layout)
        self.invert_checkbox = QCheckBox("白黒反転 (黒背景/白文字)"); self.invert_checkbox.setChecked(True) 
        preprocess_layout.addRow(self.invert_checkbox)
        preprocess_group.setLayout(preprocess_layout); right_layout.addWidget(preprocess_group)
        
        advanced_group = QGroupBox("3. 高度な前処理"); advanced_layout = QFormLayout()
        self.resize_combo = QComboBox(); self.resize_combo.addItems(["なし (100%)", "200% (Linear 高速)", "200% (Cubic 高画質)"]); self.resize_combo.setCurrentIndex(2) 
        advanced_layout.addRow("アップスケール:", self.resize_combo)
        # モルフォロジーの選択肢を変更
        self.morph_combo = QComboBox(); self.morph_combo.addItems(["なし", "収縮 (Erode)", "膨張 (Dilate)", "開放 (Open)", "閉鎖 (Close)"]); self.morph_combo.setCurrentIndex(1) 
        advanced_layout.addRow("モルフォロジー:", self.morph_combo)
        advanced_group.setLayout(advanced_layout); right_layout.addWidget(advanced_group)

        # ▼▼▼ 追加: Tesseract設定 ▼▼▼
        tesseract_group = QGroupBox("4. Tesseract設定")
        tesseract_layout = QFormLayout()
        self.oem_combo = QComboBox()
        # OEMモード: 0=Legacy, 1=LSTM, 2=Legacy+LSTM, 3=Default(LSTM if available)
        self.oem_combo.addItems(["3 (Default)", "1 (LSTM only)", "0 (Legacy only)", "2 (Legacy + LSTM)"])
        self.oem_combo.setToolTip("OCR Engine Mode. LSTM (1 or 3) is generally better.")
        tesseract_layout.addRow("Engine Mode (OEM):", self.oem_combo)
        # PSMは 7 (1行) で固定
        tesseract_group.setLayout(tesseract_layout)
        right_layout.addWidget(tesseract_group)
        # ▲▲▲ 追加 ▲▲▲

        self.start_button = QPushButton("OCR実行 (テスト)")
        self.start_button.setStyleSheet("font-size: 18px; background-color: #007BFF; color: white; padding: 10px;")
        self.start_button.clicked.connect(self.run_ocr_test) 
        right_layout.addWidget(self.start_button)
        self.ocr_result_label = QLabel("OCR結果: ---")
        self.ocr_result_label.setStyleSheet("font-size: 16px; background-color: #EEE; padding: 5px;")
        self.ocr_result_label.setWordWrap(True) 
        right_layout.addWidget(self.ocr_result_label)
        right_layout.addStretch() 

        self.init_threads()
        QTimer.singleShot(100, self.update_roi_drawing) 
        self.roi_x1_input.textChanged.connect(self.update_roi_and_preprocess)
        self.roi_y1_input.textChanged.connect(self.update_roi_and_preprocess)
        self.roi_x2_input.textChanged.connect(self.update_roi_and_preprocess)
        self.roi_y2_input.textChanged.connect(self.update_roi_and_preprocess)
        self.threshold_slider.valueChanged.connect(self.update_preprocess_image) 
        self.invert_checkbox.stateChanged.connect(self.update_preprocess_image)
        self.resize_combo.currentIndexChanged.connect(self.update_preprocess_image) 
        self.morph_combo.currentIndexChanged.connect(self.update_preprocess_image) 
        # OEMの変更はOCR実行時にのみ影響

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
        self.update_preprocess_image() # リアルタイムで前処理画像は更新

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

    # --- リアルタイム前処理 (変更なし) ---
    def update_preprocess_image(self):
        if self.current_cv_frame is None: return
        try:
            x1 = int(self.roi_x1_input.text()); y1 = int(self.roi_y1_input.text())
            x2 = int(self.roi_x2_input.text()); y2 = int(self.roi_y2_input.text())
            if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1: raise ValueError("ROI座標が不正です")
            
            roi_frame = self.current_cv_frame[y1:y2, x1:x2]
            gray_frame = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
            self.display_debug_image(gray_frame, self.gray_debug_label) 

            resize_idx = self.resize_combo.currentIndex()
            if resize_idx == 1: gray_frame = cv2.resize(gray_frame, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
            elif resize_idx == 2: gray_frame = cv2.resize(gray_frame, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            
            threshold_val = self.threshold_slider.value()
            threshold_type = cv2.THRESH_BINARY if self.invert_checkbox.isChecked() else cv2.THRESH_BINARY_INV
            _, binary_frame = cv2.threshold(gray_frame, threshold_val, 255, threshold_type)
            
            morph_idx = self.morph_combo.currentIndex()
            if morph_idx == 1: binary_frame = cv2.erode(binary_frame, self.morph_kernel, iterations=1)
            elif morph_idx == 2: binary_frame = cv2.dilate(binary_frame, self.morph_kernel, iterations=1)
            elif morph_idx == 3: binary_frame = cv2.morphologyEx(binary_frame, cv2.MORPH_OPEN, self.morph_kernel)
            elif morph_idx == 4: binary_frame = cv2.morphologyEx(binary_frame, cv2.MORPH_CLOSE, self.morph_kernel)
            
            self.processed_ocr_frame = binary_frame
            self.display_debug_image(binary_frame, self.binary_debug_label)

        except (ValueError, cv2.error):
            self.gray_debug_label.setText("Grayscale (ROI invalid)")
            self.binary_debug_label.setText("Binary (Final)")
            self.processed_ocr_frame = None

    def display_debug_image(self, cv_img, label: QLabel):
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
    # --- リアルタイム前処理 (ここまで) ---

    # ▼▼▼ 修正: run_ocr_test (OEM設定を使用) ▼▼▼
    def run_ocr_test(self):
        """「OCR実行 (テスト)」ボタンが押されたときの処理"""
        print(f"\n--- Tesseract OCR テスト開始 ---")
        
        if self.processed_ocr_frame is None:
            print("エラー: 処理対象の画像がありません (ROIを確認してください)")
            self.ocr_result_label.setText("OCR結果: エラー (ROIが不正)")
            return

        frame_to_ocr = self.processed_ocr_frame
        
        # --- OEM設定を取得 ---
        oem_index = self.oem_combo.currentIndex()
        # Combo BoxのIndexに対応するOEM値 (0=Legacy, 1=LSTM, 2=Legacy+LSTM, 3=Default)
        oem_map = {0: 3, 1: 1, 2: 0, 3: 2} 
        oem_value = oem_map.get(oem_index, 3) # 不明な場合はDefault (3)
        
        # --- Tesseract設定 (PSM 7 + OEM) ---
        config_psm7 = f'--psm 7 --oem {oem_value} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        print(f"  Tesseract Config: {config_psm7}")

        # --- OCR実行 ---
        try:
            start_time = time.time()
            raw_text = pytesseract.image_to_string(frame_to_ocr, lang='eng', config=config_psm7)
            end_time = time.time()
            print(f"  OCR実行時間: {end_time - start_time:.3f} 秒")
        except Exception as e:
            self.ocr_result_label.setText(f"OCR結果: エラー (Tesseract実行失敗: {e})"); print(f"Pytesseractエラー: {e}"); return

        # --- 整形 & 16桁チェック ---
        processed_code = re.sub(r'[^A-Z0-9]', '', raw_text.upper()) 
        
        if len(processed_code) != 16:
            final_code = processed_code[:16] # とりあえず16桁に切り詰める
            result_text = f"RAW: {raw_text.strip()}\nFinal: {final_code}\n<font color='red'><b>警告: {len(processed_code)}桁です (16桁ではありません)</b></font>"
            print(f"エラー: 16桁ではありません (認識: '{processed_code}')")
        else:
             final_code = processed_code
             result_text = f"RAW: {raw_text.strip()}\nFinal (16-digit): {final_code}"
             print(f"認識結果 (16桁): '{final_code}'")

        # --- GUIに結果を表示 ---
        self.ocr_result_label.setText(result_text)
        print("--- テスト完了 ---")
    # ▲▲▲ 修正 ▲▲▲
        
    def closeEvent(self, event):
        self.camera_thread.stop()
        event.accept()

# --- プログラム実行 ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())