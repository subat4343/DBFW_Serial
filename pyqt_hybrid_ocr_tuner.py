import sys
import cv2
import pytesseract
import re
import numpy as np
import time

try:
    import easyocr
except ImportError:
    print("エラー: 'easyocr' が見つかりません。'pip install easyocr' を実行してください。")
    sys.exit(-1)
try:
    import torch
except ImportError:
    print("エラー: 'torch' (PyTorch) が見つかりません。'pip install torch torchvision' を実行してください。")
    sys.exit(-1)

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
    pytesseract.get_tesseract_version()
    print("Tesseract-OCR 連携OK.")
except Exception as e:
    print(f"エラー: Tesseract-OCRが見つかりません。パス指定 (tesseract_cmd) を確認してください。")
    sys.exit(-1)

# (CameraThread, CameraLabel, OcrThread は変更なし)
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

class OcrThread(QThread):
    result_signal = pyqtSignal(str, str, str) # 3つの方法の結果を別々に送信

    def __init__(self, e_reader):
        super().__init__()
        self.e_reader = e_reader
        self.frame_to_ocr = None
        self.running = True
        self.request_ocr = False

    def set_image(self, image):
        self.frame_to_ocr = image.copy() if image is not None else None
        self.request_ocr = True

    def run(self):
        while self.running:
            if self.request_ocr and self.frame_to_ocr is not None:
                self.request_ocr = False
                frame_ocr = self.frame_to_ocr
                final_A, final_B, final_C = "Error", "Error", "Error" # 初期値

                # --- 方法A: Tesseract (1行 psm7) ---
                try:
                    config_psm7 = '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                    raw_text_A = pytesseract.image_to_string(frame_ocr, lang='eng', config=config_psm7)
                    processed_A = re.sub(r'[^A-Z0-9]', '', raw_text_A.upper())
                    final_A = processed_A[:16]
                except Exception as e: final_A = f"Tess A Error: {e}"[:20]

                # --- 方法B: Tesseract (4ブロック psm8) ---
                try:
                    config_psm8 = '--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                    h, w = frame_ocr.shape
                    if w < 4: raise ValueError("ROI W<4")
                    strip_w = w // 4
                    strips = [ frame_ocr[:, 0:strip_w], frame_ocr[:, strip_w:strip_w*2],
                               frame_ocr[:, strip_w*2:strip_w*3], frame_ocr[:, strip_w*3:w] ]
                    result_blocks = []
                    for strip in strips:
                        raw_text_B = pytesseract.image_to_string(strip, lang='eng', config=config_psm8)
                        processed_B = re.sub(r'[^A-Z0-9]', '', raw_text_B.upper())
                        result_blocks.append(processed_B[:4])
                    final_B = "".join(result_blocks)
                except Exception as e: final_B = f"Tess B Error: {e}"[:20]

                # --- 方法C: EasyOCR (AI) ---
                try:
                    ocr_results_C = self.e_reader.readtext(frame_ocr, detail=0)
                    raw_text_C = " ".join(ocr_results_C)
                    processed_C = re.sub(r'[^A-Z0-9]', '', raw_text_C.upper())
                    final_C = processed_C[:16]
                except Exception as e: final_C = f"EasyOCR Error: {e}"[:20]

                self.result_signal.emit(final_A, final_B, final_C)
            else:
                self.msleep(50)

    def stop(self):
        self.running = False
        self.wait()

class MainWindow(QMainWindow):
    """
    リアルタイムOCR調整GUI (v8: Realtime Compare)
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQt Realtime Hybrid OCR Tuner + Compare")
        self.setGeometry(100, 100, 1000, 700)

        self.current_cv_frame = None
        self.processed_ocr_frame = None
        self.morph_kernel = np.ones((3,3), np.uint8)

        print("EasyOCRモデルをロード中です... (初回は時間がかかります)")
        try:
            self.e_reader = easyocr.Reader(['en'], gpu=False)
            print("EasyOCR ロード完了。")
        except Exception as e:
            print(f"EasyOCRリーダーの初期化に失敗: {e}"); sys.exit(-1)

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
        self.morph_combo = QComboBox(); self.morph_combo.addItems(["なし", "収縮 (ノイズ除去)", "膨張 (穴埋め)", "開放 (ノイズ除去+)", "閉鎖 (穴埋め+)"]); self.morph_combo.setCurrentIndex(1)
        advanced_layout.addRow("モルフォロジー:", self.morph_combo)
        advanced_group.setLayout(advanced_layout); right_layout.addWidget(advanced_group)

        # ▼▼▼ 追加: 正解コード入力欄 ▼▼▼
        compare_group = QGroupBox("4. 比較")
        compare_layout = QFormLayout()
        self.ground_truth_input = QLineEdit()
        self.ground_truth_input.setPlaceholderText("ここに正解のコードを入力 (スペースなし)")
        # 正解コードが変更されたら、表示を更新
        self.ground_truth_input.textChanged.connect(self.trigger_result_update)
        compare_layout.addRow("正解コード:", self.ground_truth_input)
        compare_group.setLayout(compare_layout)
        right_layout.addWidget(compare_group)
        # ▲▲▲ 追加 ▲▲▲

        self.ocr_result_label = QLabel("OCR結果 (リアルタイム): ---")
        self.ocr_result_label.setStyleSheet("font-size: 14px; background-color: #EEE; padding: 5px; line-height: 1.5;") # line-height調整
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

        # ▼▼▼ 追加: 最後に保持した結果を格納する変数 ▼▼▼
        self.last_results = {"A": "", "B": "", "C": ""}
        # ▲▲▲ 追加 ▲▲▲

    def init_threads(self):
        self.camera_thread = CameraThread()
        self.camera_thread.frame_signal.connect(self.update_camera_feed)
        self.camera_thread.start()
        self.ocr_thread = OcrThread(self.e_reader)
        self.ocr_thread.result_signal.connect(self.update_ocr_results) # 3つの結果を受け取る
        self.ocr_thread.start()

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
            self.ocr_thread.set_image(self.processed_ocr_frame)

        except (ValueError, cv2.error):
            self.gray_debug_label.setText("Grayscale (ROI invalid)")
            self.binary_debug_label.setText("Binary (Final)")
            self.processed_ocr_frame = None
            self.ocr_thread.set_image(None)

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

    # ▼▼▼ 追加: 正解コード入力時に結果表示を更新するトリガー ▼▼▼
    def trigger_result_update(self):
        """正解コードが変更されたときにOCR結果表示を強制的に更新"""
        # 最後に受け取った結果を使って表示を更新する
        self.update_ocr_results(
            self.last_results["A"],
            self.last_results["B"],
            self.last_results["C"]
        )
    # ▲▲▲ 追加 ▲▲▲

    # ▼▼▼ 修正: OCR結果表示と比較ロジック ▼▼▼
    @pyqtSlot(str, str, str)
    def update_ocr_results(self, final_A, final_B, final_C):
        """OCRスレッドから結果を受け取り、比較してラベルに表示する"""
        # 最後に受け取った結果を保存
        self.last_results["A"] = final_A
        self.last_results["B"] = final_B
        self.last_results["C"] = final_C

        # 正解コードを取得（大文字化、スペース除去）
        ground_truth = re.sub(r'[^A-Z0-9]', '', self.ground_truth_input.text().upper())

        results_text = []
        methods = {"A": final_A, "B": final_B, "C": final_C}
        method_names = {
            "A": "Tesseract psm7",
            "B": "Tesseract psm8",
            "C": "EasyOCR on Binary"
        }

        for key, result in methods.items():
            line = f"<b>{key} ({method_names[key]}):</b><br/>&nbsp;&nbsp;{result}" # HTMLで太字とインデント
            # 正解コードが入力されていて、かつ16桁で、かつ一致した場合
            if ground_truth and len(result) == 16 and result == ground_truth:
                line += ' <font color="green">✅ Match!</font>' # HTMLで緑色のチェックマーク
            results_text.append(line)

        # HTML形式でラベルに設定
        self.ocr_result_label.setText("<br/>---<br/>".join(results_text))
    # ▲▲▲ 修正 ▲▲▲

    def closeEvent(self, event):
        self.camera_thread.stop()
        self.ocr_thread.stop()
        event.accept()

# --- プログラム実行 ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())