import sys
import cv2
# ▼▼▼ Tesseract関連をすべて削除 ▼▼▼
# import pytesseract 
import re
import numpy as np
# ▼▼▼ EasyOCR をインポート ▼▼▼
try:
    import easyocr
except ImportError:
    print("="*50)
    print("エラー: 'easyocr' が見つかりません。")
    print("コマンドプロンプトで 'pip install easyocr' を実行してください。")
    print("="*50)
    sys.exit(-1)

# ▼▼▼ PyTorch (torch) のチェック ▼▼▼
try:
    import torch
except ImportError:
    print("="*50)
    print("エラー: 'torch' (PyTorch) が見つかりません。")
    print("コマンドプロンプトで 'pip install torch torchvision' を実行してください。")
    print("="*50)
    sys.exit(-1)

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, 
    QHBoxLayout, QPushButton, QLabel, QLineEdit, QFormLayout,
    QGroupBox
    # QSlider, QCheckBox は削除
)
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QTimer, pyqtSlot, QRect
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen

# --- Tesseractのパス設定は不要 ---

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
                # EasyOCRはRGBカラー画像をそのまま使う
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); h, w, ch = rgb_image.shape
                qt_image = QImage(rgb_image.data, w, h, ch * w, QImage.Format.Format_RGB888)
                self.frame_signal.emit(qt_image.copy(), frame) # 生フレーム(frame)も渡す
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
    OCR調整用GUIウィンドウ (v5: EasyOCR)
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQt OCR Tuner (v5: EasyOCR)")
        self.setGeometry(100, 100, 800, 600) # サイズを戻す

        self.current_cv_frame = None 
        # ▼▼▼ 削除: 前処理後の画像は不要 ▼▼▼
        # self.processed_ocr_frame = None 
        
        # ▼▼▼ EasyOCRリーダーの初期化 ▼▼▼
        # (初回起動時、モデルのダウンロードとロードに時間がかかります)
        print("EasyOCRモデルをロード中です... (初回は時間がかかります)")
        try:
            # 'en' (英語) のみを使用。GPUは使わない(False)設定
            self.e_reader = easyocr.Reader(['en'], gpu=False)
            print("EasyOCR ロード完了。")
        except Exception as e:
            print(f"EasyOCRリーダーの初期化に失敗: {e}")
            sys.exit(-1)
        # ▲▲▲ EasyOCR ▲▲▲
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout() 
        central_widget.setLayout(main_layout)

        # --- 1. 左側: カメラのみ ---
        left_layout = QVBoxLayout()
        main_layout.addLayout(left_layout, 7) 
        self.camera_label = CameraLabel("カメラ待機中...")
        self.camera_label.mouse_pos_signal.connect(self.update_camera_coords)
        left_layout.addWidget(self.camera_label) # 100%
 
        # ▼▼▼ 削除: デバッグビュー (Gray, Binary) は不要 ▼▼▼
        # debug_layout = ...
        # ▲▲▲ 削除 ▲▲▲

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
        
        # ▼▼▼ 削除: 前処理設定 (スライダー, チェックボックス) は不要 ▼▼▼
        # preprocess_group = ...
        # ▲▲▲ 削除 ▲▲▲

        # 2c. 実行ボタン
        self.start_button = QPushButton("OCR実行 (EasyOCR)")
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
        self.roi_x1_input.textChanged.connect(self.update_roi_drawing)
        self.roi_y1_input.textChanged.connect(self.update_roi_drawing)
        self.roi_x2_input.textChanged.connect(self.update_roi_drawing)
        self.roi_y2_input.textChanged.connect(self.update_roi_drawing)
        # スライダー等は削除

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
        # 前処理は不要なので update_preprocess_image() は削除

    @pyqtSlot(int, int)
    def update_camera_coords(self, x, y):
        if x == -1: self.cam_coord_label.setText("---")
        else: self.cam_coord_label.setText(f"({x}, {y})")
            
    def update_roi_drawing(self):
        try:
            x1=int(self.roi_x1_input.text()); y1=int(self.roi_y1_input.text())
            x2=int(self.roi_x2_input.text()); y2=int(self.roi_y2_input.text())
            self.camera_label.set_roi(x1, y1, x2, y2)
        except ValueError: self.camera_label.set_roi(0, 0, 0, 0) 

    # ▼▼▼ 削除: update_preprocess_image は不要 ▼▼▼
    # def update_preprocess_image(self): ...
    
    def run_ocr_test(self):
        """「OCR実行 (EasyOCR)」ボタンが押されたときの処理"""
        print(f"\n--- EasyOCR 実行開始 ---")
        
        if self.current_cv_frame is None:
            print("エラー: カメラフレームがありません")
            self.ocr_result_label.setText("OCR結果: エラー (カメラ未起動)")
            return
            
        try:
            x1 = int(self.roi_x1_input.text()); y1 = int(self.roi_y1_input.text())
            x2 = int(self.roi_x2_input.text()); y2 = int(self.roi_y2_input.text())
            if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
                raise ValueError("ROI座標が不正です")

            # ROI領域を切り抜き
            roi_frame = self.current_cv_frame[y1:y2, x1:x2]

        except (ValueError, cv2.error):
            self.ocr_result_label.setText("OCR結果: エラー (ROIが不正)")
            return

        # --- Step 1: 前処理（ノイズ除去＆二値化） ---
        try:
            gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (3,3), 0)
            gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        except Exception as e:
            print(f"前処理エラー: {e}")
            self.ocr_result_label.setText("OCR結果: エラー (前処理失敗)")
            return

        # --- Step 2: OCRを複数回実行して安定化 ---
        all_codes = []
        for i in range(3):
            try:
                ocr_results = self.e_reader.readtext(
                    gray, detail=0,
                    text_threshold=0.4, low_text=0.2, link_threshold=0.3,
                    decoder='greedy', contrast_ths=0.3, adjust_contrast=0.5
                )
                raw_text = " ".join(ocr_results)
                processed = re.sub(r'[^A-Z0-9]', '', raw_text.upper())
                processed = (processed.replace('O', '0')
                                       .replace('I', '1')
                                       .replace('B', '8')
                                       .replace('Z', '2'))
                if processed:
                    all_codes.append(processed)
            except Exception as e:
                print(f"OCR試行{i+1}失敗: {e}")

        if not all_codes:
            self.ocr_result_label.setText("OCR結果: 認識失敗 (文字なし)")
            print("OCR結果: 有効な文字がありませんでした。")
            return

        # --- Step 3: 最頻値を採用 ---
        final_code = max(set(all_codes), key=all_codes.count)
        final_code = final_code[:16]

        print(f"OCR試行結果: {all_codes}")
        print(f"最終採用コード: {final_code}")

        # --- Step 4: GUI表示 ---
        self.ocr_result_label.setText(
            f"EasyOCR結果(試行): {all_codes}\n"
            f"最終コード: {final_code}"
        )
        print("--- 完了 ---")

        
    def closeEvent(self, event):
        self.camera_thread.stop()
        event.accept()

# --- プログラム実行 ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())