import sys
import cv2
import pytesseract
import pyautogui
import pygetwindow
import re
import numpy as np
import time 
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, 
    QHBoxLayout, QPushButton, QLabel, QLineEdit, QFormLayout,
    QGroupBox, QListWidget 
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

# -----------------------------------------------
# (CameraThread, RelativeCoordThread, CameraLabel は変更なし)
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

class RelativeCoordThread(QThread):
    """ (変更なし) """
    coord_signal = pyqtSignal(str) 
    def __init__(self): super().__init__(); self.running = True; self.target_title = ""
    def run(self):
        while self.running:
            if not self.target_title: self.coord_signal.emit("---"); self.msleep(100); continue
            try:
                windows = pygetwindow.getWindowsWithTitle(self.target_title)
                if windows:
                    target_win = windows[0]; mx, my = pyautogui.position(); wx, wy = target_win.topleft
                    self.coord_signal.emit(f"({mx - wx}, {my - wy})")
                else: self.coord_signal.emit("Not Found")
            except Exception as e: self.coord_signal.emit(f"Error")
            self.msleep(100)
    def set_target_title(self, title): self.target_title = title
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

# ▼▼▼ 修正: BatchInputThread (3回目クリック追加) ▼▼▼
class BatchInputThread(QThread):
    """ 自動操作スレッド """
    progress_signal = pyqtSignal(str) 
    finished_signal = pyqtSignal() 

    def __init__(self, codes_to_input, settings):
        super().__init__()
        self.codes = codes_to_input
        self.settings = settings
        self.running = True

    def run(self):
        try:
            target_title = self.settings['title']
            paste_x = self.settings['paste_x']
            paste_y = self.settings['paste_y']
            click_x = self.settings['click_x']
            click_y = self.settings['click_y']
            # 追加
            close_x = self.settings['close_x'] 
            close_y = self.settings['close_y']

            windows = pygetwindow.getWindowsWithTitle(target_title)
            if not windows:
                self.progress_signal.emit(f"エラー: ウィンドウ '{target_title}' が見つかりません。")
                return 
            
            target_win = windows[0]

            for i, code in enumerate(self.codes):
                if not self.running:
                    self.progress_signal.emit("一括処理が中断されました。")
                    return
                
                status_msg = f"入力中 ({i+1}/{len(self.codes)}): {code}"
                self.progress_signal.emit(status_msg); print(status_msg)

                if target_win.isMinimized: target_win.restore()
                target_win.activate()
                
                wx, wy = target_win.topleft
                
                # 座標計算
                abs_paste_x = wx + paste_x; abs_paste_y = wy + paste_y
                abs_click_x = wx + click_x; abs_click_y = wy + click_y
                abs_close_x = wx + close_x; abs_close_y = wy + close_y # 追加

                # 1. 貼り付け位置をクリック + 入力
                pyautogui.click(abs_paste_x, abs_paste_y, duration=0.1)
                pyautogui.write(code, interval=0.01)
                
                # 2. 決定ボタンをクリック
                pyautogui.click(abs_click_x, abs_click_y, duration=0.1)
                
                # 3. ▼▼▼ 追加: 閉じるボタンをクリック ▼▼▼
                #    (決定後の待機時間。ダイアログが閉じるのを待つなど)
                time.sleep(2.0) 
                pyautogui.click(abs_close_x, abs_close_y, duration=0.1)
                # ▲▲▲ 追加 ▲▲▲
                
                # 4. 次の入力までの待機
                time.sleep(1.0) 

            self.progress_signal.emit(f"完了: {len(self.codes)}件のコードが入力されました。")
            time.sleep(1) 

        except Exception as e:
            self.progress_signal.emit(f"自動操作エラー: {e}"); print(f"自動操作エラー: {e}")
        finally:
            if self.running:
                self.finished_signal.emit() 

    def stop(self):
        self.running = False; print("一括入力スレッドに停止リクエスト送信...")
# ▲▲▲ 修正 ▲▲▲
# -----------------------------------------------

class MainWindow(QMainWindow):
    """
    シリアルコード一括入力 (vFinal Batch + Close Btn)
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("シリアルコード一括入力 (vFinal Batch + Close Btn)")
        self.setGeometry(100, 100, 1000, 700) 

        self.current_cv_frame = None 
        self.processed_cv_frame = None 
        self.morph_kernel = np.ones((3,3), np.uint8)
        self.batch_thread = None
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(); central_widget.setLayout(main_layout)
        left_layout = QVBoxLayout(); main_layout.addLayout(left_layout, 7) 
        self.camera_label = CameraLabel("カメラ待機中...")
        self.camera_label.mouse_pos_signal.connect(self.update_camera_coords)
        left_layout.addWidget(self.camera_label, 3) 
        debug_layout = QHBoxLayout()
        self.binary_debug_label = QLabel("Binary (Final)")
        self.binary_debug_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.binary_debug_label.setStyleSheet("background-color: #333; color: white;")
        debug_layout.addWidget(self.binary_debug_label)
        left_layout.addLayout(debug_layout, 1) 

        right_layout = QVBoxLayout(); main_layout.addLayout(right_layout, 3) 
        
        # ▼▼▼ 修正: GUIに「閉じる 座標」を追加 ▼▼▼
        self.settings_group = QGroupBox("A. 設定"); settings_layout = QFormLayout()
        self.roi_x1_input = QLineEdit("100"); self.roi_y1_input = QLineEdit("200")
        self.roi_x2_input = QLineEdit("600"); self.roi_y2_input = QLineEdit("250")
        roi_layout1 = QHBoxLayout(); roi_layout1.addWidget(self.roi_x1_input); roi_layout1.addWidget(self.roi_y1_input)
        settings_layout.addRow("ROI 左上 (X1, Y1):", roi_layout1)
        roi_layout2 = QHBoxLayout(); roi_layout2.addWidget(self.roi_x2_input); roi_layout2.addWidget(self.roi_y2_input)
        settings_layout.addRow("ROI 右下 (X2, Y2):", roi_layout2)
        self.cam_coord_label = QLabel("---"); self.cam_coord_label.setStyleSheet("font-weight: bold; color: green;")
        settings_layout.addRow("カメラ内 座標:", self.cam_coord_label)
        self.title_input = QLineEdit("DBSCGFW")
        settings_layout.addRow("対象Windowタイトル:", self.title_input)
        self.coord_display_label = QLabel("---"); self.coord_display_label.setStyleSheet("font-weight: bold; color: blue;")
        settings_layout.addRow("Window相対座標:", self.coord_display_label)
        self.paste_x_input = QLineEdit("300"); self.paste_y_input = QLineEdit("440")
        paste_coords_layout = QHBoxLayout(); paste_coords_layout.addWidget(self.paste_x_input); paste_coords_layout.addWidget(self.paste_y_input)
        settings_layout.addRow("貼付 座標 (X, Y):", paste_coords_layout)
        self.click_x_input = QLineEdit("1000"); self.click_y_input = QLineEdit("860")
        click_coords_layout = QHBoxLayout(); click_coords_layout.addWidget(self.click_x_input); click_coords_layout.addWidget(self.click_y_input)
        settings_layout.addRow("決定 座標 (X, Y):", click_coords_layout)
        
        # 追加
        self.close_x_input = QLineEdit("0"); self.close_y_input = QLineEdit("0") 
        close_coords_layout = QHBoxLayout(); close_coords_layout.addWidget(self.close_x_input); close_coords_layout.addWidget(self.close_y_input)
        settings_layout.addRow("閉じる 座標 (X, Y):", close_coords_layout)
        
        self.settings_group.setLayout(settings_layout); right_layout.addWidget(self.settings_group) 
        # ▲▲▲ 修正 ▲▲▲
        
        workflow_group = QGroupBox("B. ワークフロー"); workflow_layout = QVBoxLayout()
        self.recognize_button = QPushButton("① 認識してプールに追加")
        self.recognize_button.setStyleSheet("font-size: 16px; background-color: #007BFF; color: white; padding: 8px;")
        self.recognize_button.clicked.connect(self.recognize_and_add_to_pool)
        workflow_layout.addWidget(self.recognize_button)
        self.pool_list_widget = QListWidget(); workflow_layout.addWidget(self.pool_list_widget)
        self.delete_button = QPushButton("② 選択を削除")
        self.delete_button.setStyleSheet("font-size: 14px; background-color: #dc3545; color: white; padding: 5px;")
        self.delete_button.clicked.connect(self.delete_selected_item)
        workflow_layout.addWidget(self.delete_button)
        self.batch_start_button = QPushButton("③ 一括自動入力 開始")
        self.batch_start_button.setStyleSheet("font-size: 18px; background-color: #4CAF50; color: white; padding: 10px;")
        self.batch_start_button.clicked.connect(self.start_batch_input)
        workflow_layout.addWidget(self.batch_start_button)
        self.status_label = QLabel("ステータス: 待機中")
        self.status_label.setStyleSheet("font-size: 14px; background-color: #EEE; padding: 5px;")
        self.status_label.setWordWrap(True); workflow_layout.addWidget(self.status_label)
        workflow_group.setLayout(workflow_layout); right_layout.addWidget(workflow_group)
        right_layout.addStretch() 

        self.init_threads()
        QTimer.singleShot(100, self.update_roi_drawing) 
        self.roi_x1_input.textChanged.connect(self.update_roi_and_preprocess)
        self.roi_y1_input.textChanged.connect(self.update_roi_and_preprocess)
        self.roi_x2_input.textChanged.connect(self.update_roi_and_preprocess)
        self.roi_y2_input.textChanged.connect(self.update_roi_and_preprocess)

    def init_threads(self):
        self.camera_thread = CameraThread(); self.camera_thread.frame_signal.connect(self.update_camera_feed); self.camera_thread.start()
        self.coord_thread = RelativeCoordThread(); self.coord_thread.coord_signal.connect(self.update_coords)
        self.title_input.textChanged.connect(self.coord_thread.set_target_title)
        self.coord_thread.set_target_title(self.title_input.text()); self.coord_thread.start()

    @pyqtSlot(QImage, object)
    def update_camera_feed(self, qt_image, cv_frame):
        self.current_cv_frame = cv_frame 
        scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
            self.camera_label.width(), self.camera_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
        self.camera_label.setPixmap(scaled_pixmap)
        self.update_preprocess_view()

    @pyqtSlot(str)
    def update_coords(self, coord_str): self.coord_display_label.setText(coord_str)

    @pyqtSlot(int, int)
    def update_camera_coords(self, x, y):
        if x == -1: self.cam_coord_label.setText("---")
        else: self.cam_coord_label.setText(f"({x}, {y})")
            
    def update_roi_and_preprocess(self):
        self.update_roi_drawing()
        self.update_preprocess_view()

    def update_roi_drawing(self):
        try:
            x1=int(self.roi_x1_input.text()); y1=int(self.roi_y1_input.text())
            x2=int(self.roi_x2_input.text()); y2=int(self.roi_y2_input.text())
            self.camera_label.set_roi(x1, y1, x2, y2)
        except ValueError: self.camera_label.set_roi(0, 0, 0, 0) 

    def update_preprocess_view(self):
        if self.current_cv_frame is None: return
        try:
            x1 = int(self.roi_x1_input.text()); y1 = int(self.roi_y1_input.text())
            x2 = int(self.roi_x2_input.text()); y2 = int(self.roi_y2_input.text())
            if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1: raise ValueError("ROI座標が不正です")
            roi_frame = self.current_cv_frame[y1:y2, x1:x2]
            gray_frame = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
            gray_frame = cv2.resize(gray_frame, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            _, binary_frame = cv2.threshold(gray_frame, 140, 255, cv2.THRESH_BINARY)
            binary_frame = cv2.erode(binary_frame, self.morph_kernel, iterations=1)
            self.processed_cv_frame = binary_frame 
            self.display_debug_image(binary_frame, self.binary_debug_label)
        except (ValueError, cv2.error):
            self.binary_debug_label.setText("Binary (ROI invalid)")
            self.processed_cv_frame = None

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

    def recognize_and_add_to_pool(self):
        print("--- 認識実行 ---")
        if self.processed_cv_frame is None:
            self.status_label.setText("ステータス: エラー (ROIが不正)")
            return
        frame_to_ocr = self.processed_cv_frame
        try:
            config_psm7 = '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            raw_text = pytesseract.image_to_string(frame_to_ocr, lang='eng', config=config_psm7)
        except Exception as e:
            self.status_label.setText(f"ステータス: Tesseract エラー {e}")
            return
        processed_code = re.sub(r'[^A-Z0-9]', '', raw_text.upper()) 
        if len(processed_code) != 16:
            error_msg = f"ステータス: 桁数エラー ({len(processed_code)}桁)"
            print(f"エラー: 16桁ではありません (認識: '{processed_code}')")
            self.status_label.setText(error_msg)
            return 
        final_code = processed_code
        self.pool_list_widget.addItem(final_code)
        self.status_label.setText(f"追加: {final_code}")
        print(f"プールに追加: '{final_code}'")
    
    def delete_selected_item(self):
        selected_items = self.pool_list_widget.selectedItems()
        if not selected_items:
            self.status_label.setText("ステータス: 削除するコードを選択してください")
            return
        for item in selected_items:
            row = self.pool_list_widget.row(item)
            self.pool_list_widget.takeItem(row)
            self.status_label.setText(f"削除: {item.text()}")
            print(f"プールから削除: {item.text()}")
    
    # ▼▼▼ 修正: start_batch_input (3つ目の座標を渡す) ▼▼▼
    def start_batch_input(self):
        """「③ 一括自動入力 開始」ボタンが押されたときの処理"""
        count = self.pool_list_widget.count()
        if count == 0:
            self.status_label.setText("ステータス: プールが空です")
            return
        codes = [self.pool_list_widget.item(i).text() for i in range(count)]
        try:
            settings = {
                'title': self.title_input.text(),
                'paste_x': int(self.paste_x_input.text()),
                'paste_y': int(self.paste_y_input.text()),
                'click_x': int(self.click_x_input.text()),
                'click_y': int(self.click_y_input.text()),
                'close_x': int(self.close_x_input.text()), # 追加
                'close_y': int(self.close_y_input.text()), # 追加
            }
        except ValueError:
            self.status_label.setText("ステータス: 座標(X, Y)が数字ではありません")
            return

        self.set_gui_enabled(False)
        self.status_label.setText(f"一括入力開始... (全 {count} 件)")
        
        self.batch_thread = BatchInputThread(codes, settings)
        self.batch_thread.progress_signal.connect(self.update_status_label)
        self.batch_thread.finished_signal.connect(self.on_batch_finished) 
        self.batch_thread.start()
    # ▲▲▲ 修正 ▲▲▲

    @pyqtSlot(str)
    def update_status_label(self, text):
        self.status_label.setText(text)
        
    def set_gui_enabled(self, enabled):
        self.recognize_button.setEnabled(enabled)
        self.delete_button.setEnabled(enabled)
        self.batch_start_button.setEnabled(enabled)
        self.settings_group.setEnabled(enabled) 
    
    @pyqtSlot()
    def on_batch_finished(self):
        print("バッチ処理スレッド正常終了。ウィンドウを閉じます。")
        self.close() 

    def closeEvent(self, event):
        print("終了処理を開始...")
        self.camera_thread.stop()
        self.coord_thread.stop()
        if self.batch_thread and self.batch_thread.isRunning():
            print("バッチ処理を中断しています...")
            self.batch_thread.stop()
            self.batch_thread.wait(2000) 
        else:
            print("バッチ処理は実行中でないか、既に完了しています。")
        event.accept()

# --- プログラム実行 ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())