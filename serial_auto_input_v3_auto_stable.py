import sys
import cv2
import pytesseract
import pyautogui
import pygetwindow
import re
import numpy as np
import time
import datetime
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QPushButton, QLabel, QLineEdit, QFormLayout,
    QGroupBox, QListWidget, QSlider, QCheckBox, QComboBox,
    QStyledItemDelegate, QStyle
)
from PyQt6.QtCore import (
    QThread, pyqtSignal, Qt, QTimer, pyqtSlot, QRect, QSettings
)
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QTextDocument

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
    def stop(self): 
        self.running = False

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
    def stop(self): 
        self.running = False

class CameraLabel(QLabel):
    """ (修正) ROI描画とマウス座標特定機能 + Successオーバーレイ """
    mouse_pos_signal = pyqtSignal(int, int) 
    def __init__(self, text):
        super().__init__(text); self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background-color: black; color: white;"); self.pixmap_size = None
        self.image_origin = None; self.roi_rect = QRect(); self.setMouseTracking(True)
        self.original_size = (1, 1) # ▼▼▼ 追加: 元画像のサイズ (初期値) ▼▼▼
        self.show_success_overlay = False # ▼▼▼ 追加 ▼▼▼

    def setPixmap(self, pixmap):
        super().setPixmap(pixmap); self.pixmap_size = pixmap.size()
        if self.pixmap_size:
            margin_x = (self.width() - self.pixmap_size.width()) // 2
            margin_y = (self.height() - self.pixmap_size.height()) // 2
            self.image_origin = (margin_x, margin_y)
        else: self.image_origin = None
    # ▼▼▼ 追加: 元画像のサイズをセットするメソッド ▼▼▼
    def set_original_size(self, w, h): self.original_size = (w, h)

    def set_roi(self, x1, y1, x2, y2): self.roi_rect = QRect(x1, y1, x2 - x1, y2 - y1); self.update()
    
    # ▼▼▼ 追加 ▼▼▼
    def set_success_overlay(self, visible):
        """Success! 表示のオン/オフを切り替え、再描画を要求する"""
        self.show_success_overlay = visible
        self.update() # paintEventをトリガー
    # ▲▲▲ 追加 ▲▲▲

    def mouseMoveEvent(self, event):
        if self.image_origin:
            ox, oy = self.image_origin; local_x = event.pos().x() - ox; local_y = event.pos().y() - oy
            if 0 <= local_x < self.pixmap_size.width() and 0 <= local_y < self.pixmap_size.height():
                # ▼▼▼ 修正: スケールを考慮して元画像の座標に変換 ▼▼▼
                org_w, org_h = self.original_size
                if org_w > 0 and org_h > 0:
                    scale_x = self.pixmap_size.width() / org_w
                    scale_y = self.pixmap_size.height() / org_h
                    real_x = int(local_x / scale_x); real_y = int(local_y / scale_y)
                    self.mouse_pos_signal.emit(real_x, real_y)
                else:
                    self.mouse_pos_signal.emit(local_x, local_y)
            else: self.mouse_pos_signal.emit(-1, -1)

    # ▼▼▼ 修正 (paintEvent) ▼▼▼
    def paintEvent(self, event):
        super().paintEvent(event) 
        painter = QPainter(self) # Painterを早期に初期化

        # 1. ROIの描画
        if self.image_origin and not self.roi_rect.isNull():
            ox, oy = self.image_origin            
            # ▼▼▼ 追加: ROI座標を表示サイズに合わせてスケーリング ▼▼▼
            org_w, org_h = self.original_size
            scale_x = self.pixmap_size.width() / org_w if org_w > 0 else 1.0
            scale_y = self.pixmap_size.height() / org_h if org_h > 0 else 1.0
            r = self.roi_rect
            scaled_rect = QRect(int(r.x()*scale_x), int(r.y()*scale_y), int(r.width()*scale_x), int(r.height()*scale_y))
            draw_rect = scaled_rect.translated(ox, oy)
            # ▲▲▲ 追加 ▲▲▲

            pen = QPen(Qt.GlobalColor.red, 2)
            painter.setPen(pen)
            painter.drawRect(draw_rect)

        # 2. Successオーバーレイの描画
        if self.show_success_overlay:
            # フォント設定
            font = painter.font()
            font.setPointSize(48) # 大きなフォントサイズ
            font.setBold(True)
            painter.setFont(font)
            
            # ペン（色）設定
            pen = QPen(Qt.GlobalColor.green, 3) # 太い緑色のペン
            painter.setPen(pen)
            
            # 中央揃えでテキストを描画
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Success!")

        painter.end() # 最後にPainterを終了
    # ▲▲▲ 修正 ▲▲▲
# ▼▼▼ 追加: 数字だけを赤く表示するためのデリゲート ▼▼▼
class NumberHighlightDelegate(QStyledItemDelegate):
    def paint(self, painter, option, index):
        painter.save()
        
        # 1. 選択状態の背景を描画
        if option.state & QStyle.StateFlag.State_Selected:
            painter.fillRect(option.rect, option.palette.highlight())

        # 2. テキストを取得し、数字部分をHTMLで赤色・太字に装飾
        text = index.data()
        # 数字(0-9)を 赤色(#FF0000) かつ 太字(<b>) に置換
        html_text = re.sub(r'([0-9]+)', r'<font color="#FF0000"><b>\1</b></font>', text)

        # 3. 通常テキストの色（選択時は白、通常は黒などシステム準拠）を取得
        text_color = option.palette.text().color().name()
        if option.state & QStyle.StateFlag.State_Selected:
            text_color = option.palette.highlightedText().color().name()

        # 4. HTML描画用のドキュメントを作成
        doc = QTextDocument()
        doc.setDefaultFont(option.font)
        doc.setHtml(f"<div style='color:{text_color}'>{html_text}</div>")
        
        # 5. 指定位置に描画
        painter.translate(option.rect.x(), option.rect.y())
        doc.drawContents(painter)

        painter.restore()
# ▲▲▲ 追加 ▲▲▲

#
# -----------------------------------------------
# (BatchInputThread は変更なし)
# -----------------------------------------------

class BatchInputThread(QThread):
    """ 自動操作スレッド """
    progress_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(self, codes_to_input, settings):
        super().__init__()
        self.codes = codes_to_input
        self.settings = settings
        self.running = True

    def save_codes_to_file(self):
        """現在のコードリストを日時付きファイルに保存する"""
        try:
            now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"result_{now}.txt"
            with open(filename, "w") as f:
                f.write("\n".join(self.codes))
            return filename
        except Exception as e:
            return None

    def run(self):
        try:
            target_title = self.settings['title']
            paste_x = self.settings['paste_x']
            paste_y = self.settings['paste_y']
            click_x = self.settings['click_x']
            click_y = self.settings['click_y']
            close_x = self.settings['close_x']
            close_y = self.settings['close_y']
            wait_paste_first = self.settings['wait_paste_first']
            wait_close = self.settings['wait_close']
            wait_paste_loop = self.settings['wait_paste_loop']
            windows = pygetwindow.getWindowsWithTitle(target_title)
            if not windows:
                self.progress_signal.emit(f"エラー: ウィンドウ '{target_title}' が見つかりません。")
                return
            
            target_win = windows[0]

            # ▼▼▼【重要】初回アクティブ化のための待機処理 ▼▼▼
            # (PyQtアプリから対象アプリへフォーカスを移すための時間)
            self.progress_signal.emit("対象ウィンドウの準備中...")
            if target_win.isMinimized: target_win.restore()
            target_win.activate()
            
            # OSがウィンドウのフォーカスを切り替えるための「猶予時間」
            time.sleep(wait_paste_first) 
            # ▲▲▲ 追加 ▲▲▲

            for i, code in enumerate(self.codes):
                if not self.running:
                    self.progress_signal.emit("一括処理が中断されました。")
                    # ▼▼▼ 追加: 中断時のファイル保存 ▼▼▼
                    saved = self.save_codes_to_file()
                    if saved: self.progress_signal.emit(f"ログ保存: {saved}")
                    # ▲▲▲ 追加 ▲▲▲
                    return
                
                status_msg = f"入力中 ({i+1}/{len(self.codes)}): {code}"
                self.progress_signal.emit(status_msg); print(status_msg)

                # (念のためループ内でもアクティブ化するが、初回ほどの待機は不要)
                if target_win.isMinimized: target_win.restore()
                target_win.activate()
                
                wx, wy = target_win.topleft
                
                # 座標計算
                abs_paste_x = wx + paste_x; abs_paste_y = wy + paste_y
                abs_click_x = wx + click_x; abs_click_y = wy + click_y
                abs_close_x = wx + close_x; abs_close_y = wy + close_y 

                # 1. 貼り付け位置をクリック + 入力
                pyautogui.click(abs_paste_x, abs_paste_y, duration=0.1)
                pyautogui.write(code, interval=0.001)
                
                # 2. 決定ボタンをクリック
                pyautogui.click(abs_click_x, abs_click_y, duration=0.1)
                
                # 3. 閉じるボタンをクリック
                time.sleep(wait_close) 
                pyautogui.click(abs_close_x, abs_close_y, duration=0.1)
                
                # 4. 次の入力までの待機
                time.sleep(wait_paste_loop) 

            self.progress_signal.emit(f"完了: {len(self.codes)}件のコードが入力されました。")
            # ▼▼▼ 追加: 完了時のファイル保存 ▼▼▼
            saved = self.save_codes_to_file()
            if saved: self.progress_signal.emit(f"ログ保存完了: {saved}")
            # ▲▲▲ 追加 ▲▲▲
            time.sleep(1) 

        except Exception as e:
            self.progress_signal.emit(f"自動操作エラー: {e}"); print(f"自動操作エラー: {e}")
        finally:
            if self.running:
                self.finished_signal.emit() 

    def stop(self):
        self.running = False; print("一括入力スレッドに停止リクエスト送信...")

# ▼▼▼ 追加: 自動認識と安定化のためのOCRスレッド ▼▼▼
class OcrRecognitionThread(QThread):
    """
    指定された画像を継続的にOCRし、
    4回連続で同じ16桁のコードが認識されたらシグナルを発行するスレッド。
    """
    code_stable_signal = pyqtSignal(str) # 安定したコードを送信
    status_signal = pyqtSignal(str)      # 現在の認識状況を送信

    def __init__(self, stability_count=4):
        super().__init__()
        self.running = True
        self.frame_to_ocr = None
        self.request_ocr = False
        self.last_code = ""
        self.streak = 0
        self.STABILITY_COUNT = stability_count

    def set_image(self, image):
        """メインスレッドから処理対象の画像を受け取る"""
        self.frame_to_ocr = image.copy() if image is not None else None
        self.request_ocr = True

    def fix_char_pre_classification(self, char, h, max_h):
        """
        高さで「数値」か「文字」かを事前判定し、強制的に型にはめる
        前提: 数値(2-9)は背が高く、文字(A-Z)は背が低い
        追加対応: 8vsB, 6vsG, 9vsD, Zvs2, Svs5
        除外文字: 0, 1, I, O (これらは他へマッピング)
        """

        # 高さの比率（現在の文字 ÷ 最大の文字）
        ratio = h / max_h
        # 閾値: 最大高さの 92% 以上なら「数値」とみなす
        is_tall_number = ratio > 0.92

        if is_tall_number:
            # --- 数値グループ (2-9) としての補正 ---
            if char == 'Z': return '2'
            if char == 'S': return '5'
            if char == 'B': return '8' # 8 vs B
            if char == 'G': return '6' # 6 vs G

            # 9の誤認識パターン (O, 0, I, 1, D, Q など)
            # 0, 1, O, I は存在しないため、これらは 9 の可能性が高い
            # D, Q も背が高いなら 9 (Dは本来背が低いはずだが、誤認識でここに来た場合)
            if char in ['O', '0', 'D', 'Q', 'I', '1', 'l']:
                return '9'

            return char

        else:
            # --- 文字グループ (A-Z) としての補正 ---
            if char == '2': return 'Z'
            if char == '5': return 'S'
            if char == '8': return 'B' # 8 vs B
            if char == '6': return 'G' # 6 vs G

            # Dの誤認識パターン (0, O)
            # 背が低いのに 0 や O と認識された -> D の可能性が高い
            if char in ['0', 'O']:
                return 'D'

        return char

    def run(self):
        config_psm7 = '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        
        while self.running:
            if self.request_ocr and self.frame_to_ocr is not None:
                self.request_ocr = False # フラグをリセット
                frame_ocr = self.frame_to_ocr

                try:
                    # ▼▼▼ 変更: image_to_boxes で座標付きOCRを実行 ▼▼▼
                    # 座標を取得して、形状による補正を行う
                    boxes = pytesseract.image_to_boxes(frame_ocr, lang='eng', config=config_psm7)
                    
                    char_data_list = []
                    
                    # 1. まず全文字のデータをパースしてリスト化
                    for b in boxes.splitlines():
                        b = b.split(' ')
                        if len(b) < 6: continue
                        
                        char = b[0]
                        # TesseractのY座標を取得 (高さ計算のみに使用)
                        y1_tess = int(b[2])

                        y2_tess = int(b[4])

                        h_char = y2_tess - y1_tess

                        char_data_list.append({
                            'char': char,
                            'h': h_char
                        })

                    if not char_data_list:
                        continue

                    # 2. 高さの基準値（最大値）を算出
                    # ノイズで極端に大きな枠が取れてしまう場合は調整が必要ですが、基本は最大値でOK
                    max_h = max(c['h'] for c in char_data_list)

                    # 3. 高さによる補正を行いながら文字列を結合
                    final_chars = []
                    for c_data in char_data_list:
                        fixed_char = self.fix_char_pre_classification(c_data['char'], c_data['h'], max_h)
                        final_chars.append(fixed_char)

                    raw_text = "".join(final_chars)
                    processed_code = re.sub(r'[^A-Z0-9]', '', raw_text.upper())
                    # ▲▲▲ 変更終了 ▲▲▲
                    # 16桁かチェック
                    if len(processed_code) != 16:
                        if self.streak > 0: # ストリークが途切れた
                            self.status_signal.emit(f"スキャン中... (ストリーク途絶)")
                        self.streak = 0
                        self.last_code = ""
                    else:
                        # 16桁の場合、前回のコードと比較
                        if processed_code == self.last_code:
                            self.streak += 1
                        else:
                            # 新しいコードを認識
                            self.last_code = processed_code
                            self.streak = 1
                        
                        status_msg = f"認識中: ...{self.last_code[-6:]} (一致: {self.streak}/{self.STABILITY_COUNT})"
                        self.status_signal.emit(status_msg)

                        # 安定化カウントに達したか
                        if self.streak == self.STABILITY_COUNT:
                            self.code_stable_signal.emit(self.last_code)
                            # 連続で追加されないようにリセット
                            self.last_code = ""
                            self.streak = 0

                except Exception as e:
                    self.status_signal.emit(f"OCRエラー: {e}")
                    self.last_code = ""
                    self.streak = 0
            
            else:
                # 処理する画像がない場合は待機
                self.msleep(50)

    def stop(self):
        self.running = False
        self.wait()
# ▲▲▲ 追加 ▲▲▲


class MainWindow(QMainWindow):
    """
    シリアルコード一括入力 (v3 Auto-Stable + Settings)
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("シリアルコード一括入力 (v3 Auto-Stable + Settings)")
        self.setGeometry(100, 100, 1000, 700) 

        self.current_cv_frame = None 
        self.processed_cv_frame = None 
        self.morph_kernel = np.ones((3,3), np.uint8)
        self.batch_thread = None
        self.ocr_thread = None # ▼▼▼ 変更
        
        # ▼▼▼ 追加 ▼▼▼
        # Success表示用のワンショットタイマー
        self.success_timer = QTimer(self)
        self.success_timer.setSingleShot(True)
        self.success_timer.timeout.connect(self.hide_success_overlay)
        # ▲▲▲ 追加 ▲▲▲

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
        
        # ▼▼▼ 修正: GUIに前処理設定を追加 ▼▼▼
        self.settings_group = QGroupBox("A. 設定"); settings_layout = QFormLayout()
        
        # --- ROI
        self.roi_x1_input = QLineEdit("100"); self.roi_y1_input = QLineEdit("200")
        self.roi_x2_input = QLineEdit("600"); self.roi_y2_input = QLineEdit("250")
        roi_layout1 = QHBoxLayout(); roi_layout1.addWidget(self.roi_x1_input); roi_layout1.addWidget(self.roi_y1_input)
        settings_layout.addRow("ROI 左上 (X1, Y1):", roi_layout1)
        roi_layout2 = QHBoxLayout(); roi_layout2.addWidget(self.roi_x2_input); roi_layout2.addWidget(self.roi_y2_input)
        settings_layout.addRow("ROI 右下 (X2, Y2):", roi_layout2)
        self.cam_coord_label = QLabel("---"); self.cam_coord_label.setStyleSheet("font-weight: bold; color: green;")
        settings_layout.addRow("カメラ内 座標:", self.cam_coord_label)

        # --- 前処理 (ここから追加)
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal); self.threshold_slider.setRange(0, 255); self.threshold_slider.setValue(140)
        self.threshold_slider_label = QLabel("140"); self.threshold_slider.valueChanged.connect(lambda val: self.threshold_slider_label.setText(str(val)))
        slider_layout = QHBoxLayout(); slider_layout.addWidget(self.threshold_slider); slider_layout.addWidget(self.threshold_slider_label)
        settings_layout.addRow("二値化しきい値:", slider_layout)
        self.invert_checkbox = QCheckBox("白黒反転 (黒背景/白文字)"); self.invert_checkbox.setChecked(True)
        settings_layout.addRow(self.invert_checkbox)
        self.resize_combo = QComboBox(); self.resize_combo.addItems(["なし (100%)", "200% (Linear 高速)", "200% (Cubic 高画質)"]); self.resize_combo.setCurrentIndex(2)
        settings_layout.addRow("アップスケール:", self.resize_combo)
        self.morph_combo = QComboBox(); self.morph_combo.addItems(["なし", "収縮 (ノイズ除去)", "膨張 (穴埋め)", "開放 (ノイズ除去+)", "閉鎖 (穴埋め+)"]); self.morph_combo.setCurrentIndex(1)
        settings_layout.addRow("モルフォロジー:", self.morph_combo)
        
        # --- 自動操作
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
        self.close_x_input = QLineEdit("0"); self.close_y_input = QLineEdit("0") 
        close_coords_layout = QHBoxLayout(); close_coords_layout.addWidget(self.close_x_input); close_coords_layout.addWidget(self.close_y_input)
        settings_layout.addRow("閉じる 座標 (X, Y):", close_coords_layout)

        # --- 待機時間設定 (新規追加) ---
        self.wait_paste_first_input = QLineEdit("1.5"); self.wait_close_input = QLineEdit("3.0"); self.wait_paste_loop_input = QLineEdit("3.0")
        wait_layout = QHBoxLayout()
        wait_layout.addWidget(QLabel("初回Paste前:")); wait_layout.addWidget(self.wait_paste_first_input)
        wait_layout.addWidget(QLabel("Close前:")); wait_layout.addWidget(self.wait_close_input)
        wait_layout.addWidget(QLabel("Loop・Paste前:")); wait_layout.addWidget(self.wait_paste_loop_input)
        settings_layout.addRow("待機時間(秒):", wait_layout)
        
        self.settings_group.setLayout(settings_layout); right_layout.addWidget(self.settings_group) 
        # ▲▲▲ 修正 ▲▲▲
        
        # ▼▼▼ 修正: ワークフローから「認識ボタン」を削除 ▼▼▼
        workflow_group = QGroupBox("B. ワークフロー"); workflow_layout = QVBoxLayout()
        
        self.pool_list_widget = QListWidget(); workflow_layout.addWidget(self.pool_list_widget)
        # ▼▼▼ 追加: 数字ハイライト機能をリストに適用 ▼▼▼
        self.pool_list_widget.setItemDelegate(NumberHighlightDelegate())
        # ▲▲▲ 追加 ▲▲▲        
        self.delete_button = QPushButton("① 選択を削除")
        self.delete_button.setStyleSheet("font-size: 14px; background-color: #dc3545; color: white; padding: 5px;")
        self.delete_button.clicked.connect(self.delete_selected_item)
        workflow_layout.addWidget(self.delete_button)
        
        self.batch_start_button = QPushButton("② 一括自動入力 開始")
        self.batch_start_button.setStyleSheet("font-size: 18px; background-color: #4CAF50; color: white; padding: 10px;")
        self.batch_start_button.clicked.connect(self.start_batch_input)
        workflow_layout.addWidget(self.batch_start_button)
        
        self.status_label = QLabel("ステータス: 初期化中...")
        self.status_label.setStyleSheet("font-size: 14px; background-color: #EEE; padding: 5px;")
        self.status_label.setWordWrap(True); workflow_layout.addWidget(self.status_label)
        workflow_group.setLayout(workflow_layout); right_layout.addWidget(workflow_group)
        right_layout.addStretch() 
        # ▲▲▲ 修正 ▲▲▲

        self.init_threads()
        QTimer.singleShot(100, self.update_roi_drawing) 
        
        # ROIの変更を接続
        self.roi_x1_input.textChanged.connect(self.update_roi_and_preprocess)
        self.roi_y1_input.textChanged.connect(self.update_roi_and_preprocess)
        self.roi_x2_input.textChanged.connect(self.update_roi_and_preprocess)
        self.roi_y2_input.textChanged.connect(self.update_roi_and_preprocess)
        
        # ▼▼▼ 追加: 新しいGUI要素の変更を接続 ▼▼▼
        self.threshold_slider.valueChanged.connect(self.update_roi_and_preprocess)
        self.invert_checkbox.stateChanged.connect(self.update_roi_and_preprocess)
        self.resize_combo.currentIndexChanged.connect(self.update_roi_and_preprocess)
        self.morph_combo.currentIndexChanged.connect(self.update_roi_and_preprocess)
        # ▲▲▲ 追加 ▲▲▲
        
        # ▼▼▼ 追加: 起動時に設定を読み込む ▼▼▼
        self.load_settings()
        # ▲▲▲ 追加 ▲▲▲


    def init_threads(self):
        # カメラ
        self.camera_thread = CameraThread(); self.camera_thread.frame_signal.connect(self.update_camera_feed); self.camera_thread.start()
        
        # 座標
        self.coord_thread = RelativeCoordThread(); self.coord_thread.coord_signal.connect(self.update_coords)
        self.title_input.textChanged.connect(self.coord_thread.set_target_title)
        self.coord_thread.set_target_title(self.title_input.text()); self.coord_thread.start()
        
        # ▼▼▼ 追加: OCRスレッド ▼▼▼
        self.ocr_thread = OcrRecognitionThread()
        self.ocr_thread.code_stable_signal.connect(self.add_code_to_pool)
        self.ocr_thread.status_signal.connect(self.update_ocr_status)
        self.ocr_thread.start()
        # ▲▲▲ 追加 ▲▲▲


    @pyqtSlot(QImage, object)
    def update_camera_feed(self, qt_image, cv_frame):
        self.current_cv_frame = cv_frame 
        h, w, ch = cv_frame.shape; self.camera_label.set_original_size(w, h) # ▼▼▼ 追加: サイズ通知 ▼▼▼
        scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
            self.camera_label.width(), self.camera_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
        self.camera_label.setPixmap(scaled_pixmap)
        self.update_preprocess_view() # 毎フレーム前処理ビューも更新

    @pyqtSlot(str)
    def update_coords(self, coord_str): self.coord_display_label.setText(coord_str)

    @pyqtSlot(int, int)
    def update_camera_coords(self, x, y):
        if x == -1: self.cam_coord_label.setText("---")
        else: self.cam_coord_label.setText(f"({x}, {y})")
            
    def update_roi_and_preprocess(self):
        """ROIと前処理の両方を更新する（GUI要素が変更されたときに呼ばれる）"""
        self.update_roi_drawing()
        self.update_preprocess_view()

    def update_roi_drawing(self):
        try:
            x1=int(self.roi_x1_input.text()); y1=int(self.roi_y1_input.text())
            x2=int(self.roi_x2_input.text()); y2=int(self.roi_y2_input.text())
            self.camera_label.set_roi(x1, y1, x2, y2)
        except ValueError: self.camera_label.set_roi(0, 0, 0, 0) 

    # ▼▼▼ 修正: update_preprocess_view (ハードコードされた値をGUIから取得) ▼▼▼
    def update_preprocess_view(self):
        """
        現在のカメラフレームとGUI設定に基づき、
        前処理画像（二値化画像）を生成し、デバッグビューに表示し、
        OCRスレッドに渡す。
        """
        if self.current_cv_frame is None: return
        try:
            # 1. ROI切り出し
            # ROIは元画像の座標系で指定される
            x1 = int(self.roi_x1_input.text()); y1 = int(self.roi_y1_input.text())
            x2 = int(self.roi_x2_input.text()); y2 = int(self.roi_y2_input.text())
            if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1: raise ValueError("ROI座標が不正です")
            
            roi_frame = self.current_cv_frame[y1:y2, x1:x2]
            gray_frame = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
            
            # 2. リサイズ (GUIから取得)
            resize_idx = self.resize_combo.currentIndex()
            if resize_idx == 1: gray_frame = cv2.resize(gray_frame, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
            elif resize_idx == 2: gray_frame = cv2.resize(gray_frame, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

            # 3. 二値化 (GUIから取得)
            threshold_val = self.threshold_slider.value()
            threshold_type = cv2.THRESH_BINARY if self.invert_checkbox.isChecked() else cv2.THRESH_BINARY_INV
            _, binary_frame = cv2.threshold(gray_frame, threshold_val, 255, threshold_type)

            # 4. モルフォロジー (GUIから取得)
            morph_idx = self.morph_combo.currentIndex()
            if morph_idx == 1: binary_frame = cv2.erode(binary_frame, self.morph_kernel, iterations=1)
            elif morph_idx == 2: binary_frame = cv2.dilate(binary_frame, self.morph_kernel, iterations=1)
            elif morph_idx == 3: binary_frame = cv2.morphologyEx(binary_frame, cv2.MORPH_OPEN, self.morph_kernel)
            elif morph_idx == 4: binary_frame = cv2.morphologyEx(binary_frame, cv2.MORPH_CLOSE, self.morph_kernel)

            # 5. 結果を保存し、デバッグ表示
            self.processed_cv_frame = binary_frame 
            self.display_debug_image(binary_frame, self.binary_debug_label)
            
            # 6. OCRスレッドに画像を渡す
            if self.ocr_thread:
                self.ocr_thread.set_image(self.processed_cv_frame)

        except (ValueError, cv2.error):
            self.binary_debug_label.setText("Binary (ROI invalid)")
            self.processed_cv_frame = None
            if self.ocr_thread:
                self.ocr_thread.set_image(None)
    # ▲▲▲ 修正 ▲▲▲

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

    # ▼▼▼ 削除: recognize_and_add_to_pool (OcrRecognitionThreadに移行) ▼▼▼
    # def recognize_and_add_to_pool(self): ...
    # ▲▲▲ 削除 ▲▲▲
    
    # ▼▼▼ 修正: ボタンのラベル変更に合わせてメソッド名も変更 (内容は同じ) ▼▼▼
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
    
    # ▼▼▼ 修正: start_batch_input (OCRスレッドを停止する処理を追加) ▼▼▼
    def start_batch_input(self):
        """「② 一括自動入力 開始」ボタンが押されたときの処理"""
        count = self.pool_list_widget.count()
        if count == 0:
            self.status_label.setText("ステータス: プールが空です")
            return
        
        # OCRスレッドが動いていれば停止する
        if self.ocr_thread and self.ocr_thread.isRunning():
            print("自動認識スレッドを停止します...")
            self.ocr_thread.stop()
            self.ocr_thread.wait(1000) # 停止を待つ
            print("自動認識スレッド停止完了。")
            
        codes = [self.pool_list_widget.item(i).text() for i in range(count)]
        try:
            settings = {
                'title': self.title_input.text(),
                'paste_x': int(self.paste_x_input.text()),
                'paste_y': int(self.paste_y_input.text()),
                'click_x': int(self.click_x_input.text()),
                'click_y': int(self.click_y_input.text()),
                'close_x': int(self.close_x_input.text()), 
                'close_y': int(self.close_y_input.text()), 
                'wait_paste_first': float(self.wait_paste_first_input.text()),
                'wait_close': float(self.wait_close_input.text()),
                'wait_paste_loop': float(self.wait_paste_loop_input.text()),
            }
        except ValueError:
            self.status_label.setText("ステータス: 座標(X, Y)が数字ではありません")
            # 停止したOCRスレッドを再開
            if not self.ocr_thread or not self.ocr_thread.isRunning():
                self.init_threads_ocr_only() # OCRスレッドだけ再開
            return

        self.set_gui_enabled(False) # GUIを無効化
        self.status_label.setText(f"一括入力開始... (全 {count} 件)")
        
        self.batch_thread = BatchInputThread(codes, settings)
        self.batch_thread.progress_signal.connect(self.update_status_label)
        self.batch_thread.finished_signal.connect(self.on_batch_finished) 
        self.batch_thread.start()
    # ▲▲▲ 修正 ▲▲▲

    def init_threads_ocr_only(self):
        """OCRスレッドだけを初期化・開始するヘルパー"""
        print("OCRスレッドを再開します...")
        self.ocr_thread = OcrRecognitionThread()
        self.ocr_thread.code_stable_signal.connect(self.add_code_to_pool)
        self.ocr_thread.status_signal.connect(self.update_ocr_status)
        self.ocr_thread.start()
        # 既存の画像で即時スキャン開始
        self.update_preprocess_view()

    @pyqtSlot(str)
    def update_status_label(self, text):
        """バッチ処理スレッドからの進捗表示"""
        self.status_label.setText(text)
        
    # ▼▼▼ 追加: OCRスレッド専用のステータス更新 ▼▼▼
    @pyqtSlot(str)
    def update_ocr_status(self, text):
        """OCRスレッドからの状況表示（バッチ実行中は更新しない）"""
        if (self.batch_thread is None) or (not self.batch_thread.isRunning()):
            self.status_label.setText(text)
    # ▲▲▲ 追加 ▲▲▲

# ▼▼▼ 修正 (add_code_to_pool) ▼▼▼
    @pyqtSlot(str)
    def add_code_to_pool(self, stable_code):
        """OCRスレッドから安定化コードを受け取り、リストに追加する"""
        
        # 成功表示をトリガー (重複チェックの前に行う)
        self.camera_label.set_success_overlay(True)
        self.success_timer.start(1000) # 1秒タイマー開始
            
        # 既にリストにないか確認
        items = [self.pool_list_widget.item(i).text() for i in range(self.pool_list_widget.count())]
        if stable_code not in items:
            self.pool_list_widget.addItem(stable_code)
            self.status_label.setText(f"プール追加: {stable_code}")
            print(f"プールに自動追加: '{stable_code}'")
        else:
            print(f"コードは既にプールに存在: '{stable_code}'")
            # (表示は既に行われている)
    # ▲▲▲ 修正 ▲▲▲
        
    def set_gui_enabled(self, enabled):
        # 認識ボタンは削除された
        self.delete_button.setEnabled(enabled)
        self.batch_start_button.setEnabled(enabled)
        self.settings_group.setEnabled(enabled) 
    
    @pyqtSlot()
    def on_batch_finished(self):
        print("バッチ処理スレッド正常終了。ウィンドウを閉じます。")
        self.close() # 処理完了後にアプリを閉じる

    # ▼▼▼ 追加 (hide_success_overlay) ▼▼▼
    @pyqtSlot()
    def hide_success_overlay(self):
        """タイマーが切れたらSuccess表示を消す"""
        self.camera_label.set_success_overlay(False)
    # ▲▲▲ 追加 ▲▲▲

    # ▼▼▼ 追加: 設定保存・読込メソッド ▼▼▼
    def save_settings(self):
        """現在のGUI設定を保存する"""
        print("設定を保存しています...")
        settings = QSettings("MyCompany", "SerialAutoInput")
        
        # ROI
        settings.setValue("roi/x1", self.roi_x1_input.text())
        settings.setValue("roi/y1", self.roi_y1_input.text())
        settings.setValue("roi/x2", self.roi_x2_input.text())
        settings.setValue("roi/y2", self.roi_y2_input.text())
        
        # 前処理
        settings.setValue("preprocess/threshold", self.threshold_slider.value())
        settings.setValue("preprocess/invert", self.invert_checkbox.isChecked())
        settings.setValue("preprocess/resize_idx", self.resize_combo.currentIndex())
        settings.setValue("preprocess/morph_idx", self.morph_combo.currentIndex())
        
        # 自動操作
        settings.setValue("auto/title", self.title_input.text())
        settings.setValue("auto/paste_x", self.paste_x_input.text())
        settings.setValue("auto/paste_y", self.paste_y_input.text())
        settings.setValue("auto/click_x", self.click_x_input.text())
        settings.setValue("auto/click_y", self.click_y_input.text())
        settings.setValue("auto/close_x", self.close_x_input.text())
        settings.setValue("auto/close_y", self.close_y_input.text())

        # 待機時間
        settings.setValue("auto/wait_paste_first", self.wait_paste_first_input.text())
        settings.setValue("auto/wait_close", self.wait_close_input.text())
        settings.setValue("auto/wait_paste_loop", self.wait_paste_loop_input.text())

    def load_settings(self):
        """保存された設定をGUIに読み込む"""
        print("設定を読み込んでいます...")
        settings = QSettings("MyCompany", "SerialAutoInput")

        # ROI (デフォルト値付きで読み込み)
        self.roi_x1_input.setText(settings.value("roi/x1", "100"))
        self.roi_y1_input.setText(settings.value("roi/y1", "200"))
        self.roi_x2_input.setText(settings.value("roi/x2", "600"))
        self.roi_y2_input.setText(settings.value("roi/y2", "250"))
        
        # 前処理
        self.threshold_slider.setValue(int(settings.value("preprocess/threshold", 140)))
        self.invert_checkbox.setChecked(settings.value("preprocess/invert", True, type=bool))
        self.resize_combo.setCurrentIndex(int(settings.value("preprocess/resize_idx", 2)))
        self.morph_combo.setCurrentIndex(int(settings.value("preprocess/morph_idx", 1)))

        # 自動操作
        self.title_input.setText(settings.value("auto/title", "DBSCGFW"))
        self.paste_x_input.setText(settings.value("auto/paste_x", "300"))
        self.paste_y_input.setText(settings.value("auto/paste_y", "440"))
        self.click_x_input.setText(settings.value("auto/click_x", "1000"))
        self.click_y_input.setText(settings.value("auto/click_y", "860"))
        self.close_x_input.setText(settings.value("auto/close_x", "0"))
        self.close_y_input.setText(settings.value("auto/close_y", "0"))

        # 待機時間
        self.wait_paste_first_input.setText(settings.value("auto/wait_paste_first", "1.5"))
        self.wait_close_input.setText(settings.value("auto/wait_close", "3.0"))
        self.wait_paste_loop_input.setText(settings.value("auto/wait_paste_loop", "3.0"))  

        print("設定の読み込み完了。")
    # ▲▲▲ 追加 ▲▲▲

    # ▼▼▼ 修正: closeEvent (ブロッキングI/Oスレッドの wait() を削除) ▼▼▼
    def closeEvent(self, event):
        print("終了処理を開始...")
        
        # 1. 設定を保存
        self.save_settings()
        
        # 2. 全スレッドに停止フラグを送信
        print("カメラスレッドに停止リクエスト...")
        self.camera_thread.stop() 
        
        print("座標スレッドに停止リクエスト...")
        self.coord_thread.stop()
        
        # --- 安全なスレッドの待機 ---
        # msleepベースのスレッドはwait()しても安全
        if self.ocr_thread and self.ocr_thread.isRunning():
            print("OCRスレッドを停止中...")
            self.ocr_thread.stop()
            self.ocr_thread.wait(1000) # msleepベースなので安全
        
        # time.sleepベースのスレッドもwait()しても安全
        if self.batch_thread and self.batch_thread.isRunning():
            print("バッチ処理を中断しています...")
            self.batch_thread.stop()
            self.batch_thread.wait(2000) # time.sleepベースなので安全

        # --- デッドロックの原因となる wait() を削除 ---
        # self.camera_thread.wait(1000) 
        # self.coord_thread.wait(1000)
        # 上記2行は cap.read() や pygetwindow() でブロッキングするため呼び出さない
        
        print("スレッドに停止リクエストを送信しました。即時終了します。")
        event.accept()
    # ▲▲▲ 修正 ▲▲▲

# --- プログラム実行 ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())