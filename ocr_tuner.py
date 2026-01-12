import sys
import cv2
import pytesseract
import re
import numpy as np

# --- Tesseract-OCRのパス設定 ---
try:
    pytesseract.pytesseract.tesseract_cmd = r'D:\OCR\tesseract.exe'
    pytesseract.get_tesseract_version()
    print("Tesseract-OCR 連携OK.")
    print("-----------------------------------------")
    print(" Tesseract OCR Tuner ")
    print("-----------------------------------------")
    print(" [マウス] ドラッグでROIを指定")
    print(" [トラックバー] 二値化のしきい値を調整")
    print(" [Spaceキー] 枠内のOCRを実行")
    print(" [r キー] ROIをリセット")
    print(" [q キー] 終了")
    print("-----------------------------------------")

except Exception as e:
    print(f"エラー: Tesseract-OCRが見つかりません。パスを確認してください。")
    sys.exit(-1)

# --- グローバル変数 ---
roi = (0, 0, 0, 0) # ROI (x, y, w, h)
drawing = False    # ドラッグ中フラグ
ix, iy = -1, -1    # ドラッグ開始点
threshold_value = 127 # 二値化しきい値の初期値

def update_threshold(val):
    """トラックバーが動かされたときのコールバック"""
    global threshold_value
    threshold_value = val

def mouse_callback(event, x, y, flags, param):
    """マウス操作のコールバック"""
    global ix, iy, drawing, roi

    if event == cv2.EVENT_LBUTTONDOWN:
        # ドラッグ開始
        drawing = True
        ix, iy = x, y
        roi = (0, 0, 0, 0) # リセット

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            # ドラッグ中。現在のフレーム(param)に矩形を描画
            frame_copy = param.copy()
            cv2.rectangle(frame_copy, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.imshow('Camera (Drag to select ROI)', frame_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        # ドラッグ終了
        drawing = False
        # ROIの座標を確定 (x, y, w, h)
        x1, y1 = min(ix, x), min(iy, y)
        x2, y2 = max(ix, x), max(iy, y)
        roi = (x1, y1, x2 - x1, y2 - y1)

def run_ocr(image):
    """画像に対してOCRを実行し、結果を表示する関数"""
    print(f"\n--- OCR実行 (しきい値: {threshold_value}) ---")
    
    # Tesseractの設定
    # PSM 6: 均一なテキストブロックとして認識
    # whitelist: 英大文字と数字に限定
    config = '--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    
    try:
        raw_text = pytesseract.image_to_string(image, lang='eng', config=config)
        
        # 整形
        processed_code = re.sub(r'[^A-Z0-9]', '', raw_text.upper()) 
        
        print(f"  RAW    : '{raw_text.strip()}'")
        print(f"  Processed: '{processed_code}'")

    except Exception as e:
        print(f"  Pytesseractエラー: {e}")

# --- メイン処理 ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("エラー: カメラ 0 を開けません。")
    exit()

# ウィンドウの作成と設定
cv2.namedWindow('Camera (Drag to select ROI)')
cv2.namedWindow('Processed (for OCR)')

# マウスコールバックを設定 (paramにメインループのフレームを渡す)
# この時点では param は None だが、メインループ内で setMouseCallback を呼び直す
cv2.setMouseCallback('Camera (Drag to select ROI)', mouse_callback)

# トラックバー（二値化しきい値）を作成
cv2.createTrackbar('Threshold', 'Processed (for OCR)', threshold_value, 255, update_threshold)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # マウスコールバックに現在のフレームを渡す (ドラッグ中の描画用)
    cv2.setMouseCallback('Camera (Drag to select ROI)', mouse_callback, frame)

    # 現在のROIを取得
    x, y, w, h = roi

    # メインのカメラ映像
    frame_display = frame.copy()

    # ROIが設定されていれば、処理を実行
    if w > 0 and h > 0:
        # 1. ROIを描画
        cv2.rectangle(frame_display, (x, y), (x + w, y + h), (0, 0, 255), 2) # 確定したROIは赤色

        # 2. ROI領域を切り抜き
        roi_frame = frame[y:y+h, x:x+w]
        
        # 3. グレースケール化
        gray_frame = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        
        # 4. 二値化 (トラックバーの値を使用)
        # THRESH_BINARY_INV: しきい値より大きいピクセルを0(黒)に (白背景・黒文字の場合)
        _, binary_frame = cv2.threshold(gray_frame, threshold_value, 255, cv2.THRESH_BINARY_INV)
        
        # 処理後の画像を 'Processed' ウィンドウに表示
        cv2.imshow('Processed (for OCR)', binary_frame)
    else:
        # ROIが未選択の場合は、ダミー画像を表示
        dummy_img = np.zeros((100, 400), dtype=np.uint8)
        cv2.putText(dummy_img, 'Drag ROI on Camera window', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.imshow('Processed (for OCR)', dummy_img)


    # メインのカメラ映像を表示
    cv2.imshow('Camera (Drag to select ROI)', frame_display)

    # キー入力の待受
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        # 'q' で終了
        break
    elif key == ord('r'):
        # 'r' でROIをリセット
        roi = (0, 0, 0, 0)
        print("ROI Reset.")
    elif key == ord(' '):
        # 'スペース' でOCR実行
        if w > 0 and h > 0:
            # 処理済みの 'binary_frame' をOCRにかける
            run_ocr(binary_frame)
        else:
            print("OCR 実行不可: まずはROIをドラッグで指定してください。")

cap.release()
cv2.destroyAllWindows()