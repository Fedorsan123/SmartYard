import os
import cv2
import json
import time
import numpy as np
from datetime import datetime
from ultralytics import YOLO
from sort import Sort  # ваш sort.py + kalman_filter.py

# 1. Параметры
MODEL_PATH      = 'yolov8l.pt'
STREAM_URL      = "http://86.127.212.219/cgi-bin/faststream.jpg?stream=half&fps=15"
CONF_THRESHOLD  = 0.1
IOU_NMS         = 0.45
CLASS_CAR       = 2    # COCO-класс “car”
FPS             = 15
STOP_SEC        = 10   # секунд для фиксации spot
STOP_FRAMES     = FPS * STOP_SEC
MOVE_THRESH     = 5.0  # пиксели
IOU_SPOT_THRESH = 0.5  # IoU порог
JSON_PATH       = 'parking_state.json'
JSON_UPDATE_SEC = 2    # раз в 2 секунды

# 2. Загрузка ранее сохранённых spot’ов (только id и bbox)
parking_spots = []      # { 'id':…, 'bbox':(x1,y1,x2,y2) }
next_id = 1
if os.path.isfile(JSON_PATH):
    with open(JSON_PATH) as f:
        data = json.load(f)
        for spot in data.get('spots', []):
            parking_spots.append({
                'id': spot['id'],
                'bbox': tuple(spot['bbox'])
            })
    if parking_spots:
        next_id = max(s['id'] for s in parking_spots) + 1

# 3. Модель и трекер
model   = YOLO(MODEL_PATH)
cap     = cv2.VideoCapture(STREAM_URL)
tracker = Sort(max_age=5, min_hits=3)

history          = {}  # tid -> (cx,cy)
stationary_count = {}  # tid -> подряд стоячих кадров
track_boxes      = {}  # tid -> list of bbox для усреднения
last_json_write  = 0
frame_idx        = 0

def iou(a, b):
    xA, yA = max(a[0],b[0]), max(a[1],b[1])
    xB, yB = min(a[2],b[2]), min(a[3],b[3])
    inter = max(0,xB-xA)*max(0,yB-yA)
    areaA = (a[2]-a[0])*(a[3]-a[1])
    areaB = (b[2]-b[0])*(b[3]-b[1])
    return inter / (areaA + areaB - inter + 1e-6)

while True:
    ret, frame = cap.read()
    if not ret: break
    frame_idx += 1

    # 4. ROI: оставляем нижние 30% (отрезаем 70% сверху)
    h,w = frame.shape[:2]
    y0  = int(h * 0.7)
    roi = frame[y0: , :]

    # 5. Детекция
    results = model.predict(
        roi, conf=CONF_THRESHOLD, iou=IOU_NMS,
        classes=[CLASS_CAR], stream=False, imgsz=1024
    )
    dets = []
    for r in results:
        for b in r.boxes:
            x1,y1,x2,y2 = b.xyxy[0].cpu().numpy()
            dets.append([x1,y1,x2,y2,float(b.conf[0])])
    dets = np.array(dets) if dets else np.zeros((0,5))

    # 6. Трекинг
    tracks = tracker.update(dets)

    # 7. Поиск новых spot’ов (N секунд стоянки)
    for x1,y1,x2,y2,tid in tracks:
        tid = int(tid)
        cx,cy = (x1+x2)/2, (y1+y2)/2
        prev = history.get(tid)
        if prev:
            dist = np.hypot(cx-prev[0], cy-prev[1])
            if dist < MOVE_THRESH:
                stationary_count[tid] = stationary_count.get(tid,0) + 1
                track_boxes.setdefault(tid,[]).append((x1,y1,x2,y2))
            else:
                stationary_count[tid] = 0
                track_boxes[tid] = []
        else:
            stationary_count[tid] = 0
            track_boxes[tid] = []
        history[tid] = (cx,cy)

        if stationary_count[tid] == STOP_FRAMES:
            arr = np.array(track_boxes[tid])
            x1a,y1a,x2a,y2a = arr.mean(axis=0)
            bbox = (int(x1a),int(y1a),int(x2a),int(y2a))
            parking_spots.append({'id':next_id,'bbox':bbox})
            print(f"🟡 New spot #{next_id} at {bbox}")
            next_id += 1

    # 8. Визуализация:
    #   a) сначала все сохранённые spot’ы (жёлтым),
    #      если в spot сейчас машина (IoU>порог) — рисуем красным поверх
    occupied_count = 0
    for spot in parking_spots:
        x1,y1,x2,y2 = map(int, spot['bbox'])
        # проверяем занятость
        is_occ = any(iou(spot['bbox'],tuple(t[:4]))>IOU_SPOT_THRESH
                     for t in tracks)
        if is_occ: occupied_count += 1
        # рисуем жёлтым
        cv2.rectangle(roi,(x1,y1),(x2,y2),(0,255,255),2)
        # если занято — поверх красным
        if is_occ:
            cv2.rectangle(roi,(x1,y1),(x2,y2),(0,0,255),2)

    #   b) рисуем все текущие bbox машин, которые ещё не spot
    for x1,y1,x2,y2,tid in tracks:
        x1i,y1i,x2i,y2i = map(int,(x1,y1,x2,y2))
        cv2.rectangle(roi,(x1i,y1i),(x2i,y2i),(0,255,0),2)
        cv2.putText(roi,f"ID{int(tid)}",(x1i,y1i-5),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)

    cv2.imshow("Parking Detection", roi)
    if cv2.waitKey(1)&0xFF==27: break

    # 9. Обновляем JSON
    now = time.time()
    if now - last_json_write > JSON_UPDATE_SEC:
        state = {
            'timestamp': datetime.utcnow().isoformat()+'Z',
            'total_spots': len(parking_spots),
            'occupied': occupied_count,
            'free': len(parking_spots)-occupied_count,
            'spots': [
                {'id':s['id'],'bbox':list(s['bbox']),
                 'occupied': any(iou(s['bbox'],tuple(t[:4]))>IOU_SPOT_THRESH
                                 for t in tracks)
                } for s in parking_spots
            ]
        }
        with open(JSON_PATH,'w') as f:
            json.dump(state,f,indent=2)
        last_json_write = now

cap.release()
cv2.destroyAllWindows()
