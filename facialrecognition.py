import cv2
import os
import time
import queue
import threading
import numpy as np
from insightface.app import FaceAnalysis

# =========================
# 初始化 InsightFace
# =========================

app = FaceAnalysis(
    name="buffalo_s",
    providers=['CPUExecutionProvider']
)

app.prepare(
    ctx_id=0,
    det_size=(320, 320)
)

# =========================
# 加载人脸库
# =========================

known_embeddings = []
known_names = []

faces_dir = "faces"

for file in os.listdir(faces_dir):

    path = os.path.join(faces_dir, file)

    img = cv2.imread(path)

    if img is None:
        continue

    faces = app.get(img)

    if len(faces) == 0:
        print(f"未检测到人脸: {file}")
        continue

    embedding = faces[0].embedding

    known_embeddings.append(embedding)

    name = os.path.splitext(file)[0]

    known_names.append(name)

    print(f"已加载: {name}")

# =========================
# 全局变量
# =========================

frame_queue = queue.Queue(maxsize=1)

latest_result = []

running = True

ai_fps = 0

# =========================
# AI识别线程
# =========================

def recognize_worker():

    global latest_result
    global ai_fps
    global running

    while running:

        try:
            frame = frame_queue.get(timeout=1)

        except queue.Empty:
            continue

        start_time = time.time()

        results = []

        # AI检测
        faces = app.get(frame)

        for face in faces:

            bbox = face.bbox.astype(int)

            x1, y1, x2, y2 = bbox

            embedding = face.embedding

            name = "Unknown"

            best_similarity = 0

            # 数据库比对
            for i, known_embedding in enumerate(known_embeddings):

                similarity = np.dot(
                    embedding,
                    known_embedding
                ) / (
                    np.linalg.norm(embedding)
                    * np.linalg.norm(known_embedding)
                )

                if similarity > best_similarity:

                    best_similarity = similarity

                    if similarity > 0.55:
                        name = known_names[i]

            results.append({
                "box": (x1, y1, x2, y2),
                "name": name,
                "score": best_similarity
            })

        latest_result = results

        end_time = time.time()

        cost = end_time - start_time

        if cost > 0:
            ai_fps = 1 / cost

# =========================
# 启动AI线程
# =========================

thread = threading.Thread(
    target=recognize_worker,
    daemon=True
)

thread.start()

# =========================
# 打开摄像头
# =========================

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# =========================
# 主循环
# =========================

while True:

    ret, frame = cap.read()

    if not ret:
        print("摄像头读取失败")
        break

    # 缩小识别图像
    small_frame = cv2.resize(frame, (320, 240))

    # 放入队列
    if not frame_queue.full():
        frame_queue.put(small_frame)

    # =========================
    # 绘制识别结果
    # =========================

    for item in latest_result:

        x1, y1, x2, y2 = item["box"]

        name = item["name"]

        score = item["score"]

        # 坐标恢复
        x1 *= 2
        y1 *= 2
        x2 *= 2
        y2 *= 2

        # 画框
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # 名字
        text = f"{name} {score:.2f}"

        cv2.putText(
            frame,
            text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    # =========================
    # 显示FPS
    # =========================

    cv2.putText(
        frame,
        f"AI FPS: {ai_fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Multi-Thread InsightFace",
        (10, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2
    )

    # =========================
    # 显示窗口
    # =========================

    cv2.imshow(
        "InsightFace Threading",
        frame
    )

    # q退出
    if cv2.waitKey(1) & 0xFF == ord('q'):

        running = False

        break

# =========================
# 释放资源
# =========================

cap.release()

cv2.destroyAllWindows()