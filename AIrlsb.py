import cv2
import os
import numpy as np
from insightface.app import FaceAnalysis

# 初始化模型
app = FaceAnalysis(
    name="buffalo_l",
    providers=['CPUExecutionProvider']
)

app.prepare(ctx_id=0)

# 已知人脸
known_faces = []
known_names = []

# 读取人脸库
faces_dir = "faces"

for file in os.listdir(faces_dir):

    path = os.path.join(faces_dir, file)

    img = cv2.imread(path)

    if img is None:
        continue

    faces = app.get(img)

    if len(faces) > 0:

        # 取第一张脸
        embedding = faces[0].embedding

        known_faces.append(embedding)

        name = os.path.splitext(file)[0]
        known_names.append(name)

        print(f"已加载: {name}")

# 打开摄像头
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # 检测人脸
    faces = app.get(frame)

    for face in faces:

        bbox = face.bbox.astype(int)

        x1, y1, x2, y2 = bbox

        embedding = face.embedding

        name = "Unknown"

        best_score = -1

        # 比对数据库
        for i, known_embedding in enumerate(known_faces):

            similarity = np.dot(
                embedding,
                known_embedding
            ) / (
                np.linalg.norm(embedding)
                * np.linalg.norm(known_embedding)
            )

            if similarity > best_score:
                best_score = similarity

                # 阈值
                if similarity > 0.45:
                    name = known_names[i]

        # 画框
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # 名字
        text = f"{name} {best_score:.2f}"

        cv2.putText(
            frame,
            text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    cv2.imshow("InsightFace AI", frame)

    # q退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()