# STEP 1: Import the necessary modules.
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import numpy as np

mp_drawing = mp.solutions.drawing_utils
myPose = mp.solutions.pose
mp_drawing_styles = mp.solutions.drawing_styles
model_path="pose_landmarker_full.task"

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


# STEP 2: Create an PoseLandmarker object.
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE)

detector = vision.PoseLandmarker.create_from_options(options)

## 逐帧处理函数
def process_frame(frame):
    # 注意1: mp本身可以直接从文件中读取数据，例如 image = mp.Image.create_from_file("image.jpg")，但是默认读取带透明通道需要处理一下，如果使用的话需要单独处理一下
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    # STEP 4: Detect pose landmarks from the input image.
    detection_result = detector.detect(mp_image)

    # STEP 5: Draw pose landmarks on the input image.
    landmarks = detection_result.pose_landmarks
    pose_landmarks = landmark_pb2.NormalizedLandmarkList()
    if len(landmarks) == 0:
        return frame
    # 注意2 landmarks本身是识别多个人的对话框，所以需要遍历，这里只有一个人，所以直接取第1个，即landmarks[0]
    pose_landmarks.landmark.extend([
    landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z,visibility=landmark.visibility) for landmark in landmarks[0]
    ])
    image = mp_image.numpy_view()
    annotated_image = np.copy(image)
    # 注意3 draw_landmarks的入参图片不识别带透明通道的图片，如果包含需要单独处理一下，所以本程序在读取的时候特别处理为SRGB格式
    mp_drawing.draw_landmarks(annotated_image,pose_landmarks,myPose.POSE_CONNECTIONS, landmark_drawing_spec=mp_drawing_styles
                    .get_default_pose_landmarks_style())
    return annotated_image


# 导入opencv-python
import time

# 展示视频地址:
video_path = './dance.mov'
cap = cv2.VideoCapture(video_path)

# 无限循环，直到break被触发
while cap.isOpened():
    
    # 获取画面
    success, frame = cap.read()
    
    if not success: # 如果获取画面不成功，则退出
        print('视频获取不成功，退出')
        break
    
    ## 逐帧处理
    frame = process_frame(frame)
    
    # 展示处理后的三通道图像
    cv2.imshow('my_window',frame)
    
    key_pressed = cv2.waitKey(60) # 每隔多少毫秒毫秒，获取键盘哪个键被按下
    # print('键盘上被按下的键：', key_pressed)

    if key_pressed in [ord('q'),27]: # 按键盘上的q或esc退出（在英文输入法下）
        break
    
# 关闭摄像头
cap.release()

# 关闭图像窗口
cv2.destroyAllWindows()