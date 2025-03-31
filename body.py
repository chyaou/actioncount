import cv2
import mediapipe as mp
import numpy as np

# from PIL import Image, ImageFont, ImageDraw


# ------------------------------------------------
#   mediapipe的初始化
# ------------------------------------------------
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()


def get_angle(v1, v2):
    angle = np.dot(v1, v2) / (np.sqrt(np.sum(v1 * v1)) * np.sqrt(np.sum(v2 * v2)))
    angle = np.arccos(angle) / 3.14 * 180

    cross = v2[0] * v1[1] - v2[1] * v1[0]
    if cross < 0:
        angle = - angle
    return angle


# ------------------------------------------------
#   计算姿态
# ------------------------------------------------
def get_pos(keypoints):
    str_pose = ""
    # 计算左臂与水平方向的夹角
    keypoints = np.array(keypoints)
    v1 = keypoints[12] - keypoints[11]
    v2 = keypoints[13] - keypoints[11]
    angle_left_arm = get_angle(v1, v2)
    # 计算右臂与水平方向的夹角
    v1 = keypoints[11] - keypoints[12]
    v2 = keypoints[14] - keypoints[12]
    angle_right_arm = get_angle(v1, v2)
    # 计算左肘的夹角
    v1 = keypoints[11] - keypoints[13]
    v2 = keypoints[15] - keypoints[13]
    angle_left_elow = get_angle(v1, v2)
    # 计算右肘的夹角
    v1 = keypoints[12] - keypoints[14]
    v2 = keypoints[16] - keypoints[14]
    angle_right_elow = get_angle(v1, v2)

    if angle_left_arm < 0 and angle_right_arm < 0:
        str_pose = "LEFT_UP"  # 举左手
    elif angle_left_arm > 0 and angle_right_arm > 0:
        str_pose = "RIGHT_UP"  # 举右手
    elif angle_left_arm < 0 and angle_right_arm > 0:
        str_pose = "ALL_HANDS_UP"  # 举双手
        if abs(angle_left_elow) < 120 and abs(angle_right_elow) < 120:
            str_pose = "TRIANGLE"  # 双手举起摆三角形
    elif angle_left_arm > 0 and angle_right_arm < 0:
        str_pose = "NORMAL"  # 正常站立
        if abs(angle_left_elow) < 120 and abs(angle_right_elow) < 120:
            str_pose = "AKIMBO"  # 叉腰
    return str_pose


def process_frame(img):
    # start_time = time.time()
    h, w = img.shape[0], img.shape[1]  # 高和宽
    # 调整字体
    tl = round(0.005 * (img.shape[0] + img.shape[1]) / 2) + 1
    # tf = max(tl-1, 1)
    # BRG-->RGB
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 将RGB图像输入模型，获取 关键点 预测结果
    results = pose.process(img_RGB)
    keypoints = []
    keyappend = keypoints.append
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        for i in range(33):
            cx = int(results.pose_landmarks.landmark[i].x * w)
            cy = int(results.pose_landmarks.landmark[i].y * h)
            keyappend((cx, cy))  # 得到最终的33个关键点
    else:
        print("NO PERSON")
        struction = "NO PERSON"
        img = cv2.putText(img, struction, (25, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255, 255, 0),
                          3)
    # end_time = time.time()
    # process_time = end_time - start_time            # 图片关键点预测时间
    # fps = 1 / process_time                          # 帧率
    # colors = [[random.randint(0,255) for _ in range(3)] for _ in range(33)]
    # radius = [random.randint(8,15) for _ in range(33)]
    try:
        # for i in range(33):
        #     cx, cy = keypoints[i]
        #     img = cv2.circle(img, (cx, cy), radius[i], colors[i], -1)
        str_pose = get_pos(keypoints)  # 获取姿态
        cv2.putText(img, "POSE-{}".format(str_pose), (12, 100), cv2.FONT_HERSHEY_TRIPLEX,
                    tl / 6, (255, 0, 0), thickness=2)
    except:
        pass
    return img


# ------------------------------------------------
#   主函数
# ------------------------------------------------
def pre_image(image_path):
    # image = cv2.imread(image_path)
    # image = cv2.imdecode(np.fromfile(image, dtype=np.uint8), -1)
    # image = image.copy()
    frame = process_frame(image_path)
    return frame


def start_video():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()

        if ret:
            cv2.imshow('live', pre_image(frame))
            if cv2.waitKey(1) & 0xFF == 27:
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()

