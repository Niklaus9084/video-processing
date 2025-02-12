import cv2
import numpy as np

# 定义颜色范围（HSV颜色空间）
lower_red = np.array([0, 100, 100])
upper_red = np.array([10, 255, 255])

lower_blue = np.array([110, 50, 50])
upper_blue = np.array([130, 255, 255])

lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([30, 255, 255])

# 初始化特征点列表
red_points = []
blue_points = []
yellow_points = []

# 打开视频文件
cap = cv2.VideoCapture('input_video.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 将帧转换为HSV颜色空间
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 创建颜色掩码
    mask_red = cv2.inRange(hsv, lower_red, upper_red)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # 查找轮廓
    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_yellow, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 绘制特征点
    for contour in contours_red:
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius)
        cv2.circle(frame, center, radius, (0, 0, 255), 2)
        red_points.append(center)

    for contour in contours_blue:
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius)
        cv2.circle(frame, center, radius, (255, 0, 0), 2)
        blue_points.append(center)

    for contour in contours_yellow:
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius)
        cv2.circle(frame, center, radius, (0, 255, 255), 2)
        yellow_points.append(center)

    # 显示帧
    cv2.imshow('Frame', frame)

    # 按'q'键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()

# 绘制特征点轨迹图
def draw_trajectory(points, color):
    trajectory = np.array(points, np.int32)
    trajectory = trajectory.reshape((-1, 1, 2))
    cv2.polylines(frame, [trajectory], False, color, 2)

# 绘制红色特征点轨迹
draw_trajectory(red_points, (0, 0, 255))

# 绘制蓝色特征点轨迹
draw_trajectory(blue_points, (255, 0, 0))

# 绘制黄色特征点轨迹
draw_trajectory(yellow_points, (0, 255, 255))

# 显示特征点轨迹图
cv2.imshow('Trajectory', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
