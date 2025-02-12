import cv2
import numpy as np

def define_color_ranges():
    """定义颜色范围（HSV颜色空间）"""
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])

    lower_blue = np.array([110, 50, 50])
    upper_blue = np.array([130, 255, 255])

    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    return lower_red, upper_red, lower_blue, upper_blue, lower_yellow, upper_yellow

def initialize_feature_points():
    """初始化特征点列表"""
    return [], [], []

def process_frame(frame, lower_red, upper_red, lower_blue, upper_blue, lower_yellow, upper_yellow, color_choices):
    """处理每一帧图像，查找并绘制特征点"""
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
    red_points, blue_points, yellow_points = [], [], []
    for color_choice in color_choices:
        if color_choice == 'red':
            for contour in contours_red:
                (x, y), radius = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))
                radius = int(radius)
                cv2.circle(frame, center, radius, (0, 0, 255), 2)
                red_points.append(center)
        elif color_choice == 'blue':
            for contour in contours_blue:
                (x, y), radius = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))
                radius = int(radius)
                cv2.circle(frame, center, radius, (255, 0, 0), 2)
                blue_points.append(center)
        elif color_choice == 'yellow':
            for contour in contours_yellow:
                (x, y), radius = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))
                radius = int(radius)
                cv2.circle(frame, center, radius, (0, 255, 255), 2)
                yellow_points.append(center)

    return frame, red_points, blue_points, yellow_points

def draw_trajectory(frame, points, color):
    """绘制特征点轨迹图"""
    trajectory = np.array(points, np.int32)
    trajectory = trajectory.reshape((-1, 1, 2))
    cv2.polylines(frame, [trajectory], False, color, 2)

def process_video(video_path, color_choices):
    """处理视频文件"""
    # 定义颜色范围
    lower_red, upper_red, lower_blue, upper_blue, lower_yellow, upper_yellow = define_color_ranges()

    # 初始化特征点列表
    red_points, blue_points, yellow_points = initialize_feature_points()

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 获取视频的总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 创建窗口并调整大小
    cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Frame', 640, 480)

    # 创建进度条
    cv2.createTrackbar('Progress', 'Frame', 0, total_frames, lambda x: None)

    while cap.isOpened():
        # 获取进度条的当前位置
        current_frame = cv2.getTrackbarPos('Progress', 'Frame')

        # 设置视频的当前帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

        ret, frame = cap.read()
        if not ret:
            break

        # 处理每一帧图像
        frame, red_points, blue_points, yellow_points = process_frame(frame, lower_red, upper_red, lower_blue, upper_blue, lower_yellow, upper_yellow, color_choices)

        # 显示帧
        cv2.imshow('Frame', frame)

        # 按'q'键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    # cap.release()
    # cv2.destroyAllWindows()

    # 绘制特征点轨迹图
    if frame is not None:
        for color_choice in color_choices:
            if color_choice == 'red':
                draw_trajectory(frame, red_points, (0, 0, 255))
            elif color_choice == 'blue':
                draw_trajectory(frame, blue_points, (255, 0, 0))
            elif color_choice == 'yellow':
                draw_trajectory(frame, yellow_points, (0, 255, 255))

        # 创建窗口并调整大小
        cv2.namedWindow('Trajectory', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Trajectory', 640, 480)

        # 显示特征点轨迹图
        cv2.imshow('Trajectory', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    """主函数，统一入口"""
    video_path = 'input_video.mp4'
    color_choices = input("请选择要处理的颜色（red/blue/yellow，多个颜色用逗号分隔）：").lower().split(',')
    process_video(video_path, color_choices)

if __name__ == "__main__":
    main()