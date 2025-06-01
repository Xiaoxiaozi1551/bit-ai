import imageio
import cv2
from KCF import KCFTracker

# 定义全局变量
global start_x, start_y, end_x, end_y, drawing, frame_copy
drawing = False  # 是否正在绘制
start_x, start_y = -1, -1  # 矩形框的起始坐标
end_x, end_y = -1, -1  # 矩形框的结束坐标
frame_copy = None  # 用于绘制实时预览效果的帧的副本


def find_box(frame):
    global start_x, start_y, end_x, end_y, frame_copy

    # 在图像上绘制矩形框
    def draw_rectangle(event, x, y, flags, param):
        global start_x, start_y, end_x, end_y, drawing, frame_copy

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            start_x, start_y = x, y

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            end_x, end_y = x, y

        if drawing:
            frame_copy = frame.copy()
            cv2.rectangle(frame_copy, (start_x, start_y), (x, y), (0, 255, 0), 2)

    # 创建窗口并注册鼠标事件回调函数
    cv2.namedWindow("Frame")
    cv2.setMouseCallback("Frame", draw_rectangle)

    # 显示图像并等待用户绘制矩形框
    while True:
        if frame_copy is not None:
            cv2.imshow("Frame", frame_copy)
        else:
            cv2.imshow("Frame", frame)

        key = cv2.waitKey(1) & 0xFF

        # 按下 'r' 键重置矩形框
        if key == ord('r'):
            start_x, start_y, end_x, end_y = -1, -1, -1, -1
            frame_copy = None

        # 按下 'c' 键确认选择
        elif key == ord('c'):
            break

    # 关闭窗口
    cv2.destroyAllWindows()

    # 计算矩形框的位置和大小
    if start_x != -1 and start_y != -1 and end_x != -1 and end_y != -1:
        x = min(start_x, end_x)
        y = min(start_y, end_y)
        w = abs(end_x - start_x)
        h = abs(end_y - start_y)
        return (y + h // 2, x + w // 2), (h, w)
    else:
        return None, None


if __name__ == '__main__':
    # 读取视频
    reader = imageio.get_reader("bullying.mp4")
    first_frame = reader.get_data(0)

    # 在第一帧中找人脸
    position, img_size = find_box(first_frame)

    bbox = (position[1] - img_size[1] // 2, position[0] - img_size[0] // 2, img_size[1], img_size[0])

    # 初始化KCFTracker的KCF跟踪器
    sot = KCFTracker()
    sot.initial(first_frame, position, img_size)

    # 输出视频设置
    output_video_path = "1.mp4"
    fps = reader.get_meta_data()['fps']
    writer = imageio.get_writer(output_video_path, fps=fps)

    # 跟踪目标
    for _, frame in enumerate(reader):
        simple_position = sot.update(frame)
        # 绿色框
        cv2.rectangle(frame,
                      (int(simple_position[1] - img_size[1] // 2), int(simple_position[0] - img_size[0] // 2)),
                      (int(simple_position[1] + img_size[1] // 2), int(simple_position[0] + img_size[0] // 2)),
                      (0, 255, 0), 2)

        writer.append_data(frame)

    writer.close()
    print('ok')