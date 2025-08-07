import socket
import time
import json

import numpy as np
import pyrealsense2 as rs
import cv2
import subprocess
from gpt_english import *

import re
def pixel_to_world(u, v, depth, intrinsic_matrix):
    """
    将像素坐标转换为世界坐标

    参数:
    u, v: 像素坐标
    depth: 深度值(单位:米)
    intrinsic_matrix: 3x3相机内参矩阵

    返回:
    world_point: 世界坐标系下的3D点坐标[X, Y, Z]
    """
    # 获取相机内参
    fx = intrinsic_matrix[0, 0]  # 焦距x
    fy = intrinsic_matrix[1, 1]  # 焦距y
    cx = intrinsic_matrix[0, 2]  # 主点x
    cy = intrinsic_matrix[1, 2]  # 主点y

    # 反投影
    X = (u - cx) * depth / fx
    Y = (v - cy) * depth / fy
    Z = depth

    return np.array([X, Y, Z])
def camera_to_global_coordinate(camera_point, robot_pose):
    """
    将相机坐标系下的3D点转换为全局坐标系下的位置和朝向

    参数:
    camera_point: 相机坐标系下的3D点 [X_cam, Y_cam, Z_cam]
    robot_pose: 机器人在全局坐标系下的位姿 [x_robot, y_robot, theta_robot]

    返回:
    global_pose: 全局坐标系下的位置和朝向 [x, y, theta]
    """
    # 1. 首先将相机坐标系下的点转换到机器人局部坐标系
    # 相机的Z轴朝前，X轴朝右，Y轴朝下
    x_local = camera_point[2]  # 相机的Z轴对应局部X轴
    y_local = -camera_point[0]  # 相机的-X轴对应局部Y轴

    # 2. 获取机器人在全局坐标系下的位置和朝向
    x_robot = robot_pose[0]
    y_robot = robot_pose[1]
    theta_robot = robot_pose[2]  # 机器人朝向角

    # 3. 构建从局部到全局的转换矩阵
    rotation_matrix = np.array([
        [np.cos(theta_robot), -np.sin(theta_robot)],
        [np.sin(theta_robot), np.cos(theta_robot)]
    ])

    # 4. 将局部坐标转换到全局坐标
    local_point = np.array([x_local, y_local])
    global_point = rotation_matrix @ local_point

    # 5. 加上机器人的全局位置偏移
    x_global = global_point[0] + x_robot
    y_global = global_point[1] + y_robot

    # 6. 计算全局朝向角（相对于全局坐标系）
    theta_global = np.arctan2(y_global - y_robot, x_global - x_robot)

    return np.array([x_global, y_global, theta_global])




intrinsic = np.array([
                [606,   0, 335],  # fx, 0, cx
                [  0, 606, 239.00863647460938],  # 0, fy, cy
                [  0,   0,   1]
            ])

class RealSenseCamera:
    def __init__(self, width=640, height=480, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        
        # 初始化相机
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # 配置图像流
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        
        # 创建对齐对象
        self.align = rs.align(rs.stream.color)
        self.is_running = False

    def start(self):
        """启动相机"""
        if not self.is_running:
            try:
                profile = self.pipeline.start(self.config)

                depth_profile = profile.get_stream(rs.stream.depth)
                depth_canshu = depth_profile.as_video_stream_profile().get_intrinsics()

                rgb_profile = profile.get_stream(rs.stream.color)
                rgb_canshu = rgb_profile.as_video_stream_profile().get_intrinsics()
                
                print(depth_canshu.fx,depth_canshu.fy,depth_canshu.ppx,depth_canshu.ppy)
                # exit()
                self.is_running = True
                # 预热相机
                print("等待相机预热...")
                for _ in range(30):
                    self.pipeline.wait_for_frames()
                print("相机已准备就绪")
            except Exception as e:
                print(f"启动相机失败: {str(e)}")
    
    def stop(self):
        """停止相机"""
        if self.is_running:
            self.pipeline.stop()
            self.is_running = False
        
    def get_frame(self):
        """获取单帧对齐的RGBD图像
        返回值:
            color_image: RGB图像 (numpy array)
            depth_image: 深度图像 (numpy array)
            depth_colormap: 深度图的彩色可视化 (numpy array)
        """
        if not self.is_running:
            print("相机未启动，请先启动相机。")
            return None, None, None
        
        try:
            # 获取帧
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            
            # 获取对齐后的帧
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                print("未能获取完整的帧")
                return None, None, None
                
            # 转换为numpy数组
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())            
            # exit()
            # depth_image = np.asanyarray(depth_frame.get_data())
            # print(np.min(depth_image),np.max(depth_image))
            
            # 创建深度图像的彩色映射
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03), 
                cv2.COLORMAP_JET
            )
            
            return color_image, depth_image, depth_colormap
            
        except Exception as e:
            print(f"获取帧失败: {str(e)}")
            return None, None, None
            
    def __del__(self):
        """清理资源"""
        self.stop()







host, port = '192.168.10.10', 31001 # host为底盘ip，port为端口

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect((host, port))
print("Connected to the server.")

# 前后左右
points = {
    "forward": '/api/joy_control?angular_velocity=0&linear_velocity=1',
    "back": '/api/joy_control?angular_velocity=0.0&linear_velocity=-0.25',
    "left": '/api/joy_control?angular_velocity=1.5 &linear_velocity=0.0',
    "right": '/api/joy_control?angular_velocity=-1.5&linear_velocity=0.0'
} # 线速度、角速度可自行更改
direction = "forward"
status = '/api/robot_status'






def localpolicy(camera: RealSenseCamera,object_name):

    client.send(status.encode("utf-8"))
    data = client.recv(1024).decode()
    print('222',data)

    data_json = json.loads(data)
    results = data_json['results']
    # print(data_json['results'])
    # print(results['current_pose'])
    pose = results['current_pose']
# print(pose['theta'])
    theta = results['current_pose']['theta']
    # print(pose['theta'])
    print(theta)
    x,y = pose['x'],pose['y']


    for i in range(1,5):
        rgb, depth,depth_colormap = camera.get_frame()
        cv2.imwrite('rgb.png',rgb)
        cv2.imwrite("Depth.png", depth_colormap)

        result = gpt_image(object_name=object_name)
        print(result)
        if ('yes' in result) or ('Yes' in result):
            bofang('I have seen'+object_name+', now go to the final goal position')


            response_text = gpt_point(object_name)

            points = []
            
            # 匹配 [(x1, y1), (x2, y2), ...] 格式
            pattern1 = r'\[\s*\(([^)]+)\)\s*(?:,\s*\(([^)]+)\)\s*)*\]'
            match1 = re.search(pattern1, response_text)
            
            if match1:
                # 提取完整的列表字符串
                list_str = match1.group(0)
                try:
                    # 使用ast.literal_eval安全解析
                    import ast
                    parsed_points = ast.literal_eval(list_str)
                    if isinstance(parsed_points, list):
                        points = [(float(x), float(y)) for x, y in parsed_points]
                except (ValueError, SyntaxError):
                    pass
    
            center_x = round(sum(p[0] for p in points) / len(points), 2)
            center_y = round(sum(p[1] for p in points) / len(points), 2)

            print(f"中点坐标: ({center_x}, {center_y})")
            point = [center_x,center_y]
            x = point[0]
            y = point[1]
            pixel_coord = (int(x),int(y))  # 像素坐标 (u,v)
            # pixel_coord = (479,50) 
            depth_value = depth[int(y),int(x)]/1000

            
            return [center_x,center_y], rgb, depth
            
        bofang('I didn\'t see'+object_name+', now begin to perform panoramic rotation')
        point = '/api/move?location=' + f'{x},{y},{theta-0.8*i}' # location为地图上的x,y,theta
        print(point)
        client.send(point.encode("utf-8"))
        data = client.recv(1024).decode()
        print(data)
        time.sleep(8)
    return None,rgb,depth





from luyin import record_audio





instructions = record_audio(duration=5)





thinking = gpt_thinking(instructions)


object_name = gpt_object(thinking)



if 'meeting room' in instructions:
    room = 'meeting room'
elif 'workstations' in instructions:
    room = 'workstation'
elif 'room' in instructions:
    room = 'tea room'
elif 'balcony' in instructions:
    room = 'balcony'
else:
    room = gpt_room(object_name)


print(room)
# exit()

bofang('In this 3D scene, I discovered scenes such as tea room, meeting room, workstations, and balcony. After previous thinking, the most likely area where '+object_name+' exists is '+room+' ，Now I\'m going to '+room+' to search for it.')



# time.sleep(10)
# exit()
camera = RealSenseCamera()
camera.start()

# color_img, depth_img, depth_colormap = camera.get_frame()


if 'workstation' in room:

    x_min, y_min = 3.8, -7.6
    x_max, y_max = 5.0, -8.64

elif ('tea' in room) or ('Tea' in room):
    x_min, y_min = -1.591, -24.087
    x_max, y_max = -0.171, -23.307


elif 'balcony' in room:
    x_min, y_min = 1.989, 0.493
    x_max, y_max = 3.189, -0.247

else:
    x_min, y_min = 3.8, -7.6
    x_max, y_max = 5.0, -8.64
random_point = np.array(
    [
        np.random.uniform(x_min,x_max),
        np.random.uniform(y_min,y_max)
    ]
)

print(random_point)
# exit()



robopoint = '/api/move?location=' + f'{random_point[0]},{random_point[1]},6.0169' # location为地图上的x,y,theta
print(robopoint)
client.send(robopoint.encode("utf-8"))
data = client.recv(1024).decode()
print(data)
time.sleep(2)



# x = input()
# time.sleep(40)

while 1:

    client.send(status.encode("utf-8"))
    data = client.recv(1024).decode()
    print('1111',data)
    try:
        data_json = json.loads(data)
        results = data_json['results']
        print(data_json['results'])
        print(results['move_status'])
        if results['move_status']=='succeeded':
            bofang('I have successfully arrive '+room+' now begin to explore in this scene to find '+object_name)
            time.sleep(2)
            break
        
        time.sleep(2)
    except:

        time.sleep(2)

client.send(status.encode("utf-8"))
data = client.recv(1024).decode()
print('1111',data)

time.sleep(1)
# time.sleep(65)
# client.send(status.encode("utf-8"))
# data = client.recv(1024).decode()
# print(data)
# data_json = json.loads(data)
# results = data_json['results']
# print(data_json['results'])





#local policy
point,rgb,depth_img = localpolicy(camera=camera,object_name=object_name)




if point is not None:


    client.send(status.encode("utf-8"))
    data = client.recv(1024).decode()
    print('333',data)


    time.sleep(1)
    client.send(status.encode("utf-8"))
    data = client.recv(1024).decode()
    print('333',data)

    data_json = json.loads(data)
    results = data_json['results']
    print(data_json['results'])
    print(results['current_pose'])
    pose = results['current_pose']
    print(pose['theta'])



    x = point[0]
    y = point[1]
    pixel_coord = (int(x),int(y))  # 像素坐标 (u,v)
    # pixel_coord = (479,50) 
    depth_value = depth_img[int(y),int(x)]/1000

    
        
    tolerance = 0.3
    if ('coffee machine' in object_name) or ('frigerator' in object_name) or ('windows' in object_name) in object_name:
        if depth_value>5:
            depth_value-=2
        elif depth_value>4:
            depth_value-=0.9
        else:
            depth_value-=0.5
        tolerance = 0.2
    print(depth_img)
    print(depth_value-1) # 深度值(米)
    # exit()
    world_point = pixel_to_world(pixel_coord[0], pixel_coord[1], depth_value, intrinsic)
    print(f"World coordinates: X={world_point[0]:.3f}, Y={world_point[1]:.3f}, Z={world_point[2]:.3f}")



    robot_pose= np.array([pose['x'], pose['y'], pose['theta']])
    world = camera_to_global_coordinate(world_point,robot_pose)
    print(world)
    robopoint = '/api/move?location=' + f'{world[0]},{world[1]},{world[2]}&occupied_tolerance={tolerance}' # location为地图上的x,y,theta
    print(robopoint)
    client.send(robopoint.encode("utf-8"))
    data = client.recv(1024).decode()
    print(data)
    # exit()
    while 1:

        client.send(status.encode("utf-8"))
        data = client.recv(1024).decode()
        print('1111',data)
        try:
            data_json = json.loads(data)
            results = data_json['results']
            print(data_json['results'])
            print(results['move_status'])
            if results['move_status']=='succeeded':
                bofang('I have successfully find the goal '+object_name)
                time.sleep(2)
                break
            
            time.sleep(2)
        except:

            time.sleep(2)


    exit()



# 继续探索