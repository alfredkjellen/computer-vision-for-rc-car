import cv2
import numpy as np
import json
import os
import math
import pandas as pd
import time

class CarDetector:
    def __init__(self, video_file=None, competitor_name='Player', flipped=True):
        if video_file is not None:
            self.video_file = video_file
            self.video_source = os.path.join("recordings", self.video_file)
            self.cap = cv2.VideoCapture(self.video_source)        
        else:
            self.video_file = 'live_feed'
            self.cap = cv2.VideoCapture(0)
        
        self.competitor_name = competitor_name
        self.flipped = flipped

        self.save_file = "videos.json"

        if not self.cap.isOpened():
            raise ValueError("Unable to open video source.")
 
        self.color_ranges = {
            "car_full": (np.array([0, 0, 230]), np.array([180, 50, 255])),}

        self.angle = 90
        self.position = None

        self.corners = None
        self.obstacles = None
        self.finish_line = None

        self.pixels_per_square = 150
        self.horizontal_squares = 3
        self.vertical_squares = 4
        self.width = self.pixels_per_square * self.horizontal_squares
        self.height = self.pixels_per_square * self.vertical_squares

        self.centimeters_per_square = 50
        self.scale_factor = self.pixels_per_square / self.centimeters_per_square 

        self.started = False
        self.finished = False

    def read_capture(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        frame = cv2.flip(frame, -1) if self.flipped else frame
        return frame

    def pressed(self, key):
        return cv2.waitKey(10) & 0xFF == ord(key)
    
    def put_text(self, frame, text, line=0):
        x = 5
        y = (line + 1) * 15
        cv2.putText(frame, text, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1, cv2.LINE_AA)


    def load_json(self, video_name, file_name):
        if os.path.exists(file_name):
            with open(file_name, "r") as file:
                data = json.load(file)
                return data.get(video_name)
        else:
            return None

    def select_corners(self):
        corners = []
        while True:
            #ret, frame = self.cap.read()
            frame = self.read_capture()
            if frame is None:
                print("Could not read frame")
                self.cap.release()
                cv2.destroyAllWindows()
                return
 
            cv2.imshow("Select Corners", frame)
            cv2.setMouseCallback("Select Corners", self.select_points, corners)

            print("Click to select corners.")
            print("Press 'n' to skip to the next frame if the corners aren't visible.")

            while len(corners) < 4:
                cv2.imshow("Select Corners", frame)
                if self.pressed('n'):
                    self.points = []
                    break
                elif self.pressed('q'):
                    print("Exiting...")
                    self.cap.release()
                    cv2.destroyAllWindows()
                    return

            if len(corners) == 4:
                cv2.destroyWindow("Select Corners")
                self.corners = corners
                return                                


    def calibrate_scale(self, real_distance_cm):
        while True:
            frame = self.get_transformed_frame()
            cv2.imshow("Calibrate Scale", frame)
            points = []

            def on_mouse(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    points.append((x, y))
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                    cv2.imshow("Calibrate Scale", frame)
                    if len(points) == 2:
                        cv2.line(frame, points[0], points[1], (255, 0, 0), 2)
                        cv2.imshow("Calibrate Scale", frame)

            cv2.setMouseCallback("Calibrate Scale", on_mouse)
            
            print("Select two points representing the real-world distance.")
            print("Press 'c' to confirm or 'r' to reset.")

            while len(points) < 2:
                if self.pressed('n'):
                    points = []
                    break

            if len(points) == 2:
                pixel_distance = np.linalg.norm(np.array(points[0]) - np.array(points[1]))            
                scale_factor = pixel_distance / real_distance_cm
                self.scale_factor = scale_factor
                cv2.destroyWindow("Calibrate Scale")
                return scale_factor

    def select_points(self, event, x, y, flags, points):
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append((x, y))
                print(f"Point selected: {x}, {y}")
    
    def save_detection_data(self, data):
        filename = os.path.join("video_data", f"{self.video_file}.csv")
        if not os.path.exists("video_data"):
            os.makedirs("video_data")
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"Data saved to {filename}")

    def save_video_data(self):
        json_path = self.save_file
        try:
            if os.path.exists(json_path):
                with open(json_path, 'r') as json_file:
                    video_data = json.load(json_file)
            else:
                video_data = {}
        except (json.JSONDecodeError, FileNotFoundError):
            video_data = {}
        
        video_entry = {}
        if self.corners is not None:
            video_entry['corners'] = self.corners
        
        if self.obstacles is not None:
            video_entry['obstacles'] = [obstacle.tolist() for obstacle in self.obstacles]
        
        if self.scale_factor is not None:
            video_entry['scale_factor'] = self.scale_factor

        if self.finish_line is not None:
            video_entry['finish_line'] = self.finish_line.tolist()
        
        video_data[self.video_file] = video_entry
        
        try:
            with open(json_path, 'w') as json_file:
                json.dump(video_data, json_file, indent=4)
            print(f"Video data saved to {json_path}")
        except Exception as e:
            print(f"Error saving video data: {e}")
    
    def load_video_data(self):
        json_path = self.save_file        
        try:
            if not os.path.exists(json_path):
                print(f"No save file found at {json_path}")
                return False
            
            with open(json_path, 'r') as json_file:
                video_data = json.load(json_file)
                        
            if self.video_file not in video_data:
                print(f"No data found for video {self.video_file}")
                return False
            
            video_entry = video_data[self.video_file]
            
            self.corners = video_entry.get('corners', None)

            obstacles = video_entry.get('obstacles', None)
            self.obstacles = [np.array(obstacle, dtype=float) for obstacle in obstacles] if obstacles is not None else None
            
            self.scale_factor = video_entry.get('scale_factor', None)
            
            finish_line = video_entry.get('finish_line', None)
            self.finish_line = np.array(finish_line) if finish_line is not None else None
            
            return all(item is not None for item in [self.corners, self.obstacles, self.scale_factor, self.finish_line])
        
        except json.JSONDecodeError:
            print(f"Error decoding JSON from {json_path}")
            return False
        except Exception as e:
            print(f"Unexpected error loading video data: {e}")
            return False

    def select_obstacles(self):
            obstacles = []
            points = []
            text = "Select corners for each obstacle."

            while True:
                frame = self.get_transformed_frame()
                
                self.put_text(frame, text)
                cv2.imshow("Select Obstacles", frame)

                def on_mouse(event, x, y, flags, param):
                    debug_frame = frame.copy()

                    if event == cv2.EVENT_LBUTTONDOWN:
                        points.append((x, y))
                        cv2.circle(debug_frame, (x, y), 4, (0, 255, 0), -1)

                    points_len = len(points)
                    for i in range(points_len):
                        if i != points_len - 1:
                            cv2.line(debug_frame, points[i], points[(i+1) % points_len], (255, 0,0 ), 2)

                    for obstacle in obstacles:
                        obstacle_len = len(obstacle)
                        for i in range(obstacle_len):
                            cv2.line(debug_frame, obstacle[i], obstacle[(i+1) % obstacle_len], (255, 0,0 ), 2)
                        
                    cv2.imshow("Select Obstacles", debug_frame)

                cv2.setMouseCallback("Select Obstacles", on_mouse)
                
                while True:
                    if self.pressed(' '):
                        if len(points) != 0:
                            obstacles.append(list(points))
                            points = []
                            
                    if self.pressed('r'):
                        points = []
                    if self.pressed('n'):
                        break
                    if self.pressed('\r'):
                        cv2.destroyWindow("Select Obstacles")
                        obstacles = [np.array(obstacle, dtype=float) for obstacle in obstacles]
                        self.obstacles = obstacles
                        return obstacles

    def select_finish_line(self):
        points = []
        text = "Select 2 points for the finish line. "

        while True:
            #ret, frame = self.cap.read()
            frame = self.read_capture()

            frame = self.perspective_transform(frame)
            window_name = "Select finish line"
            self.put_text(frame, text)
            cv2.imshow(window_name, frame)
    
            def on_mouse(event, x, y, flags, param):
                debug_frame = frame.copy()
                                
                if len(points) < 2:
                    if event == cv2.EVENT_LBUTTONDOWN:
                            points.append((x, y))
                            
                for point in points:
                    cv2.circle(debug_frame, point, 4, (0, 255, 0), -1)

                if len(points) == 2:
                            cv2.line(debug_frame, points[0], points[1], (255, 0,0 ), 2)

                cv2.imshow(window_name, debug_frame)
               
            cv2.setMouseCallback(window_name, on_mouse)

            while True:                    
                if self.pressed('n'):
                    break
                
                if self.pressed('r'):
                    points = []

                if self.pressed('\r'):
                    if len(points) == 2:
                        cv2.destroyWindow(window_name)
                        finish_line = np.array(points)
                        self.finish_line = finish_line
                        return finish_line
                    
    def perspective_transform(self, frame, debug=False):
        corners = np.array(self.corners, dtype=np.float32)
        top_left = corners[np.argmin(corners.sum(axis=1))]      
        top_right = corners[np.argmax(corners[:, 0] - corners[:, 1])]
        bottom_right = corners[np.argmax(corners.sum(axis=1))]
        bottom_left = corners[np.argmin(corners[:, 0] - corners[:, 1])]
       
        src_points = np.array([
            top_left,
            top_right,
            bottom_right,
            bottom_left
        ], dtype=np.float32)

        dst_points = np.array([
            [0, 0],
            [self.width, 0],      
            [self.width, self.height],
            [0, self.height]      
        ], dtype=np.float32)
       
        if debug:
            debug_frame = frame.copy()
            for point in src_points:
                cv2.circle(debug_frame, tuple(point.astype(int)), 5, (0, 255, 0), -1)
            cv2.imshow("Warp debug", debug_frame)

        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        warped = cv2.warpPerspective(frame, matrix, (self.width, self.height))
        return warped
    
    def get_points_and_polygon(self, contours, car_mask):
        largest_contour = max(contours, key=cv2.contourArea)    
        epsilon = 0.10 * cv2.arcLength(largest_contour, True)
        polygon = cv2.approxPolyDP(largest_contour, epsilon, True)
        points = polygon.reshape(-1, 2)
        return points, polygon
        
    def create_history_mask(self, frame, history, window_size=60):
        if len(history) < 100: ###
            return frame
        
        if history.positions[-1] is None:
            window_size = 1000 ###
        
        current_pos = history.get_latest_position()
        if current_pos is None:
            return frame

        p1 = np.array([current_pos[0] - window_size, current_pos[1] - window_size])
        p2 = np.array([current_pos[0] + window_size, current_pos[1] - window_size])
        p3 = np.array([current_pos[0] + window_size, current_pos[1] + window_size])
        p4 = np.array([current_pos[0] - window_size, current_pos[1] + window_size])
        points = np.array([p1, p2, p3, p4], dtype=np.int32)

        history_mask = np.zeros_like(frame)
        cv2.fillPoly(history_mask, [points], 255)
        result = cv2.bitwise_and(frame, history_mask)
        cv2.imshow("history result", result)
        
        cv2.polylines(self.debug_frame, [points], isClosed=True, color=(0, 255, 0), thickness=2)
        return result
    

    def calculate_distance(self, p1, p2):
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return np.sqrt(dx**2 + dy**2)

    def points_to_angle(self, p1, p2):        
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        angle_rad = math.atan2(-dy, dx)
        angle = (math.degrees(angle_rad) + 360) % 360
        return angle
    
    def angle_difference(self, angle1, angle2):
        difference = abs(angle1 - angle2)
        return min(difference, 360 - difference)

    def determine_angle(self, points, center, history):
        latest_angle = history.get_latest_angle()
        angles = []

        for i in range(len(points)):
            p1 = points[i]
            p2 = points[(i+1) % len(points)]
            midpoint = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
            angle = self.points_to_angle(midpoint, center)
            angles.append(angle)            

        angle = min(angles, key=lambda x: self.angle_difference(x, latest_angle))

        return angle



    def find_corners_with_polygon(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours is None or len(contours) == 0:
            return None
        
        largest_contour = max(contours, key=cv2.contourArea)
        epsilon = 0.10 * cv2.arcLength(largest_contour, True)
        polygon = cv2.approxPolyDP(largest_contour, epsilon, True)
        corners = polygon.reshape(-1, 2)
        cv2.polylines(self.debug_frame, polygon, True, (255, 0, 0), thickness=2)
        return corners

    def calculate_center(self, points):
        if len(points) == 0:
            return None
        center_x = np.mean([point[0]for point in points])
        center_y = np.mean([point[1]for point in points])
        center = (center_x, center_y)
        return center

    
    def color_detection(self, frame, history):
        self.debug_frame = frame.copy()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        car_mask = cv2.inRange(hsv, *self.color_ranges['car_full'])
        car_mask = self.create_history_mask(car_mask, history)

        center, angle = None, None

        for obstacle in self.obstacles:
            obstacle = np.array(obstacle, dtype=np.int32)
            cv2.fillPoly(car_mask, [obstacle], 0)

        corners = self.find_corners_with_polygon(car_mask)
        if corners is None:
            return center, angle
        
        center = self.calculate_center(corners)
        if center is None or len(corners) != 4:
            return center, angle
    
        angle = self.determine_angle(corners, center, history)
        if angle is None:
            return center, angle
        

        center_int = (int(center[0]), int(center[1]))
        cv2.circle(self.debug_frame, center_int, 3, (255, 0, 0), -1)
        cv2.putText(self.debug_frame,f"({center[0]:.0f}, {center[1]:.0f}) {angle:.0f} deg",(center_int[0] + 50, center_int[1] + 50),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255, 0, 0),1,cv2.LINE_AA)
        
        arrow_length = 50
        angle_rad = math.radians(angle)
        x = round(center_int[0] + math.cos(angle_rad) * arrow_length)    
        y = round(center_int[1] - math.sin(angle_rad) * arrow_length)
        arrow_end = (x, y)
        cv2.arrowedLine(frame, center_int, arrow_end, (0, 255, 0), 2)
    
        return center, angle



    def run(self):
        
        if self.video_file:
            is_saved = self.load_video_data()

            if self.corners is None:
                print("No saved corners.")
                self.select_corners()
        else:
            print("Using live feed.")
            self.select_corners()
        
        # if self.scale_factor is None:
        #     print("No saved scale factor.")
        #     self.calibrate_scale(200)

        if self.obstacles is None:
            print("No saved obstacles.")
            self.select_obstacles()

        if self.finish_line is None:
            print("No saved finsish line.")
            self.select_finish_line()
    
        if not is_saved:
            self.save_video_data()

        self.start_detection()

    def get_transformed_frame(self):
        #ret, frame = self.cap.read()
        frame = self.read_capture()
        if frame is None:
            return None
        
        warped = self.perspective_transform(frame)
        return warped
    
    def start_detection(self):    
        frame_count = 0

        history = History()
        history.append(self.position, self.angle)
        start_time = time.time() ###
        while True:
            time_ = time.time() - start_time

            frame = self.get_transformed_frame()
            if frame is None:
                break
            
            position, angle = self.color_detection(frame, history)
            history.append(position, angle)
            self.position = position if position is not None else self.position
            self.angle = angle if position is not None else self.angle
            
            if not self.started:
                self.started = position is not None

            
            if not self.finished:
                has_crossed_finish_line = self.crossed_finish_line(history)
                if has_crossed_finish_line:
                    valid_finish = history.valid_finish(frame_count, time_)
                    if valid_finish is not None:
                        self.finished = True                        

                        self.save_finish_time(valid_finish)
            
            self.debug_detection(frame_count, time_)

            frame_count += 1
       
            if self.pressed('q'):
                break

        #self.continue_detection(corners, scale_factor, obstacles, history)
        # self.cap.release()
        # cv2.destroyAllWindows()


    def debug_detection(self, frame_count, time_):
        if self.position is not None and self.angle is not None:
            n, e, s, w = self.get_distance_to_obstacles()
            x, y = map(int, self.position)
            distance_points = [
                (x, y - n), 
                (x + e, y),  
                (x, y + s),  
                (x - w, y)   
            ]
            
            for destination in distance_points:
                destination = tuple(map(int, destination))
                cv2.line(self.debug_frame, (x, y), destination, (255, 0, 0), thickness=2)

        for obstacle in self.obstacles:
                    for i in range(len(obstacle)):
                        pt1 = tuple(map(int, obstacle[i]))
                        pt2 = tuple(map(int, obstacle[(i+1) % len(obstacle)]))
                        cv2.line(self.debug_frame, pt1, pt2, (255, 0, 0), thickness=2)

        cv2.line(self.debug_frame, self.finish_line[0], self.finish_line[1], (200, 255, 0), 2)


        for i in range(self.horizontal_squares):
            x = self.pixels_per_square * (i+1) 
            y = self.height
            p1 = np.array([x,0])
            p2 = np.array([x,y])
            cv2.line(self.debug_frame, p1,p2, (10,10,10), 1)

        for i in range(self.vertical_squares):
            x = self.width
            y = self.pixels_per_square * (i+1)
            p1 = np.array([0,y])
            p2 = np.array([x,y])
            cv2.line(self.debug_frame, p1,p2, (10,10,10), 1)


        
        self.put_text(self.debug_frame, f"frame: {frame_count} time: {time_:.2f}")
        
        if self.finished:
            self.put_text(self.debug_frame, f"finished at frame {frame_count} and time: ")
                

        cv2.imshow("Car and obstacles", self.debug_frame)

    def get_distance_to_obstacles(self, direction=None):
        corners = np.array([[0,0], [self.width,0], [self.width,self.height], [0,self.height]])
        north, east, south, west = self.calculate_distance_for_polygon(corners)

        for obstacle in self.obstacles:
            obstacle_north, obstacle_east, obstacle_south, obstacle_west = self.calculate_distance_for_polygon(obstacle)
            north = min(north, obstacle_north)
            east = min(east, obstacle_east)
            south = min(south, obstacle_south)
            west = min(west, obstacle_west)

        distances = {"north": north, "east": east, "south": south, "west": west}
        return distances[direction] if direction else (north, east, south, west)

    def calculate_distance_for_polygon(self, polygon):
        polygon = np.array(polygon)
        directions = np.array([
            [0, -1],
            [1, 0],   
            [0, 1],   
            [-1, 0]    
        ])
        distances = []
        
        for direction in directions:
            min_distance = float('inf')
            
            for i in range(len(polygon)):
                p1 = polygon[i]
                p2 = polygon[(i + 1) % len(polygon)]                
                intersection_distance = self.ray_intersection_distance(
                    ray_origin=self.position, 
                    ray_direction=direction, 
                    line_start=p1, 
                    line_end=p2
                )
                
                if intersection_distance is not None and intersection_distance >= 0:
                    min_distance = min(min_distance, intersection_distance)
            
            distances.append(min_distance)
        
        return tuple(distances)

    def ray_intersection_distance(self, ray_origin, ray_direction, line_start, line_end):
        line_vector = line_end - line_start        
        ray_cross_line = np.cross(ray_direction, line_vector)
        if np.abs(ray_cross_line) < 1e-10:
            return None
        t = np.cross(line_start - ray_origin, line_vector) / ray_cross_line
        u = np.cross(line_start - ray_origin, ray_direction) / ray_cross_line
        if 0 <= u <= 1 and t >= 0:
            return t    
        return None


    def detect_motion(self, frame, scale_factor, obstacles): ###
        mask = self.object_detector.apply(frame)
        cv2.imshow("Mask 1", mask)
        position, angle = None, None

        for obstacle in obstacles:
            obstacle = np.array(obstacle, dtype=np.int32)
            cv2.fillPoly(mask, [obstacle], 0)
        

        _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
        cv2.imshow("Mask 2", mask)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) != 0:
            largest_contour = max(contours, key=cv2.contourArea)
            if largest_contour is not None:
                rect = cv2.minAreaRect(largest_contour)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)
 
        cv2.imshow("Motion detection", frame)
        
            
        return position, angle, frame
    

    def crossed_finish_line(self, history):
        if history.get_latest_position() is None:
            return False
        current_pos = np.array(history.get_latest_position())
        line_start, line_end = self.finish_line
        line_vector = line_end - line_start
        pos_vector = current_pos - line_start        
        cross_product = np.cross(line_vector, pos_vector)
        return cross_product < 0
    
    def test_scale_factor(self, scale_factor):
        while True:
            frame = self.get_transformed_frame()

            cv2.imshow("Test scale", frame)
            points = []

            def on_mouse(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    points.append((x, y))
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                    cv2.imshow("Test scale", frame)
                    if len(points) == 2:
                        cv2.line(frame, points[0], points[1], (255, 0, 0), 2)
                        cv2.imshow("Test scale", frame)

            cv2.setMouseCallback("Test scale", on_mouse)
            print("Select two points representing the real-world distance.")

            while len(points) < 2:
                if self.pressed('n'):
                    points = []
                    break

            if len(points) == 2:
                pixel_distance = np.linalg.norm(np.array(points[0]) - np.array(points[1]))            
                real_distance_cm = pixel_distance / scale_factor
                cv2.putText(frame, f"Distance: {real_distance_cm} cm", points[0],cv2.FONT_HERSHEY_SIMPLEX,0.5,(255, 0, 0),1,cv2.LINE_AA,)
                print(f'Distance {real_distance_cm}')


    def save_finish_time(self, frame_count, time_):
        pass



class History():
    def __init__(self) -> None:
        self.positions = []
        self.angles = []
        self.finished = []


    def append(self, position, angle):
        self.positions.append(position)
        self.angles.append(angle)

    def valid_finish(self, frame_count, time):
        self.finished.append([frame_count, time])

    def get_latest_position(self):
        for position in reversed(self.positions):
            if position is not None:
                return position
        
        print("No valid position found")
        return None
        
    def get_latest_angle(self):
        for angle in reversed(self.angles):
            if angle is not None:
                return angle
        
        print("No valid angle found")
        return None
    
    def __len__(self):
        return len(self.positions)


if __name__ == "__main__":
    #detector = CarDetector("video1.avi")
    #detector = CarDetector("forward.avi")
    #detector = CarDetector("right.avi")

    #detector = CarDetector("backward.avi")
    #detector = CarDetector("left.avi")
    #detector = CarDetector("obstacles1.avi")
    detector = CarDetector()
    detector.run()

###
