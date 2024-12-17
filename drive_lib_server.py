import requests
from math import sqrt
import time

class DriveLib:
    """
    A library for controlling a RC-car via HTTP requests to a Raspberry Pi server. 
    Handles movement, position, and angle management.
    """
    def __init__(self, competitor_name='Player'):
        """
        Initializes the DriveLib instance and starts camera detection.

        Parameters:
        competitor_name (str): The name of the competitor to be sent to the server.

        Returns:
        None
        """
        self.raspberry_pi_ip = "192.168.0.22"
        self._post_to_server('start_detection', {'competitor_name':competitor_name})
        
        while self.angle is None or self._position is None:
            time.sleep(0.05)
        time.sleep(2)
        
    def _get_from_server(self, endpoint):
        """
        Sends a GET request to the specified server endpoint.

        Parameters:
        endpoint (str): The API endpoint to fetch data from.

        Returns:
        dict: The JSON response from the server if successful, None otherwise.
        """
        url = f"http://{self.raspberry_pi_ip}:80/{endpoint}"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Failed to get {endpoint}: {response.json()}")
                return None
        except Exception as e:
            print(f"Error fetching from server: {e}")
            return None
    
    def _post_to_server(self, endpoint, data):
        """
        Sends a POST request with JSON data to the specified server endpoint.

        Parameters:
        endpoint (str): The API endpoint to send data to.
        data (dict): The JSON data to include in the request.

        Returns:
        None
        """
        url = f"http://{self.raspberry_pi_ip}:80/{endpoint}"
        try:
            response = requests.post(url, json=data)
            if response.status_code != 200:
                print(f"Failed to post to {endpoint}: {response.json()}")
        except Exception as e:
            print(f"Error posting to server: {e}")


    @property
    def angle(self):
        """
        Retrieves the current angle of the car from the server.

        Returns:
        float: The current angle in degrees, or None if unavailable.
        """
        data = self._get_from_server("get_ang")
        return data["angle"] if data else None

    @property
    def _position(self):
        """
        Retrieves the raw position (x, y) of the car from the server.

        Returns:
        tuple: The raw position as (x, y), or None if unavailable.
        """
        data = self._get_from_server("get_pos")
        return tuple(data["position"]) if data else None

    @property
    def position(self):
        """
        Retrieves the scaled position (x, y) of the car.

        Returns:
        tuple: The scaled position as (x, y).
        """
        return (self._position[0] / self._scale_factor, self._position[1] / self._scale_factor)
    
    @property
    def north(self):
        """
        Retrieves the northward distance from the car to an obstacle, adjusted by scale and offset.

        Returns:
        float: The northward distance, or an empty list if unavailable.
        """
        data = self._get_from_server("get_distances")
        return (data["distances"][0] / self._scale_factor) - 7 if data else []
    
    @property
    def east(self):
        """
        Retrieves the eastward distance from the car to an obstacle, adjusted by scale and offset.

        Returns:
        float: The eastward distance, or an empty list if unavailable.
        """
        data = self._get_from_server("get_distances")
        return (data["distances"][1] / self._scale_factor) -7 if data else []
    
    @property
    def south(self):
        """
        Retrieves the southward distance from the rcarobot to an obstacle, adjusted by scale and offset.

        Returns:
        float: The southward distance, or an empty list if unavailable.
        """
        data = self._get_from_server("get_distances")
        return (data["distances"][2] / self._scale_factor) - 7 if data else []
    
    @property
    def west(self):
        """
        Retrieves the westward distance from the car to an obstacle, adjusted by scale and offset.

        Returns:
        float: The westward distance, or an empty list if unavailable.
        """
        data = self._get_from_server("get_distances")
        return (data["distances"][3] / self._scale_factor) - 7 if data else []
    
    @property
    def _scale_factor(self):
        """
        The scaling factor for distance and position calculations.

        Returns:
        int: The scaling factor (default is 2).
        """
        return 2
    
    def go_forward(self, distance):
        """
        Moves the car forward by a specified distance.

        Parameters:
        distance (float): The distance to move forward in centimeters.

        Returns:
        None
        """
        start_position = self.position
        start_angle = self.angle
        print(f'Startposition: x:{round(start_position[0])}, y:{round(start_position[1])}')
        print(f'Startvinkel: {round(start_angle)}°\n')
        
        target_distance = distance
        max_step = 25

        while abs(target_distance) > 5:
            step_distance = min(max_step, abs(target_distance)) 
            step_distance *= 1 if target_distance > 0 else -1  

            print(f"Kör {'framåt' if step_distance > 0 else 'bakåt'} {round(abs(step_distance))} cm\n")
            command = {"action": "forward", "distance": step_distance}
            self._send_command(command)
            
            current_position = self.position
            current_angle = self.angle
            
            print(f'Ny position: x:{round(current_position[0])}, y:{round(current_position[1])}')
            print(f'Ny vinkel: {round(current_angle)}°\n')
            
            distance_travelled = self.calculate_distance(start_position, current_position)
            angle_difference = self.angle_difference(start_angle, current_angle) 

            print(f'Körd distans: {round(distance_travelled)} cm')
            print(f'Vridning: {round(angle_difference)}°\n')            

            if abs(angle_difference) > 10:  
                print(f"Korrigerar vinkel:")
                correction_angle = -angle_difference 
                self.turn(correction_angle)
            
            if step_distance > 0:
                target_distance -= distance_travelled
            else:
                target_distance -= (distance_travelled * -1)
            start_position = current_position  
            
            print(f"Kvarvarande distans: {round(target_distance)} cm\n")
                
    def go_backward(self, distance):
        """
        Moves the car backward by a specified distance.

        Parameters:
        distance (float): The distance to move backward in centimeters.

        Returns:
        None
        """
        start_position = self.position
        print(f'Startpostion: x:{round(start_position[0])} y:{round(start_position[1])}')
        target_distance = distance
        
        while True:
            command = {"action": "backward", "distance": target_distance}
            self._send_command(command)
            
            current_position = self.position
            print(f'Ny position: x:{round(current_position[0])} y:{round(current_position[1])}')
            distance_travelled = self.calculate_distance(start_position, current_position)
            print(f'Körd distans: {round(distance_travelled)}')
            
            if abs(distance_travelled - target_distance) <= 5:
                break
            else:
                target_distance -= distance_travelled
                start_position = current_position
                if target_distance > 0:
                    print(f"Adjusting backward by {target_distance} cm")
                else:
                    print(f"Adjusting forward by {target_distance} cm")
                
    def turn(self, angle):
        """
        Rotates the car by a specified angle.

        Parameters:
        angle (float): The angle to turn in degrees.

        Returns:
        None
        """
        start_angle = self.angle
        print(f'Startangle: {round(start_angle)}°\n')
        target_angle = angle

        while abs(target_angle) > 5:
            command = {"action": "left", "angle": target_angle}
            self._send_command(command)

            current_angle = self.angle
            print(f'New angle: {round(current_angle)}')

            turned_angle = self.angle_difference(start_angle, current_angle)
            print(f'Turned degrees: {round(turned_angle)}°\n')

            target_angle -= turned_angle

            if target_angle > 0:
                print(f"Adjusting left turn by {target_angle}°\n")
            else:
                print(f"Adjusting right turn by {abs(target_angle)}°\n")
                
            start_angle = current_angle
            
    def calculate_distance(self, position1, position2):
        """
        Calculates the Euclidean distance between two positions.

        Parameters:
        position1 (tuple): The first position as (x, y).
        position2 (tuple): The second position as (x, y).

        Returns:
        float: The distance between the two positions.
        """
        x1, y1 = position1
        x2, y2 = position2
        return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    
    def angle_difference(self, start_angle, current_angle):
        """
        Calculates the smallest difference between two angles, accounting for wrap-around at 360 degrees.

        Parameters:
        start_angle (float): The starting angle in degrees.
        current_angle (float): The current angle in degrees.

        Returns:
        float: The smallest angle difference in degrees.
        """
        diff = (current_angle - start_angle) % 360
        if diff > 180:
            diff -= 360 
        return diff

    def _send_command(self, command):
        """
        Sends a command to the server to execute an action.

        Parameters:
        command (dict): The command to send, containing action details.

        Returns:
        None
        """
        try:
            response = requests.post(f"http://{self.raspberry_pi_ip}:80/execute", json={"command": command})
            if response.status_code != 200:
                print("Failed to send command:", response.status_code)
        except Exception as e:
            print("Error sending command:", e)

