from ultralytics import YOLO
import cv2
import pickle
from utils import get_center_of_bbox, measure_distance

class PlayerTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
    
    def choose_and_filter_players(self, court_keypoints, player_detections):
        player_detections_first_frame = player_detections[0]
        chosen_player = self.choose_players(court_keypoints, player_detections_first_frame)
        filtered_player_detections = []
        for player_dict in player_detections:
            filtered_player_dict = {track_id: bbox for track_id, bbox in player_dict.items() if track_id in chosen_player}
            filtered_player_detections.append(filtered_player_dict)
        return filtered_player_detections

    def choose_players(self, court_keypoints, player_dict):
        distances = []
        for track_id, bbox in player_dict.items():
            player_center = get_center_of_bbox(bbox)

            min_distance = float('inf')
            for i in range(0,len(court_keypoints),2):
                court_keypoint = (court_keypoints[i], court_keypoints[i+1])
                distance = measure_distance(player_center, court_keypoint)
                min_distance = min(min_distance, distance)
            distances.append((track_id, min_distance))
        
        # sorrt the distances in ascending order
        distances.sort(key = lambda x: x[1])
        # Choose the first 2 tracks
        chosen_players = [distances[0][0], distances[1][0]]
        return chosen_players
    
    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        '''
        Takes a list of frames as input and returns a list of dictionaries of player detections.
        Each dictionary should have the track_id as the key and the bounding box as the value.
        The index of the dictionary in the list should correspond to the index of the frame in the input list.
        Added a stub system to read and write the player detections to a file to avoid running the detection again and again.
        '''
        if read_from_stub and stub_path:
            with open(stub_path, 'rb') as f:
                player_detections = pickle.load(f)
            return player_detections

        player_detections = []

        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)
        
        if stub_path:
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detections, f)
        
        return player_detections

    def detect_frame(self, frame):
        '''
        Takes a frame as input and returns a dictionary of player detections.
        The dictionary should have the track_id as the key and the bounding box as the value.
        It takes a single frame, runs the YOLO detections and finds the ID of boxes, class of boxes and the bounding box coordinates.
        It then filters out the boxes which are of class 'person' and appends the player id and bounding box to the dictionary.
        '''
        results = self.model.track(frame, persist=True)[0] # persist=True to track objects across frames
        id_name_dict = results.names
        
        player_dict = {}
        for box in results.boxes:
            track_id = int(box.id.tolist()[0])
            result = box.xyxy.tolist()[0]
            object_cls_id = box.cls.tolist()[0]
            object_cls_name = id_name_dict[object_cls_id]
            if object_cls_name == 'person':
                player_dict[track_id] = result

        return player_dict
    
    def draw_bboxes(self, video_frames, player_detections):
        '''
        Takes a list of frames and a list of player detections as input and returns a list of frames with bounding boxes drawn around the players.
        The player_detections is a list of dictionaries where each dictionary has the track_id as the key and the bounding box as the value.
        The index of the dictionary in the list should correspond to the index of the frame in the input list.
        '''
        output_video_frames = []
        for frame, player_dict in zip(video_frames, player_detections):
            for track_id, bbox in player_dict.items():
                # Convert coordinates to integers
                x1, y1, x2, y2 = map(int, bbox)
                cv2.putText(frame, f"Player ID: {track_id}", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            output_video_frames.append(frame)
        return output_video_frames

