from ultralytics import YOLO
import cv2
import pickle
import pandas as pd

class BallTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
    
    def detect_frames(self, frames, read_from_stub=False, stub_path=None, interpolate=True):
        '''
        Takes a list of frames as input and returns a list of dictionaries of player detections.
        Each dictionary should have the track_id as the key and the bounding box as the value.
        The index of the dictionary in the list should correspond to the index of the frame in the input list.
        Added a stub system to read and write the player detections to a file to avoid running the detection again and again.
        '''
        if read_from_stub and stub_path:
            with open(stub_path, 'rb') as f:
                ball_detections = pickle.load(f)
            return ball_detections

        ball_detections = []

        for frame in frames:
            ball_dict = self.detect_frame(frame)
            ball_detections.append(ball_dict)

        # Interpolate the ball detections
        if interpolate:
            ball_detections = [x.get(1,[]) for x in ball_detections]
            # convert the list into pandas dataframe
            df_ball_detections = pd.DataFrame(ball_detections,columns=['x1','y1','x2','y2'])

            # interpolate the missing values
            df_ball_detections = df_ball_detections.interpolate()
            df_ball_detections = df_ball_detections.bfill()

            ball_detections = [{1:x} for x in df_ball_detections.to_numpy().tolist()]
        
        if stub_path:
            with open(stub_path, 'wb') as f:
                pickle.dump(ball_detections, f)
        
        return ball_detections

    def detect_frame(self, frame):
        '''
        Takes a frame as input and returns a dictionary of player detections.
        The dictionary should have the track_id as the key and the bounding box as the value.
        It takes a single frame, runs the YOLO detections and finds the ID of boxes, class of boxes and the bounding box coordinates.
        It then filters out the boxes which are of class 'person' and appends the player id and bounding box to the dictionary.
        '''
        results = self.model.predict(frame, conf=0.15)[0]

        ball_dict={}
        for box in results.boxes:
            result = box.xyxy.tolist()[0]
            ball_dict[1] = result

        return ball_dict
    
    def draw_bboxes(self, video_frames, ball_detections):
        '''
        Takes a list of frames and a list of player detections as input and returns a list of frames with bounding boxes drawn around the players.
        The player_detections is a list of dictionaries where each dictionary has the track_id as the key and the bounding box as the value.
        The index of the dictionary in the list should correspond to the index of the frame in the input list.
        '''
        output_video_frames = []
        for frame, ball_dict in zip(video_frames, ball_detections):
            for track_id, bbox in ball_dict.items():
                # Convert coordinates to integers
                x1, y1, x2, y2 = map(int, bbox)
                cv2.putText(frame, f"Ball ID: {track_id}", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            output_video_frames.append(frame)
        return output_video_frames
    
    def get_ball_shot_frames(self,ball_positions):
        ball_positions = [x.get(1,[]) for x in ball_positions]
        # convert the list into pandas dataframe
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        df_ball_positions['ball_hit'] = 0

        df_ball_positions['mid_y'] = (df_ball_positions['y1'] + df_ball_positions['y2'])/2
        df_ball_positions['mid_y_rolling_mean'] = df_ball_positions['mid_y'].rolling(window=5, min_periods=1, center=False).mean()
        df_ball_positions['delta_y'] = df_ball_positions['mid_y_rolling_mean'].diff()
        minimum_change_frames_for_hit = 25
        for i in range(1,len(df_ball_positions)- int(minimum_change_frames_for_hit*1.2) ):
            negative_position_change = df_ball_positions['delta_y'].iloc[i] >0 and df_ball_positions['delta_y'].iloc[i+1] <0
            positive_position_change = df_ball_positions['delta_y'].iloc[i] <0 and df_ball_positions['delta_y'].iloc[i+1] >0

            if negative_position_change or positive_position_change:
                change_count = 0 
                for change_frame in range(i+1, i+int(minimum_change_frames_for_hit*1.2)+1):
                    negative_position_change_following_frame = df_ball_positions['delta_y'].iloc[i] >0 and df_ball_positions['delta_y'].iloc[change_frame] <0
                    positive_position_change_following_frame = df_ball_positions['delta_y'].iloc[i] <0 and df_ball_positions['delta_y'].iloc[change_frame] >0

                    if negative_position_change and negative_position_change_following_frame:
                        change_count+=1
                    elif positive_position_change and positive_position_change_following_frame:
                        change_count+=1
            
                if change_count>minimum_change_frames_for_hit-1:
                    df_ball_positions['ball_hit'].iloc[i] = 1

        frame_nums_with_ball_hits = df_ball_positions[df_ball_positions['ball_hit']==1].index.tolist()

        return frame_nums_with_ball_hits

