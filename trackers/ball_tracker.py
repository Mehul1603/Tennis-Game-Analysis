from ultralytics import YOLO
import torch
import cv2
import pickle
import pandas as pd
import numpy as np
from itertools import groupby
from scipy.spatial import distance
import os
from tracknet_models import BallTrackerNet  # Import the TrackNet model

class BallTracker:
    def __init__(self, model_path, use_tracknet=False):
        """
        Initialize the ball tracker with either YOLO or TrackNet model
        
        Args:
            model_path: Path to the model file
            use_tracknet: Whether to use TrackNet model instead of YOLO
        """
        self.use_tracknet = use_tracknet
        self.model_path = model_path
        
        if use_tracknet:
            # Initialize TrackNet model
            self.model = BallTrackerNet()
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model = self.model.to(self.device)
            self.model.eval()
        else:
            # Initialize YOLO model
            self.model = YOLO(model_path)
    
    def detect_frames(self, frames, read_from_stub=False, stub_path=None, interpolate=True):
        """
        Detect ball in all frames
        
        Args:
            frames: List of video frames
            read_from_stub: Whether to read detections from a pickle file
            stub_path: Path to the pickle file
            interpolate: Whether to interpolate ball positions
            
        Returns:
            List of dictionaries with ball positions for each frame
        """
        if read_from_stub and stub_path and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                ball_detections = pickle.load(f)
            return ball_detections

        # Choose detection method based on model type
        if self.use_tracknet:
            ball_detections = self._detect_frames_tracknet(frames)
        else:
            ball_detections = self._detect_frames_yolo(frames)

        # Save detections to stub file if path is provided
        if stub_path:
            with open(stub_path, 'wb') as f:
                pickle.dump(ball_detections, f)
        
        return ball_detections
    
    def _detect_frames_yolo(self, frames):
        """Use YOLO to detect ball in frames"""
        ball_detections = []
        
        for frame in frames:
            ball_dict = self.detect_frame(frame)
            ball_detections.append(ball_dict)

        # Interpolate the ball detections using pandas
        if True:  # Always interpolate with YOLO for consistency
            # Extract ball positions
            ball_positions = [x.get(1, []) for x in ball_detections]
            
            # Convert to DataFrame for interpolation
            df_ball_detections = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])
            
            # Interpolate missing values
            df_ball_detections = df_ball_detections.interpolate()
            df_ball_detections = df_ball_detections.bfill()
            
            # Convert back to dictionary format
            ball_detections = [{1: x} for x in df_ball_detections.to_numpy().tolist()]
        
        return ball_detections
    
    def _detect_frames_tracknet(self, frames):
        """Use TrackNet to detect ball in frames"""
        # Initialize outputs
        height, width = frames[0].shape[:2]
        target_height, target_width = 360, 640
        ball_track = []
        
        # Add initial placeholder entries
        ball_track.append({1: None})
        ball_track.append({1: None})
        
        # Process frames with TrackNet model
        for num in range(2, len(frames)):
            # Prepare input: current frame and two previous frames
            img = cv2.resize(frames[num], (target_width, target_height))
            img_prev = cv2.resize(frames[num-1], (target_width, target_height))
            img_preprev = cv2.resize(frames[num-2], (target_width, target_height))  # Fixed dimension order

            
            # Concatenate frames for TrackNet input
            imgs = np.concatenate((img, img_prev, img_preprev), axis=2)
            imgs = imgs.astype(np.float32)/255.0
            imgs = np.rollaxis(imgs, 2, 0)
            inp = np.expand_dims(imgs, axis=0)
            
            # Get model prediction
            with torch.no_grad():
                out = self.model(torch.from_numpy(inp).float().to(self.device))
                output = out.argmax(dim=1).detach().cpu().numpy()
            
            # Process output to get ball coordinates
            x_pred, y_pred = self._postprocess_tracknet(output)
            
            # Scale coordinates back to original frame size
            if x_pred is not None and y_pred is not None:
                x_pred = int(x_pred * width / target_width)
                y_pred = int(y_pred * height / target_height)
                
                # Convert center point to bounding box format
                ball_size = min(width, height) // 30  # Approximate ball size
                x1 = max(0, x_pred - ball_size)
                y1 = max(0, y_pred - ball_size)
                x2 = min(width, x_pred + ball_size)
                y2 = min(height, y_pred + ball_size)
                
                ball_track.append({1: [x1, y1, x2, y2]})
            else:
                ball_track.append({1: None})
        
        # Apply advanced interpolation
        ball_track = self._enhance_ball_track(ball_track)
        
        return ball_track
    
    def _postprocess_tracknet(self, output):
        """Convert TrackNet output to ball coordinates"""
        output = output[0]  # Get first item from batch
        
        # Check dimensionality of output
        if len(output.shape) == 1:
            # 1D output case - need to convert from flat index to 2D coordinates
            target_width = 640  # This should match your target_width value from earlier
            max_idx = np.argmax(output)
            
            # Convert flat index to 2D coordinates
            x_pred = max_idx % target_width
            y_pred = max_idx // target_width
        else:
            # 2D output case - original approach
            h, w = output.shape
            index = np.unravel_index(np.argmax(output), output.shape)
            y_pred, x_pred = index
        
        # Check if prediction is confident enough (non-zero)
        confidence = output.flatten()[np.argmax(output.flatten())]
        if confidence == 0:
            return None, None
        
        return x_pred, y_pred

    
    def _enhance_ball_track(self, ball_track):
        """Apply advanced interpolation techniques from TrackNet"""
        # Convert to format compatible with TrackNet functions
        tracknet_format = []
        for frame_dict in ball_track:
            bbox = frame_dict.get(1)
            if bbox is not None:
                # Convert bbox to center point
                x_center = (bbox[0] + bbox[2]) / 2
                y_center = (bbox[1] + bbox[3]) / 2
                tracknet_format.append((x_center, y_center))
            else:
                tracknet_format.append((None, None))
        
        # Calculate distances between points
        dists = self._calculate_distances(tracknet_format)
        
        # Remove outliers
        tracknet_format = self._remove_outliers(tracknet_format, dists)
        
        # Split track into subtracks
        subtracks = self._split_track(tracknet_format)
        
        # Apply interpolation to each subtrack
        for r in subtracks:
            subtrack = tracknet_format[r[0]:r[1]]
            interpolated_subtrack = self._interpolate_subtrack(subtrack)
            tracknet_format[r[0]:r[1]] = interpolated_subtrack
        
        # Convert back to the original format
        enhanced_ball_track = []
        for i, point in enumerate(tracknet_format):
            if point[0] is not None and point[1] is not None:
                x_center, y_center = point
                # Estimate the ball size based on original frame dimensions
                # This would be better if we had the actual frame size
                ball_size = 10  # Default value, adjust as needed
                
                # Convert center point back to bounding box
                x1 = max(0, int(x_center - ball_size))
                y1 = max(0, int(y_center - ball_size))
                x2 = int(x_center + ball_size)
                y2 = int(y_center + ball_size)
                
                enhanced_ball_track.append({1: [x1, y1, x2, y2]})
            else:
                # If we encounter a frame with no valid point after interpolation,
                # use the previous frame's bounding box if available
                if i > 0 and enhanced_ball_track[i-1].get(1) is not None:
                    enhanced_ball_track.append(enhanced_ball_track[i-1])
                else:
                    enhanced_ball_track.append({1: None})
        
        return enhanced_ball_track
    
    def _calculate_distances(self, ball_track):
        """Calculate distances between consecutive ball points"""
        dists = [-1, -1]  # First two frames have no previous points
        
        for i in range(2, len(ball_track)):
            curr_point = ball_track[i]
            prev_point = ball_track[i-1]
            
            if curr_point[0] is not None and prev_point[0] is not None:
                dist = distance.euclidean(curr_point, prev_point)
            else:
                dist = -1
                
            dists.append(dist)
            
        return dists
    
    def _remove_outliers(self, ball_track, dists, max_dist=100):
        """Remove outliers from ball track based on distances"""
        outliers = list(np.where(np.array(dists) > max_dist)[0])
        
        for i in outliers[:]:  # Use a copy to avoid modifying during iteration
            if i+1 < len(dists):
                if (dists[i+1] > max_dist) or (dists[i+1] == -1):       
                    ball_track[i] = (None, None)
                    if i in outliers:
                        outliers.remove(i)
            
            if i-1 >= 0 and dists[i-1] == -1:
                ball_track[i-1] = (None, None)
                
        return ball_track
    
    def _split_track(self, ball_track, max_gap=4, max_dist_gap=80, min_track=5):
        """Split ball track into subtracks for better interpolation"""
        # Create a list where 0 means valid point, 1 means None
        list_det = [0 if x[0] is not None else 1 for x in ball_track]
        
        # Group consecutive values
        groups = [(k, sum(1 for _ in g)) for k, g in groupby(list_det)]
        
        cursor = 0
        min_value = 0
        result = []
        
        for i, (k, l) in enumerate(groups):
            if (k == 1) and (i > 0) and (i < len(groups) - 1):
                # Calculate distance between points before and after gap
                if cursor-1 >= 0 and cursor+l < len(ball_track):
                    if ball_track[cursor-1][0] is not None and ball_track[cursor+l][0] is not None:
                        dist = distance.euclidean(ball_track[cursor-1], ball_track[cursor+l])
                        if (l >= max_gap) or (dist/l > max_dist_gap):
                            if cursor - min_value > min_track:
                                result.append([min_value, cursor])
                                min_value = cursor + l
            cursor += l
            
        if len(list_det) - min_value > min_track:
            result.append([min_value, len(list_det)])
            
        return result
    
    def _interpolate_subtrack(self, coords):
        """Interpolate missing points in a subtrack"""
        def nan_helper(y):
            return np.isnan(y), lambda z: z.nonzero()[0]
        
        # Extract x and y coordinates
        x = np.array([x[0] if x[0] is not None else np.nan for x in coords])
        y = np.array([x[1] if x[1] is not None else np.nan for x in coords])
        
        # Interpolate x coordinates
        nons, yy = nan_helper(x)
        if np.any(~nons):  # Make sure there are non-NaN values
            x[nons] = np.interp(yy(nons), yy(~nons), x[~nons])
            
        # Interpolate y coordinates
        nans, xx = nan_helper(y)
        if np.any(~nans):  # Make sure there are non-NaN values
            y[nans] = np.interp(xx(nans), xx(~nans), y[~nans])
        
        # Combine interpolated coordinates
        track = [*zip(x, y)]
        return track
    
    def detect_frame(self, frame):
        """Detect ball in a single frame using YOLO"""
        results = self.model.predict(frame, conf=0.15)[0]
        
        ball_dict = {}
        for box in results.boxes:
            result = box.xyxy.tolist()[0]
            ball_dict[1] = result
            
        return ball_dict
    
    def draw_bboxes(self, video_frames, ball_detections):
        """Draw bounding boxes on frames"""
        output_video_frames = []
        
        for frame, ball_dict in zip(video_frames, ball_detections):
            for track_id, bbox in ball_dict.items():
                if bbox is not None:
                    # Convert coordinates to integers
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.putText(frame, f"Ball ID: {track_id}", (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            
            output_video_frames.append(frame)
            
        return output_video_frames
    
    def get_ball_shot_frames(self, ball_positions):
        """Detect frames where the ball is hit"""
        # Extract ball positions, handling None values
        positions = []
        for pos_dict in ball_positions:
            bbox = pos_dict.get(1)
            if bbox is not None:
                positions.append(bbox)
            else:
                # Use a placeholder for missing detections
                positions.append([np.nan, np.nan, np.nan, np.nan])
        
        # Convert to DataFrame
        df_ball_positions = pd.DataFrame(positions, columns=['x1', 'y1', 'x2', 'y2'])
        
        df_ball_positions['ball_hit'] = 0
        
        # Calculate midpoint of y coordinate
        df_ball_positions['mid_y'] = (df_ball_positions['y1'] + df_ball_positions['y2'])/2
        df_ball_positions['mid_y_rolling_mean'] = df_ball_positions['mid_y'].rolling(window=5, min_periods=1, center=False).mean()
        df_ball_positions['delta_y'] = df_ball_positions['mid_y_rolling_mean'].diff()
        
        minimum_change_frames_for_hit = 25
        for i in range(1, len(df_ball_positions) - int(minimum_change_frames_for_hit*1.2)):
            negative_position_change = df_ball_positions['delta_y'].iloc[i] > 0 and df_ball_positions['delta_y'].iloc[i+1] < 0
            positive_position_change = df_ball_positions['delta_y'].iloc[i] < 0 and df_ball_positions['delta_y'].iloc[i+1] > 0
            
            if negative_position_change or positive_position_change:
                change_count = 0 
                for change_frame in range(i+1, i+int(minimum_change_frames_for_hit*1.2)+1):
                    if change_frame >= len(df_ball_positions):
                        break
                    
                    negative_position_change_following_frame = df_ball_positions['delta_y'].iloc[i] > 0 and df_ball_positions['delta_y'].iloc[change_frame] < 0
                    positive_position_change_following_frame = df_ball_positions['delta_y'].iloc[i] < 0 and df_ball_positions['delta_y'].iloc[change_frame] > 0
                    
                    if negative_position_change and negative_position_change_following_frame:
                        change_count += 1
                    elif positive_position_change and positive_position_change_following_frame:
                        change_count += 1
            
                if change_count > minimum_change_frames_for_hit-1:
                    df_ball_positions.loc[i, 'ball_hit'] = 1
        
        frame_nums_with_ball_hits = df_ball_positions[df_ball_positions['ball_hit']==1].index.tolist()
        
        return frame_nums_with_ball_hits
    
    def draw_ball_circles(self, video_frames, ball_detections):
        output_video_frames = []
        for frame, ball_dict in zip(video_frames, ball_detections):
            for track_id, bbox in ball_dict.items():
                if bbox is not None:
                    # Convert bounding box to circle parameters
                    x1, y1, x2, y2 = map(int, bbox)
                    
                    # Calculate center of the ball
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    
                    # Calculate radius (half the width or height of bounding box)
                    radius = max((x2 - x1) // 2, (y2 - y1) // 2)
                    
                    # Draw the circle
                    cv2.circle(frame, (center_x, center_y), radius, (0, 0, 255), 2)
                    
                    # Optionally add label
                    cv2.putText(frame, f"Ball ID: {track_id}", (center_x, center_y-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    
                output_video_frames.append(frame)
        return output_video_frames

