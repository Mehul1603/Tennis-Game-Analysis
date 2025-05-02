from utils import (read_video, 
                   save_video,
                   measure_distance,
                   draw_player_stats,
                   convert_pixel_distance_to_meters
                   )
from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector
from mini_court import MiniCourt
import cv2
import constants
import numpy as np
import pandas as pd
from copy import deepcopy

def main():
    # Read video
    video_path = 'input_videos/input_video.mp4'

    # Extract frames
    frames = read_video(video_path)

    # Detect players
    player_tracker = PlayerTracker(model_path='yolo11x')
    player_detections = player_tracker.detect_frames(frames, read_from_stub=True, stub_path='./tracker_stubs/player_detections.pkl')

    # Detect ball using TrackNet model
    ball_tracker = BallTracker(model_path='model_best.pt', use_tracknet=True)
    ball_detections = ball_tracker.detect_frames(frames, read_from_stub=True, stub_path='tracker_stubs/ball_detections.pkl')

    # Draw court lines
    court_line_model_path = "models/keypoints_model.pth"
    court_line_detector = CourtLineDetector(court_line_model_path)
    keypoints = court_line_detector.predict(frames[0])

    # Filter Players
    filtered_player_detections = player_tracker.choose_and_filter_players(keypoints, player_detections)

    # Generate mini court
    mini_court = MiniCourt(frames[0])

    # Detect ball shots
    ball_shot_frames= ball_tracker.get_ball_shot_frames(ball_detections)

    # Convert positions to mini court positions
    player_mini_court_detections, ball_mini_court_detections = mini_court.convert_bounding_boxes_to_mini_court_coordinates(filtered_player_detections, 
                                                                                                          ball_detections,
                                                                                                          keypoints)

    player_stats_data = [{
        'frame_num': 0,
        'player_1_number_of_shots': 0,
        'player_1_total_shot_speed': 0,
        'player_1_last_shot_speed': 0,
        'player_1_total_player_speed': 0,
        'player_1_last_player_speed': 0,
        'player_2_number_of_shots': 0,
        'player_2_total_shot_speed': 0,
        'player_2_last_shot_speed': 0,
        'player_2_total_player_speed': 0,
        'player_2_last_player_speed': 0,
    }]
    
    # Create a mapping of shot frames for easy lookup
    shot_frame_map = {frame: idx for idx, frame in enumerate(ball_shot_frames)}
    
    # Window size for calculating player speeds
    speed_window = 5  # frames
    
    # Process each frame (starting from frame 1, since frame 0 is already in the list)
    for frame_num in range(1, len(frames)):
        # Create a new stats entry based on the previous one
        prev_stats = player_stats_data[-1]
        current_stats = deepcopy(prev_stats)
        current_stats['frame_num'] = frame_num
        
        # Check if this is a ball shot frame
        if frame_num in shot_frame_map and shot_frame_map[frame_num] < len(ball_shot_frames) - 1:
            shot_idx = shot_frame_map[frame_num]
            start_frame = ball_shot_frames[shot_idx]
            end_frame = ball_shot_frames[shot_idx + 1]
            
            # Calculate ball shot info
            ball_shot_time_in_seconds = (end_frame - start_frame) / 24  # 24fps
            
            distance_covered_by_ball_pixels = measure_distance(
                ball_mini_court_detections[start_frame][1],
                ball_mini_court_detections[end_frame][1]
            )
            distance_covered_by_ball_meters = convert_pixel_distance_to_meters(
                distance_covered_by_ball_pixels,
                constants.DOUBLE_LINE_WIDTH,
                mini_court.get_width_of_mini_court()
            )
            
            # Speed of the ball shot in km/h
            speed_of_ball_shot = distance_covered_by_ball_meters / ball_shot_time_in_seconds * 3.6
            
            # Determine which player shot the ball
            player_positions = player_mini_court_detections[start_frame]
            player_shot_ball = min(
                player_positions.keys(), 
                key=lambda player_id: measure_distance(
                    player_positions[player_id],
                    ball_mini_court_detections[start_frame][1]
                )
            )
            
            # Update shot stats ONLY at shot frames
            current_stats[f'player_{player_shot_ball}_number_of_shots'] += 1
            current_stats[f'player_{player_shot_ball}_total_shot_speed'] += speed_of_ball_shot
            current_stats[f'player_{player_shot_ball}_last_shot_speed'] = speed_of_ball_shot
        
        # Calculate player speeds for EVERY frame
        for player_id in [1, 2]:
            if frame_num > speed_window:
                # Get positions from sliding window
                start_pos = player_mini_court_detections[frame_num - speed_window][player_id]
                end_pos = player_mini_court_detections[frame_num][player_id]
                
                # Calculate distance and speed
                distance_pixels = measure_distance(start_pos, end_pos)
                distance_meters = convert_pixel_distance_to_meters(
                    distance_pixels,
                    constants.DOUBLE_LINE_WIDTH,
                    mini_court.get_width_of_mini_court()
                )
                
                time_window = speed_window / 24  # 24fps
                current_speed = distance_meters / time_window * 3.6
                
                # Update current player speed 
                current_stats[f'player_{player_id}_last_player_speed'] = current_speed
                current_stats[f'player_{player_id}_total_player_speed'] += current_speed
        
        # Add this frame's stats to the list
        player_stats_data.append(current_stats)
    
    # Convert to DataFrame
    player_stats_data_df = pd.DataFrame(player_stats_data)
    
    # Calculate averages with safe division
    for player_id in [1, 2]:
        # Average shot speed
        player_stats_data_df[f'player_{player_id}_average_shot_speed'] = (
            player_stats_data_df[f'player_{player_id}_total_shot_speed'] / 
            player_stats_data_df[f'player_{player_id}_number_of_shots'].replace(0, np.nan)
        ).fillna(0)
        
        # Average player speed - calculate as cumulative average
        player_stats_data_df[f'player_{player_id}_average_player_speed'] = (
            player_stats_data_df[f'player_{player_id}_total_player_speed'] / 
            player_stats_data_df.index  # Use frame index (adding 1 implicitly handles division by zero)
        ).fillna(0)


    # Draw output
    ## Draw Player Bounding Boxes
    output_frames= player_tracker.draw_bboxes(frames, player_detections)
    output_frames= ball_tracker.draw_ball_circles(output_frames, ball_detections)

    ## Draw court Keypoints
    output_frames  = court_line_detector.draw_keypoints_on_video(output_frames, keypoints)

    # Draw Mini Court
    output_frames = mini_court.draw_mini_court(output_frames)
    output_frames = mini_court.draw_points_on_mini_court(output_frames,player_mini_court_detections)
    output_frames = mini_court.draw_points_on_mini_court(output_frames,ball_mini_court_detections, color=(0,255,255))    

    # Draw Player Stats
    output_frames = draw_player_stats(output_frames,player_stats_data_df)

    # Create video from frames
    save_video(output_frames, 'output_videos/video.avi')

if __name__ == '__main__':
    main()