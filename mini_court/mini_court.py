import cv2
import numpy as np
import sys
sys.path.append('../')
import constants
from utils import (
    convert_meters_to_pixel_distance,
    convert_pixel_distance_to_meters,
    get_foot_position,
    get_closest_keypoint_index,
    get_height_of_bbox,
    measure_xy_distance,
    get_center_of_bbox,
    measure_distance
)

class MiniCourt():
    def __init__(self,frame):
        self.drawing_rectangle_width = 250 # width of the tennis mini-court
        self.drawing_rectangle_height = 500 # height of the tennis mini-court
        self.buffer = 50 # space between the edge of the video and the beginning of the mini-court
        self.padding_court=20 # space between the beginning of the mini-court and t

        self.set_canvas_background_box_position(frame)
        self.set_mini_court_position()
        self.set_court_drawing_key_points()
        self.set_court_lines()


    def convert_meters_to_pixels(self, meters):
        return convert_meters_to_pixel_distance(meters,
                                                constants.DOUBLE_LINE_WIDTH,
                                                self.court_drawing_width
                                            )

    def set_court_drawing_key_points(self):
        drawing_key_points = [0]*28

        # point 0 
        drawing_key_points[0] , drawing_key_points[1] = int(self.court_start_x), int(self.court_start_y)
        # point 1
        drawing_key_points[2] , drawing_key_points[3] = int(self.court_end_x), int(self.court_start_y)
        # point 2
        drawing_key_points[4] = int(self.court_start_x)
        drawing_key_points[5] = self.court_start_y + self.convert_meters_to_pixels(constants.HALF_COURT_LINE_HEIGHT*2)
        # point 3
        drawing_key_points[6] = drawing_key_points[0] + self.court_drawing_width
        drawing_key_points[7] = drawing_key_points[5] 
        # #point 4
        drawing_key_points[8] = drawing_key_points[0] +  self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[9] = drawing_key_points[1] 
        # #point 5
        drawing_key_points[10] = drawing_key_points[4] + self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[11] = drawing_key_points[5] 
        # #point 6
        drawing_key_points[12] = drawing_key_points[2] - self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[13] = drawing_key_points[3] 
        # #point 7
        drawing_key_points[14] = drawing_key_points[6] - self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[15] = drawing_key_points[7] 
        # #point 8
        drawing_key_points[16] = drawing_key_points[8] 
        drawing_key_points[17] = drawing_key_points[9] + self.convert_meters_to_pixels(constants.NO_MANS_LAND_HEIGHT)
        # # #point 9
        drawing_key_points[18] = drawing_key_points[16] + self.convert_meters_to_pixels(constants.SINGLE_LINE_WIDTH)
        drawing_key_points[19] = drawing_key_points[17] 
        # #point 10
        drawing_key_points[20] = drawing_key_points[10] 
        drawing_key_points[21] = drawing_key_points[11] - self.convert_meters_to_pixels(constants.NO_MANS_LAND_HEIGHT)
        # # #point 11
        drawing_key_points[22] = drawing_key_points[20] +  self.convert_meters_to_pixels(constants.SINGLE_LINE_WIDTH)
        drawing_key_points[23] = drawing_key_points[21] 
        # # #point 12
        drawing_key_points[24] = int((drawing_key_points[16] + drawing_key_points[18])/2)
        drawing_key_points[25] = drawing_key_points[17] 
        # # #point 13
        drawing_key_points[26] = int((drawing_key_points[20] + drawing_key_points[22])/2)
        drawing_key_points[27] = drawing_key_points[21] 

        self.drawing_key_points=drawing_key_points

    def set_court_lines(self):
        self.lines = [
            (0, 2),
            (4, 5),
            (6,7),
            (1,3),
            
            (0,1),
            (8,9),
            (10,11),
            (10,11),
            (2,3)
        ]

    def set_mini_court_position(self):
        self.court_start_x = self.start_x + self.padding_court
        self.court_start_y = self.start_y + self.padding_court
        self.court_end_x = self.end_x - self.padding_court
        self.court_end_y = self.end_y - self.padding_court
        self.court_drawing_width = self.court_end_x - self.court_start_x

    def set_canvas_background_box_position(self,frame):
        frame= frame.copy()

        self.end_x = frame.shape[1] - self.buffer
        self.end_y = self.buffer + self.drawing_rectangle_height
        self.start_x = self.end_x - self.drawing_rectangle_width
        self.start_y = self.end_y - self.drawing_rectangle_height

    def draw_court(self,frame):
        for i in range(0, len(self.drawing_key_points),2):
            x = int(self.drawing_key_points[i])
            y = int(self.drawing_key_points[i+1])
            cv2.circle(frame, (x,y),5, (0,0,255),-1)

        # draw Lines
        for line in self.lines:
            start_point = (int(self.drawing_key_points[line[0]*2]), int(self.drawing_key_points[line[0]*2+1]))
            end_point = (int(self.drawing_key_points[line[1]*2]), int(self.drawing_key_points[line[1]*2+1]))
            cv2.line(frame, start_point, end_point, (0, 0, 0), 2)

        # Draw net
        net_start_point = (self.drawing_key_points[0], int((self.drawing_key_points[1] + self.drawing_key_points[5])/2))
        net_end_point = (self.drawing_key_points[2], int((self.drawing_key_points[1] + self.drawing_key_points[5])/2))
        cv2.line(frame, net_start_point, net_end_point, (255, 0, 0), 2)

        return frame

    def draw_background_rectangle(self,frame):
        shapes = np.zeros_like(frame,np.uint8)
        # Draw the rectangle
        cv2.rectangle(shapes, (self.start_x, self.start_y), (self.end_x, self.end_y), (255, 255, 255), cv2.FILLED)
        out = frame.copy()
        alpha=0.5
        mask = shapes.astype(bool)
        out[mask] = cv2.addWeighted(frame, alpha, shapes, 1 - alpha, 0)[mask]

        return out

    def draw_mini_court(self,frames):
        output_frames = []
        for frame in frames:
            frame = self.draw_background_rectangle(frame)
            frame = self.draw_court(frame)
            output_frames.append(frame)
        return output_frames

    def get_start_point_of_mini_court(self):
        return (self.court_start_x,self.court_start_y)
    def get_width_of_mini_court(self):
        return self.court_drawing_width
    def get_court_drawing_keypoints(self):
        return self.drawing_key_points

    def get_mini_court_coordinates(self,
                                   object_position,
                                   closest_key_point, 
                                   closest_key_point_index, 
                                   player_height_in_pixels,
                                   player_height_in_meters
                                   ):
        
        distance_from_keypoint_x_pixels, distance_from_keypoint_y_pixels = measure_xy_distance(object_position, closest_key_point)

        # Conver pixel distance to meters
        distance_from_keypoint_x_meters = convert_pixel_distance_to_meters(distance_from_keypoint_x_pixels,
                                                                           player_height_in_meters,
                                                                           player_height_in_pixels
                                                                           )
        distance_from_keypoint_y_meters = convert_pixel_distance_to_meters(distance_from_keypoint_y_pixels,
                                                                                player_height_in_meters,
                                                                                player_height_in_pixels
                                                                          )
        
        # Convert to mini court coordinates
        mini_court_x_distance_pixels = self.convert_meters_to_pixels(distance_from_keypoint_x_meters)
        mini_court_y_distance_pixels = self.convert_meters_to_pixels(distance_from_keypoint_y_meters)
        closest_mini_coourt_keypoint = ( self.drawing_key_points[closest_key_point_index*2],
                                        self.drawing_key_points[closest_key_point_index*2+1]
                                        )
        
        mini_court_player_position = (closest_mini_coourt_keypoint[0]+mini_court_x_distance_pixels,
                                      closest_mini_coourt_keypoint[1]+mini_court_y_distance_pixels
                                        )

        return  mini_court_player_position

    def convert_bounding_boxes_to_mini_court_coordinates(self, player_boxes, ball_boxes, original_court_key_points):
        player_heights = {
            1: constants.PLAYER_1_HEIGHT_METERS,
            2: constants.PLAYER_2_HEIGHT_METERS
        }
        
        # Set up multiple homography matrices for different court regions
        self.setup_region_homographies(original_court_key_points)
        
        # Analyze ball trajectory to detect bounces
        bounce_frames = self.detect_bounce_frames(ball_boxes)
        
        output_player_boxes = []
        output_ball_boxes = []
        
        # Store ball positions for trajectory analysis and smoothing
        raw_ball_positions = []
        
        # First pass: calculate initial positions
        for frame_num, player_bbox in enumerate(player_boxes):
            # Handle players
            output_player_bboxes_dict = {}
            for player_id, bbox in player_bbox.items():
                foot_position = get_foot_position(bbox)
                mini_court_player_position = self.transform_to_mini_court(foot_position, 'general')
                output_player_bboxes_dict[player_id] = mini_court_player_position
            
            # Handle ball
            ball_box = ball_boxes[frame_num].get(1)
            if ball_box is not None:
                ball_position = get_center_of_bbox(ball_box)
                
                # Determine if we're near a baseline
                is_near_baseline = self.is_position_near_baseline(ball_position, original_court_key_points)
                
                # Choose appropriate transformation
                if frame_num in bounce_frames:
                    region = 'bounce'
                elif is_near_baseline:
                    region = 'baseline'
                else:
                    region = 'general'
                
                # Transform with selected region-specific homography
                mini_court_ball_position = self.transform_to_mini_court(ball_position, region)
                raw_ball_positions.append((frame_num, mini_court_ball_position))
                output_ball_boxes.append({1: mini_court_ball_position})
            else:
                output_ball_boxes.append({1: None})
                raw_ball_positions.append((frame_num, None))
            
            output_player_boxes.append(output_player_bboxes_dict)
        
        # Second pass: smooth ball trajectory and apply bounce corrections
        smoothed_ball_positions = self.smooth_ball_trajectory(raw_ball_positions, bounce_frames)
        
        # Update with smoothed positions
        for frame_num, position in smoothed_ball_positions:
            if 0 <= frame_num < len(output_ball_boxes):
                output_ball_boxes[frame_num] = {1: position}
        
        return output_player_boxes, output_ball_boxes

    def setup_region_homographies(self, original_court_key_points):
        """Set up multiple homography matrices for different court regions"""
        # Extract keypoints
        all_keypoints = []
        for i in range(0, len(original_court_key_points), 2):
            all_keypoints.append((original_court_key_points[i], original_court_key_points[i+1]))
        
        # Main court corners
        main_corners_src = np.float32([
            [all_keypoints[0][0], all_keypoints[0][1]],   # Bottom left
            [all_keypoints[1][0], all_keypoints[1][1]],   # Bottom right
            [all_keypoints[2][0], all_keypoints[2][1]],   # Top left
            [all_keypoints[3][0], all_keypoints[3][1]]    # Top right
        ])
        
        # Corresponding points in mini-court
        main_corners_dst = np.float32([
            [self.drawing_key_points[0], self.drawing_key_points[1]],   # Bottom left
            [self.drawing_key_points[2], self.drawing_key_points[3]],   # Bottom right
            [self.drawing_key_points[4], self.drawing_key_points[5]],   # Top left
            [self.drawing_key_points[6], self.drawing_key_points[7]]    # Top right
        ])
        
        # Calculate general homography
        self.homographies = {}
        self.homographies['general'], _ = cv2.findHomography(main_corners_src, main_corners_dst)
        
        # Create baseline-specific homography with more keypoints near baselines
        baseline_src = np.float32([
            # Include more baseline points
            [all_keypoints[0][0], all_keypoints[0][1]],      # Bottom left
            [all_keypoints[1][0], all_keypoints[1][1]],      # Bottom right
            [(all_keypoints[0][0] + all_keypoints[1][0])/2, all_keypoints[0][1]],  # Bottom middle
            [all_keypoints[2][0], all_keypoints[2][1]],      # Top left
            [all_keypoints[3][0], all_keypoints[3][1]],      # Top right
            [(all_keypoints[2][0] + all_keypoints[3][0])/2, all_keypoints[2][1]]   # Top middle
        ])
        
        baseline_dst = np.float32([
            [self.drawing_key_points[0], self.drawing_key_points[1]],   # Bottom left
            [self.drawing_key_points[2], self.drawing_key_points[3]],   # Bottom right
            [(self.drawing_key_points[0] + self.drawing_key_points[2])/2, self.drawing_key_points[1]],  # Bottom middle
            [self.drawing_key_points[4], self.drawing_key_points[5]],   # Top left
            [self.drawing_key_points[6], self.drawing_key_points[7]],   # Top right
            [(self.drawing_key_points[4] + self.drawing_key_points[6])/2, self.drawing_key_points[5]]   # Top middle
        ])
        
        self.homographies['baseline'], _ = cv2.findHomography(baseline_src, baseline_dst)
        
        # Use a specialized bounce homography that anchors to the court surface
        self.homographies['bounce'] = self.homographies['general']  # Start with general homography as base

    def transform_to_mini_court(self, point, region='general'):
        """Transform a point using the appropriate homography matrix"""
        if point is None:
            return None
        
        # Get the correct homography matrix
        H = self.homographies.get(region, self.homographies['general'])
        
        # Reshape point for transformation
        point_array = np.array([[[point[0], point[1]]]], dtype=np.float32)
        
        # Apply homography transformation
        transformed = cv2.perspectiveTransform(point_array, H)
        
        # Extract coordinates
        x, y = transformed[0][0]
        
        # Ensure point is within mini-court bounds
        x = max(self.court_start_x, min(x, self.court_end_x))
        y = max(self.court_start_y, min(y, self.court_end_y))
        
        return (x, y)

    def is_position_near_baseline(self, position, original_court_key_points):
        """Check if a position is near a baseline"""
        if position is None:
            return False
        
        # Define baselines
        bottom_baseline = [(original_court_key_points[0], original_court_key_points[1]), 
                        (original_court_key_points[2], original_court_key_points[3])]
        
        top_baseline = [(original_court_key_points[4], original_court_key_points[5]), 
                        (original_court_key_points[6], original_court_key_points[7])]
        
        # Calculate distance to baselines
        bottom_dist = self.distance_to_line(position, bottom_baseline)
        top_dist = self.distance_to_line(position, top_baseline)
        
        # Threshold for "near baseline" (adjust as needed)
        threshold = 50  # pixels
        
        return bottom_dist < threshold or top_dist < threshold

    def distance_to_line(self, point, line):
        """Calculate the distance from a point to a line segment"""
        x, y = point
        (x1, y1), (x2, y2) = line
        
        # Calculate perpendicular distance
        num = abs((y2-y1)*x - (x2-x1)*y + x2*y1 - y2*x1)
        den = np.sqrt((y2-y1)**2 + (x2-x1)**2)
        
        return num/den if den != 0 else float('inf')

    def detect_bounce_frames(self, ball_boxes):
        """Detect frames where the ball likely bounces"""
        bounce_frames = []
        
        # Extract y-coordinates and convert to pandas for analysis
        y_positions = []
        for i, box_dict in enumerate(ball_boxes):
            box = box_dict.get(1)
            if box is not None:
                center = get_center_of_bbox(box)
                y_positions.append((i, center[1]))
        
        if not y_positions:
            return bounce_frames
        
        # Convert to pandas for easier analysis
        import pandas as pd
        df = pd.DataFrame(y_positions, columns=['frame', 'y'])
        
        # Calculate derivatives to find direction changes
        df['y_diff'] = df['y'].diff()
        df['y_diff2'] = df['y_diff'].diff()  # Second derivative
        
        # Look for sign changes in first derivative (direction change)
        # and large magnitude in second derivative (rapid direction change)
        for i in range(2, len(df)-2):
            if (df['y_diff'].iloc[i-1] * df['y_diff'].iloc[i+1] < 0 and 
                abs(df['y_diff2'].iloc[i]) > 5):  # Threshold may need adjustment
                bounce_frames.append(df['frame'].iloc[i])
        
        return bounce_frames

    def smooth_ball_trajectory(self, positions, bounce_frames):
        """Apply smoothing to ball trajectory with special handling for bounces"""
        if len(positions) < 3:
            return positions
        
        # Extract valid positions
        valid_positions = [(frame, pos) for frame, pos in positions if pos is not None]
        if len(valid_positions) < 3:
            return positions
        
        # Create separate arrays for x and y coordinates
        frames = [frame for frame, pos in valid_positions]
        x_coords = [pos[0] for _, pos in valid_positions]
        y_coords = [pos[1] for _, pos in valid_positions]
        
        # Apply smoothing with Savitzky-Golay filter (preserves peaks better than moving average)
        try:
            from scipy.signal import savgol_filter
            window_length = min(11, len(x_coords) - 2)  # Must be odd and less than data length
            if window_length >= 3:  # Minimum required window length
                # Ensure window_length is odd
                window_length = window_length if window_length % 2 == 1 else window_length - 1
                
                x_smooth = savgol_filter(x_coords, window_length, 2)
                y_smooth = savgol_filter(y_coords, window_length, 2)
                
                # Create smoothed positions
                smoothed = [(frame, (x, y)) for frame, x, y in zip(frames, x_smooth, y_smooth)]
                
                # Handle bounce frames specially - use actual y value at exact bounce
                for bounce_frame in bounce_frames:
                    for i, (frame, _) in enumerate(valid_positions):
                        if frame == bounce_frame:
                            # Keep original bounce position but with smoothed x
                            orig_pos = valid_positions[i][1]
                            smooth_pos = smoothed[i][1]
                            smoothed[i] = (frame, (smooth_pos[0], orig_pos[1]))
                
                # Rebuild the full list with None values where appropriate
                result = []
                smooth_index = 0
                for frame, pos in positions:
                    if pos is not None and smooth_index < len(smoothed):
                        result.append((frame, smoothed[smooth_index][1]))
                        smooth_index += 1
                    else:
                        result.append((frame, None))
                
                return result
        except ImportError:
            # If scipy is not available, return original positions
            return positions
        
        # Fallback if smoothing fails
        return positions

    def draw_points_on_mini_court(self,frames,postions, color=(0,255,0)):
        for frame_num, frame in enumerate(frames):
            for _, position in postions[frame_num].items():
                x,y = position
                x= int(x)
                y= int(y)
                cv2.circle(frame, (x,y), 5, color, -1)
        return frames
