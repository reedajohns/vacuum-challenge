import cv2
import numpy as np
import yaml
from roboflow import Roboflow
import datetime

class VideoHandler:
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError("Unable to open video file.")
        self.current_frame = None

    def get_first_frame(self):
        success, frame = self.cap.read()
        if success:
            self.current_frame = frame
            return frame
        else:
            return None

    def get_next_frame(self):
        success, frame = self.cap.read()
        if success:
            self.current_frame = frame
            return frame
        return None

    def release(self):
        self.cap.release()

class FrameOverlay:
    def __init__(self, frame):
        self.frame = frame.copy()
        self.frame_original = frame.copy()

    def draw_polygon(self, points=None, window_name='Draw Polygon'):
        if points is None:
            return self._interactive_draw_polygon(window_name)
        else:
            if len(points) > 1:
                cv2.polylines(self.frame, [np.array(points)], isClosed=True, color=(255, 50, 0), thickness=6)
                self._shade_polygon(points)

    def _interactive_draw_polygon(self, window_name):
    # def draw_polygon(self, window_name='Draw Polygon'):
        def mouse_event(event, x, y, flags, param):
            nonlocal drawing
            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = "True"
                points.append((x, y))
                if len(points) > 1:
                    cv2.line(self.frame, points[-2], points[-1], (255, 50, 0), 6)
                cv2.circle(self.frame, (x, y), 8, (255, 0, 0), -1)
                cv2.imshow(window_name, self.frame)
        
        points = []
        drawing = False
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, mouse_event)
        
        while True:
            cv2.imshow(window_name, self.frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("d"):
                if len(points) > 1:
                    cv2.line(self.frame, points[-1], points[0], (0, 0, 255), 2)
                break
        cv2.destroyAllWindows()
        return points
    
    def _shade_polygon(self, points, shading_color=(128, 128, 128), opacity=0.4):
        # Create a blank image
        overlay = np.zeros_like(self.frame)
        
        # Draw the filled polygon on the blank image
        cv2.fillPoly(overlay, [np.array(points)], shading_color)
        
        # Blend the original frame and the shaded polygon image
        cv2.addWeighted(overlay, opacity, self.frame, 1 - opacity, 0, self.frame)

    def center_text_at_top(self, text, font=cv2.FONT_HERSHEY_SIMPLEX, scale=1, color=(255, 0, 0), thickness=2, background=True):
        (text_width, text_height), _ = cv2.getTextSize(text, font, scale, thickness)
        while text_width > self.frame.shape[1]:
            scale -= 0.1
            (text_width, text_height), _ = cv2.getTextSize(text, font, scale, thickness)
    
        start_x = (self.frame.shape[1] - text_width) // 2
        start_y = text_height + 10
        
        if background:
            cv2.rectangle(self.frame, (start_x - 5, start_y - text_height - 5), (start_x + text_width + 5, start_y + 5), (255, 255, 255), -1)
        
        cv2.putText(self.frame, text, (start_x, start_y), font, scale, color, thickness)

    def draw_bbox_with_label(self, bbox, label, box_color=(0, 255, 0), text_color=(255, 255, 255), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.5, font_thickness=1):
        """
        Draw a bounding box and label on the image frame.

        Parameters:
        - bbox: Tuple of (x0, y0, x1, y1) where (x0, y0) is the top-left coordinate and (x1, y1) is the bottom-right coordinate of the box.
        - label: String label to be drawn above the bounding box.
        - box_color: Color of the bounding box. Default is green.
        - text_color: Color of the label text. Default is white.
        - font: Font to be used for the label text. Default is cv2.FONT_HERSHEY_SIMPLEX.
        - font_scale: Font scale factor. Default is 0.5.
        - font_thickness: Thickness of the font. Default is 1.
        """
        
        # Draw bounding box
        x0, y0, x1, y1 = bbox
        x0, y0, x1, y1 = map(int, [x0, y0, x1, y1])
        cv2.rectangle(self.frame, (x0, y0), (x1, y1), box_color, 2)

        # Calculate the size of the text
        (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, font_thickness)

        # Draw a filled rectangle behind the text for better visibility
        cv2.rectangle(self.frame, (x0, y0 - text_height - 5), (x0 + text_width + 5, y0), box_color, -1)

        # Draw label above the bounding box
        cv2.putText(self.frame, label, (x0, y0 - 5), font, font_scale, text_color, font_thickness)

    def get_frame(self):
        return self.frame

    def set_frame(self, frame):
        self.frame = frame.copy()