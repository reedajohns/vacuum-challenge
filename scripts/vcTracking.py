import cv2
import numpy as np
import yaml
from roboflow import Roboflow
import datetime

# Heatmap / Path Class
class HeatmapGenerator:
    def __init__(self, generate_heatmap=True, generate_path=True, generate_binary_mask=True, polygon=None):
        self.heatmap_width = 0
        self.heatmap_height = 0
        self.accumulator = None
        self.path = None
        self.object_centroids = []
        self.binary_mask = None
        # Which outputs to generate
        self.generate_heatmap = generate_heatmap
        self.generate_path = generate_path
        self.generate_binary_mask = generate_binary_mask
        # Polygon
        self.polygon = None if polygon is None else np.array(polygon, dtype=np.int32)  # Convert the polygon points to numpy array or default to None

    @staticmethod
    def overlay_image_alpha(img, heatmap, alpha=0.5):
        img_float = img.astype(np.float32) / 255.0
        heatmap_float = heatmap.astype(np.float32) / 255.0
        blended = cv2.addWeighted(img_float, 1 - alpha, heatmap_float, alpha, 0)
        return (blended * 255).astype(np.uint8)

    def init_heatmap_path(self, frame):
        self.heatmap_width = frame.shape[1]
        self.heatmap_height = frame.shape[0]
        self.accumulator = np.zeros((self.heatmap_height, self.heatmap_width), dtype=np.float32)
        self.path = np.zeros((self.heatmap_height, self.heatmap_width, 3), dtype=np.uint8)
        # Initialize binary_mask if the feature is turned on (Do this to others)
        if self.generate_binary_mask:
            self.binary_mask = np.zeros((self.heatmap_height, self.heatmap_width, 3), dtype=np.uint8)
            self.binary_mask[:] = (128, 128, 128)  # Start with gray outside

            # If no polygon was provided, consider the entire frame as the polygon
            if self.polygon is None:
                self.polygon = np.array([
                    [0, 0],
                    [0, self.heatmap_height-1],
                    [self.heatmap_width-1, self.heatmap_height-1],
                    [self.heatmap_width-1, 0]
                ], dtype=np.int32)

            # Fill polygon area with green  
            cv2.fillPoly(self.binary_mask, [self.polygon], (0, 255, 0))

    def update_accumulator(self, x0, y0, x1, y1):
        centroid = ((x0 + x1) // 2, (y0 + y1) // 2)
        self.object_centroids.append(centroid)
    
        # Ensure bounding box values are integers
        self.accumulator[y0:y1, x0:x1] += 1

        # Update binary mask
        if self.generate_binary_mask:
            # Set the pixels inside the bounding box and inside the polygon to black
            mask_inside_polygon = np.zeros_like(self.binary_mask[:, :, 0], dtype=np.uint8)
            cv2.fillPoly(mask_inside_polygon, [self.polygon], 255)
            mask = (self.binary_mask[y0:y1, x0:x1] == [0, 255, 0]).all(axis=2) & (mask_inside_polygon[y0:y1, x0:x1] == 255)
            self.binary_mask[y0:y1, x0:x1][mask] = [0, 0, 0]

            # Set the pixels inside the bounding box and outside the polygon to red
            mask_outside_polygon = (self.binary_mask[y0:y1, x0:x1] == [128, 128, 128]).all(axis=2)
            self.binary_mask[y0:y1, x0:x1][mask_outside_polygon] = [0, 0, 255]

            # cv2.imshow('mask', self.binary_mask)
            # cv2.waitKey(1)
            # input('bru')

    def draw_path(self, frame):
        if not self.generate_path:
            return None

        path_frame = frame.copy()
        num_centroids = len(self.object_centroids)
        for i in range(1, num_centroids):
            alpha = i / num_centroids
            color = (int(255 * (1-alpha)), 0, int(255 * alpha))
            cv2.line(path_frame, self.object_centroids[i-1], self.object_centroids[i], color, 6)
        return path_frame
    
    def calculate_untouched_percentage(self):
        # 1. Create a binary mask for the polygon area
        polygon_mask = np.zeros_like(self.binary_mask[:, :, 0], dtype=np.uint8)
        cv2.fillPoly(polygon_mask, [self.polygon], 1)
        
        # 2. Extract the green channel of our binary_mask
        green_channel = self.binary_mask[:, :, 1]
        
        # 3. Multiply the polygon_mask with the green channel
        green_inside_polygon = polygon_mask * green_channel
        
        # Count untouched (green) pixels and total pixels inside the polygon
        untouched_pixel_count = np.sum(green_inside_polygon == 255)
        total_pixel_count = np.sum(polygon_mask == 1)
        
        # 4. Calculate the percentage
        if total_pixel_count == 0:
            return 0  # Avoid division by zero
        untouched_percentage = (untouched_pixel_count / total_pixel_count) * 100
        
        return untouched_percentage

    def process_frame(self, frame, bbox):
        if self.accumulator is None:
            self.init_heatmap_path(frame)

        x0, y0, x1, y1 = bbox
        # Ensure bounding box values are integers
        x0, y0, x1, y1 = map(int, [x0, y0, x1, y1])
        self.update_accumulator(x0, y0, x1, y1)

        heatmap_frame = None
        if self.generate_heatmap:
            heatmap = cv2.applyColorMap(np.uint8(255 * self.accumulator / self.accumulator.max()), cv2.COLORMAP_JET)
            heatmap_frame = self.overlay_image_alpha(frame, heatmap, alpha=0.7)

        path_frame = self.draw_path(frame)

        # Return binary mask frame or None depending on the toggle
        binary_mask_frame = self.binary_mask if self.generate_binary_mask else None

        return heatmap_frame, path_frame, binary_mask_frame