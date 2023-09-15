import cv2
import numpy as np
import yaml
from roboflow import Roboflow
import datetime
from vcTracking import HeatmapGenerator
from vcVideoHandler import VideoHandler, FrameOverlay

# Read config file
def read_config(file_path):
    with open(file_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    return config_dict

def concatenate_images(img_arrays, titles, m, n):
    """
    Concatenate images into a grid of m rows and n columns with respective titles.
    
    Parameters:
    - img_arrays: List of image arrays.
    - titles: List of image titles.
    - m: Number of rows.
    - n: Number of columns.

    Returns:
    - concatenated image
    """
    
    if len(img_arrays) != m*n:
        raise ValueError("The number of images provided does not match m x n.")
    
    reference_height, reference_width = img_arrays[0].shape[:2]
    
    # Resize images based on the first image's dimensions
    images = [cv2.resize(img, (reference_width, reference_height)) for img in img_arrays]
    
    # Draw titles on the images
    font_scale = 1.5  # Increased font size to 3 times
    font_thickness = 2
    for idx, title in enumerate(titles):
        (text_width, text_height), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        
        # Calculating the starting coordinates for centered text
        start_x = (reference_width - text_width) // 2
        start_y = text_height + 10

        cv2.rectangle(images[idx], (0, 0), (reference_width, text_height + 20), (255, 255, 255), -1)
        cv2.putText(images[idx], title, (start_x, start_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)
    
    # Concatenate images in m x n format
    rows = []
    for i in range(0, len(images), n):
        rows.append(np.hstack(images[i:i+n]))
    
    concatenated_img = np.vstack(rows)
    
    return concatenated_img

# Setup Robboflow Stuff
def setup_roboflow(config):
    # Set up Roboflow Model (using Hosted API now)
    rf = Roboflow(api_key=config['roboflow']['api_key'])
    project = rf.workspace().project(config['roboflow']['model'])
    model = project.version(config['roboflow']['version']).model

    # Retrun model
    return model

# Run roboflow inference
def run_inference(model, frame):
    # Run inference on the frame
    results = model.predict(frame,
                    confidence=60, 
                    overlap=30).json()
    
    return results

# Main
if __name__ == "__main__":
    # Read config file
    config_path = '../config.yaml'  # you can modify this to the path of your config file
    config = read_config(config_path)
    # Show content of config file
    print(config)  

    # Init Videohandler with video path
    video = VideoHandler(config['video']['filename'])

    # Get first frame
    frame = video.get_first_frame()
    overlay = FrameOverlay(frame)

    # Have user draw polygon
    message = 'Draw a polygon border where you will be vacuuming: When done press "d"'
    overlay.center_text_at_top(message)
    points = overlay.draw_polygon(window_name='First Frame')

    # Show user the polygon
    overlay.set_frame(frame)
    overlay.draw_polygon(points)
    prompt_text = "Is this what you want? (y/n)"
    overlay.center_text_at_top(prompt_text, background=True)
    cv2.imshow('First Frame', overlay.get_frame())

    # Have user confirm polygon
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("y"):
            break
        elif key == ord("n"):
            overlay.set_frame(frame)
            points = overlay.draw_polygon('First Frame')
            overlay.set_frame(frame)
            overlay.draw_polygon(points)
            overlay.center_text_at_top(prompt_text, background=True)
            cv2.imshow('First Frame', overlay.get_frame())

    # Destroy window
    cv2.destroyAllWindows()

    # Configure Roboflow model
    model = setup_roboflow(config)

    # Init heatmap / path
    heatmap_gen = HeatmapGenerator(generate_heatmap=True, generate_path=True, generate_binary_mask=True, polygon=points)

    # Loop through video
    heatmap_frame = None
    path_frame = None
    binary_mask = None
    percentage_complete = 0
    out = None
    while True:
        # Get next frame
        frame = video.get_next_frame()
        # Check if frame is valid
        if frame is None:
            break

        # Inference on frame
        results_cv = run_inference(model, frame)
        # Extract results
        results = []
        for prediction in results_cv['predictions']:
            # Results
            results.append([prediction['x']-prediction['width']/2, prediction['y']-prediction['height']/2, prediction['x']+prediction['width']/2, prediction['y']+prediction['height']/2, prediction['class']])

        # Overlay polygon on frame
        overlay = FrameOverlay(frame)
        overlay.draw_polygon(points)

        # Display heatmap / path
        for result in results:
            # Check if result is vacuum:
            if result[-1] != config['video']['object_track']:
                continue

            # Get bounding box
            bbox = result[:4]  # assuming you have bounding box coordinates like this
            # Overlay bbox and label
            overlay.draw_bbox_with_label(bbox, result[-1])

            # Update
            heatmap_frame, path_frame, binary_mask = heatmap_gen.process_frame(frame, bbox)
            # Calculate untouched percentage (this needs to be on only if heatmap is on)
            percentage_complete = 100.0 - heatmap_gen.calculate_untouched_percentage()
            # print(f"Percentage of untouched pixels inside the polygon: {percentage:.2f}%")
            # cv2.imshow('heatmap_frame', heatmap_frame)
            # cv2.imshow('path_frame', path_frame)
            # cv2.imshow('mask_frame', binary_mask)

        # Show frame
        overlay_frame = overlay.get_frame()
        # cv2.imshow('Video', overlay_frame)

        # Concatenate images
        if heatmap_frame is None or path_frame is None or binary_mask is None:
            heatmap_frame = np.zeros_like(frame)
            path_frame = np.zeros_like(frame)
            binary_mask = np.zeros_like(frame)
        concatenated = concatenate_images([overlay_frame, heatmap_frame, path_frame, binary_mask], ["Frame", "Heatmap", "Path", "Coverage: {:.2f}%".format(percentage_complete)], 1, 4)
        cv2.imshow("Concatenated Images", concatenated)

        # Write to video
        if config["output"]["save_output"]:
            if out is None:
                fps = 15
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID' 

                # Get timestamp to append to filenae    
                current_timestamp = datetime.datetime.now()
                formatted_timestamp = current_timestamp.strftime('%Y%m%d_%H%M%S')
                filename_save = config["output"]["save_path"].split('.')[0] + '_' + formatted_timestamp + '.mp4'
                # Create video writer
                out = cv2.VideoWriter(filename_save, fourcc, fps, (concatenated.shape[1], concatenated.shape[0]))
            out.write(concatenated)

        # Press 'q' to quit
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    # Release video and close
    out.release()
    video.release()
    cv2.destroyAllWindows()