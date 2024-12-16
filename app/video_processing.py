import cv2
import torch
from yolov7.seg.segment.modular_predict import load_model, run_inference
import os

def process_video(video_path, output_video_path, model, annotate_images_batch_func, update_progress_bar_func, conf_thres=0.5, iou_thres=0.45):
    """
    Process a video file, run YOLOv7 inference, and save the annotated video.
    """
    # Open video source
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Setup video writer for the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 output
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (640, 640))

    print(f"Processing video: {video_path}")
    print(f"Saving output to: {output_video_path}")

    processed_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        model_input_size = (640, 640)  # Replace with the correct size for your model

        if not ret:
            print("Finished processing or encountered an error.")
            break
        frame = cv2.resize(frame, model_input_size)

        # Run inference on the frame
        results = run_inference(
            model=model,
            source=frame,
            conf_thres=conf_thres,
            iou_thres=iou_thres
        )

        # Annotate the frame
        annotated_images_batch = annotate_images_batch_func(results)
        for _, annotated_frame in annotated_images_batch:
            out.write(annotated_frame)  # Write annotated frame to output video

        processed_frames += 1
        print(f"Processed {processed_frames}/{total_frames} frames", end="\r")
        update_progress_bar_func(processed_frames, total_frames)

    # Release resources
    cap.release()
    out.release()
    print("\nProcessing complete. Video saved.")


if __name__ == "__main__":
    # Example usage
    extension = '.mp4'
    video_path_base = r"C:\Users\jurri\Pictures\vid_dem\istockphoto-937868038-640_adpp_is"
    output_video_path = f"{video_path_base}/output{extension}"
    video_path = f"{video_path_base}{extension}"
    
    model, _, _, _ = load_model(
            weights=r"yolov7\seg\runs\train-seg\custom2\weights\best.pt",
            device="cpu",
        )
    model.eval()
    process_video(video_path, output_video_path, model)
