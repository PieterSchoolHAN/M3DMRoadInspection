import cv2
import torch
from yolov7.seg.segment.custom_predict import load_model, run_inference
import os


def annotate_image(image, detections):
    """
    Draw bounding boxes and labels on the image.
    """
    if detections is None or len(detections) == 0:
        return image

    for det in detections:
        bbox = det[:4].tolist()
        confidence = det[4].item()
        cls = int(det[5].item())

        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label = f"Class: {cls} Conf: {confidence:.2f}"
        label_position = (x1, y1 - 10 if y1 - 10 > 10 else y1 + 10)
        cv2.putText(image, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return image


def annotate_images_batch(results):
    """
    Annotate a batch of frames.
    """
    annotated_images = []

    for path, det, _, im0s in results:
        annotated_img = annotate_image(im0s, det)
        annotated_images.append((path, annotated_img))

    return annotated_images


def process_video(video_path, output_video_path, model, conf_thres=0.5, iou_thres=0.45):
    """
    Process a video file, run YOLOv7 inference, and save the annotated video.
    """
    # Open video source
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
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
        annotated_images_batch = annotate_images_batch(results)
        for _, annotated_frame in annotated_images_batch:
            out.write(annotated_frame)  # Write annotated frame to output video

        processed_frames += 1
        print(f"Processed {processed_frames}/{total_frames} frames", end="\r")

    # Release resources
    cap.release()
    out.release()
    print("\nProcessing complete. Video saved.")


if __name__ == "__main__":
    # Example usage
    video_path = r"C:\Users\jurri\Pictures\vid_dem\istockphoto-937868038-640_adpp_is.mp4"
    output_video_path = r"C:\Users\jurri\Pictures\vid_dem\output\output.mp4"
    model, _, _, _ = load_model(
            weights=r"yolov7\seg\runs\train-seg\custom2\weights\best.pt",
            device="cpu",
        )
    model.eval()
    process_video(video_path, output_video_path, model)
