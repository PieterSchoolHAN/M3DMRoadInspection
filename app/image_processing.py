import cv2

def annotate_image(image, detections):
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
    annotated_images = []

    for path, det, _, im0s in results:
        annotated_img = annotate_image(im0s, det)
        annotated_images.append((path, annotated_img))

    return annotated_images