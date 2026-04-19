import cv2
import numpy as np
import tensorflow as tf

MODEL_PATH = "model/person_detector.h5"
model = tf.keras.models.load_model(MODEL_PATH)

IMG_SIZE = 128
CONF_THRESHOLD = 0.75
NMS_IOU_THRESHOLD = 0.25
PAD = 200

WINDOW_SIZES = [
    (160, 320),
    (200, 400),
]

def non_max_suppression(boxes, scores, iou_threshold=0.25):
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    scores = np.array(scores)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)

        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou < iou_threshold)[0]
        order = order[inds + 1]

    return keep


def detect_persons(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("err")
        return

    image_padded = cv2.copyMakeBorder(
        image, PAD, PAD, PAD, PAD,
        cv2.BORDER_REFLECT
    )

    output = image.copy()
    H, W = image_padded.shape[:2]
    orig_H, orig_W = image.shape[:2]

    boxes = []
    confidences = []


    for (win_w, win_h) in WINDOW_SIZES:
        step_x = win_w // 3
        step_y = win_h // 3

        for y in range(0, H - win_h, step_y):
            for x in range(0, W - win_w, step_x):
                window = image_padded[y:y + win_h, x:x + win_w]
                resized = cv2.resize(window, (IMG_SIZE, IMG_SIZE))
                normalized = resized / 255.0
                input_img = np.expand_dims(normalized, axis=0)

                prediction = model.predict(input_img, verbose=0)[0][0]

                if prediction >= CONF_THRESHOLD:
                    w_box, h_box = win_w, win_h
                    aspect_ratio = h_box / w_box
                    if 1.5 < aspect_ratio < 3.5:
                        boxes.append([x - PAD, y - PAD, w_box, h_box])
                        confidences.append(float(prediction))

    if not boxes:
        return

    keep_indices = non_max_suppression(boxes, confidences, NMS_IOU_THRESHOLD)

    keep_indices = sorted(keep_indices, key=lambda i: confidences[i], reverse=True)
    final_keep = []
    for i in keep_indices:
        x1, y1, w1, h1 = boxes[i]
        cx1, cy1 = x1 + w1 // 2, y1 + h1 // 2
        too_close = False
        for j in final_keep:
            x2, y2, w2, h2 = boxes[j]
            cx2, cy2 = x2 + w2 // 2, y2 + h2 // 2
            dist = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
            if dist < (w1 + w2) * 0.5:
                too_close = True
                break
        if not too_close:
            final_keep.append(i)


    for i in final_keep:
        x, y, w, h = boxes[i]
        score = confidences[i]

        x = max(0, x)
        y = max(0, y)
        x2 = min(orig_W, x + w)
        y2 = min(orig_H, y + h)

        cv2.rectangle(output, (x, y), (x2, y2), (0, 0, 255), 3)
        cv2.putText(
            output,
            f"Person: {score:.2f}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    cv2.imshow("Detection", output)
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
detect_persons("testt.png")