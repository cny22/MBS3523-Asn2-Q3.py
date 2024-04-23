import cv2
import numpy as np

confThreshold = 0.8
classesFile = 'C:/Users/230399178/PycharmProjects/20240220_Serial_LED/coco.names'
classes = ()
with open(classesFile, 'r') as f:
    classes = f.read().splitlines()

fruit_labels = ['apple', 'banana', 'orange']
fruit_prices = {'apple': 8.00, 'banana': 4.00, 'orange': 6.00}
fruit_colors = {"apple": (0, 255, 0), "banana": (0, 255, 255), "orange": (0, 165, 255)}

net = cv2.dnn.readNetFromDarknet('darknet-master/cfg/yolov3.cfg', 'C:/Users/230399178/PycharmProjects/20240220_Serial_LED/darknet-master/yolov3.weights')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

cam = cv2.VideoCapture(0)
window_name = 'Fruit Detection'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

img = cv2.VideoCapture()

while True:
    success, img = cam.read()
    height, width, ch = img.shape

    # Reset fruit counts for each frame
    fruit_counts = {"apple": 0, "banana": 0, "orange": 0}
    total_price = 0

    blob = cv2.dnn.blobFromImage(img, 1 / 255, (320, 320), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)

    layerNames = net.getLayerNames()
    output_layers_names = net.getUnconnectedOutLayersNames()
    LayerOutputs = net.forward(output_layers_names)

    bboxes = []
    confidences = []
    class_ids = []

    for output in LayerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confThreshold and classes[class_id] in fruit_labels:  # Check if detected object is a fruit
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                bboxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(bboxes, confidences, confThreshold, 0.4)

    font = cv2.FONT_HERSHEY_PLAIN
    total_fruits = 0
    total_price = 0

    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = bboxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i], 2))
            color = fruit_colors[label] if label in fruit_colors else (0, 0, 255)  # Red color for unknown fruits

            # Draw bounding box
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

            # Draw fruit name and confidence
            cv2.putText(img, label + " " + confidence, (x, y + 20), font, 1, (255, 255, 255), 1)

            # Update fruit counts
            fruit_counts[label] += 1

            # Calculate total price
            total_price += fruit_prices.get(label, 0)

            # Draw bounding box for fruit name
            cv2.rectangle(img, (x, y - 30), (x + len(label) * 12, y), color, -1)
            cv2.putText(img, label, (x, y - 10), font, 1, (255, 255, 255), 1)
    else:
        # Display "No Fruit Detected" message on top left corner
        cv2.putText(img, "No Fruit Detected", (50, 50), font, 1.5, (0, 0, 255), 2)

    # Display fruit counts and total price on upper right corner
    fruit_text = ", ".join([f"{count} {fruit}" for fruit, count in fruit_counts.items() if count > 0])
    text_size = cv2.getTextSize(fruit_text, font, 1, 2)[0]
    text_x = width - text_size[0] - 200
    cv2.putText(img, "Fruit Counts: " + fruit_text, (text_x, 60), font, 1, (255, 0, 0), 2)

    # Generate total price text
    total_price_text = ""
    for fruit, count in fruit_counts.items():
        if count > 0:
            total_price_text += f"{count} {fruit} x ${fruit_prices[fruit]:.2f} + "
    total_price_text = total_price_text.rstrip(" + ")  # Remove trailing "+" sign

    # Adjust x-coordinate of total price text for better alignment
    total_price_x = text_x
    if len(total_price_text) > 0:
        total_price_x -= 120  # Move left
    cv2.putText(img, f"Total Price: {total_price_text} = ${total_price:.2f}", (total_price_x, 120), font, 0.8, (255, 0, 0), 2)

    cv2.imshow('Image', img)

    if cv2.waitKey(1) & 0xff == 27:
        break

cam.release()
cv2.destroyAllWindows()