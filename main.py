import cv2

# Load pre-trained MobileNet SSD model
net = cv2.dnn.readNetFromTensorflow("frozen_inference_graph.pb", "ssd_mobilenet_v2_coco.pbtxt")

# Define fruit classes
classes = ["background", "apple", "banana", "orange", "pear", "strawberry"]

# Load input image
image = cv2.imread("fruit_image.jpg")
(h, w) = image.shape[:2]

# Preprocess the image
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)

# Set the input to the neural network
net.setInput(blob)

# Run forward pass and get detection predictions
detections = net.forward()

# Loop over the detections
for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    
    # Filter out weak detections by confidence threshold
    if confidence > 0.5:
        class_id = int(detections[0, 0, i, 1])
        
        # Get the bounding box coordinates
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        
        # Draw the bounding box and label on the image
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
        label = "{}: {:.2f}%".format(classes[class_id], confidence * 100)
        cv2.putText(image, label, (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Display the output image
cv2.imshow("Fruit Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
