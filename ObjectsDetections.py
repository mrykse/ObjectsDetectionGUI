import cv2
import argparse
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import imutils

# Constructing Argument Parse to input from Command Line
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True, help='Path to prototxt')
ap.add_argument("-m", "--model", required=True, help='Path to model weights')
ap.add_argument("-c", "--confidence", type=float, default=0.7)
args = vars(ap.parse_args())

# Initialize Objects and corresponding colors which the model can detect
labels = ["background", "aeroplane", "bicycle", "bird",
          "boat", "bottle", "bus", "car", "cat", "chair", "cow",
          "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",
          "sheep", "sofa", "train", "tvmonitor"]
colors = np.random.uniform(0, 255, size=(len(labels), 3))

# Loading Caffe Model
print('[Status] Loading Model...')

nn = cv2.dnn.readNetFromCaffe(args['prototxt'], args['model'])

# Initialize Tkinter window
root = tk.Tk()
root.title("Object Detection")


# Function to detect and draw on image
def detect_and_draw(frame):
    (h, w) = frame.shape[:2]

    # Converting Frame to Blob
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 0.007843, (300, 300), 127.5)

    # Passing Blob through network to detect and predict
    nn.setInput(blob)
    detections = nn.forward()

    # Loop over the detections
    for i in np.arange(0, detections.shape[2]):

        # Extracting the confidence of predictions
        confidence = detections[0, 0, i, 2]

        # Filtering out weak predictions
        if confidence > args["confidence"]:
            # Extracting the index of the labels from the detection
            # Computing the (x,y) - coordinates of the bounding box
            idx = int(detections[0, 0, i, 1])

            # Extracting bounding box coordinates
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Drawing the prediction and bounding box
            label = "{}: {:.2f}%".format(labels[idx], confidence * 100)
            cv2.rectangle(frame, (startX, startY), (endX, endY), colors[idx], 2)

            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[idx], 2)

    return frame


# Function to open image and display detections
def open_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        # Read image
        original_image = cv2.imread(file_path)
        original_image = imutils.resize(original_image, width=500)

        # Convert images to RGB format for tkinter
        original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        original_image_rgb = Image.fromarray(original_image_rgb)

        image_with_detections = detect_and_draw(original_image)
        image_with_detections_rgb = cv2.cvtColor(image_with_detections, cv2.COLOR_BGR2RGB)
        image_with_detections_rgb = Image.fromarray(image_with_detections_rgb)

        # Convert images to Tkinter format
        original_image_tk = ImageTk.PhotoImage(original_image_rgb)
        detections_image_tk = ImageTk.PhotoImage(image_with_detections_rgb)

        # Display original image on left side
        original_image_label = tk.Label(root, image=original_image_tk)
        original_image_label.grid(row=0, column=0)

        # Display image with detections on right side
        detections_image_label = tk.Label(root, image=detections_image_tk)
        detections_image_label.grid(row=0, column=1)

        # Keep references to Tkinter image objects
        original_image_label.image = original_image_tk
        detections_image_label.image = detections_image_tk


# Button to open image
open_button = tk.Button(root, text="Open Image", command=open_image)
open_button.grid(row=1, column=0)

root.mainloop()


# # SAME CODE BUT WITH VIDEO DETECTION
# import cv2
# import time
# import imutils
# import argparse
# import numpy as np
#
# from imutils.video import FPS
# from imutils.video import VideoStream
#
#
# # Constructing Argument Parse to input from Command Line
# ap = argparse.ArgumentParser()
# ap.add_argument("-p", "--prototxt", required=True, help='Path to prototxt')
# ap.add_argument("-m", "--model", required=True, help='Path to model weights')
# ap.add_argument("-c", "--confidence", type=float, default=0.7)
# args = vars(ap.parse_args())
#
# # Initialize Objects and corresponding colors which the model can detect
# labels = ["background", "aeroplane", "bicycle", "bird",
#           "boat", "bottle", "bus", "car", "cat", "chair", "cow",
#           "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",
#           "sheep", "sofa", "train", "tvmonitor"]
# colors = np.random.uniform(0, 255, size=(len(labels), 3))
#
# # Loading Caffe Model
# print('[Status] Loading Model...')
#
# nn = cv2.dnn.readNetFromCaffe(args['prototxt'], args['model'])
# # Initialize Video Stream
# print('[Status] Starting Video Stream...')
# vs = VideoStream(src=0).start()
# time.sleep(2.0)
# fps = FPS().start()
#
# # Loop Video Stream
# while True:
#
#     # Resize Frame to 400 pixels
#     frame = vs.read()
#     frame = imutils.resize(frame, width=640)
#     (h, w) = frame.shape[:2]
#
#     # Converting Frame to Blob
#     blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
#                                  0.007843, (300, 300), 127.5)
#
#     # Passing Blob through network to detect and predict
#     nn.setInput(blob)
#     detections = nn.forward()
#
#     # Loop over the detections
#     for i in np.arange(0, detections.shape[2]):
#
#         # Extracting the confidence of predictions
#         confidence = detections[0, 0, i, 2]
#
#         # Filtering out weak predictions
#         if confidence > args["confidence"]:
#             # Extracting the index of the labels from the detection
#             # Computing the (x,y) - coordinates of the bounding box
#             idx = int(detections[0, 0, i, 1])
#
#             # Extracting bounding box coordinates
#             box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#             (startX, startY, endX, endY) = box.astype("int")
#
#             # Drawing the prediction and bounding box
#             label = "{}: {:.2f}%".format(labels[idx], confidence * 100)
#             cv2.rectangle(frame, (startX, startY), (endX, endY), colors[idx], 2)
#
#             y = startY - 15 if startY - 15 > 15 else startY + 15
#             cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[idx], 2)
#
#     cv2.imshow("Frame", frame)
#     key = cv2.waitKey(1) & 0xFF
#
#     if key == ord('q'):
#         break
#
#     fps.update()
#
# fps.stop()
#
# print("[Info] Elapsed time: {:.2f}".format(fps.elapsed()))
# print("[Info] Approximate FPS:  {:.2f}".format(fps.fps()))
#
# cv2.destroyAllWindows()
# vs.stop()
#
#
