import requests
from bs4 import BeautifulSoup
import urllib.request
import os
from datetime import datetime, timedelta

# Step 1: Fetch the webpage content
url = 'https://onemotoring.lta.gov.sg/content/onemotoring/home/driving/traffic_information/traffic-cameras/woodlands.html#trafficCameras'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
}
response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.text, 'html.parser')

# Step 2: Define current time and the 30-minute window
current_time = datetime.now()
time_threshold = current_time - timedelta(minutes=30)

# Step 3: Create folder to save images
if not os.path.exists('tuas_images'):
    os.makedirs('tuas_images')

# Step 4: Locate and filter images from Tuas camera within the last 30 minutes
traffic_cards = soup.find_all('div', class_='card')

for card in traffic_cards:
    # Find the description (e.g., Tuas)
    desc = card.find('div', class_='trf-desc').get_text(strip=True)

    if 'Tuas' in desc:
        # Get timestamp and remove time zone (SST)
        timestamp_str = card.find('span', class_='left').get_text(strip=True)

        # Remove the time zone part
        timestamp_str_no_tz = ' '.join(timestamp_str.split()[:-2])

        # Append the current year to the timestamp string
        timestamp_str_with_year = f"{timestamp_str_no_tz} {current_time.year}"

        # Parse the remaining timestamp
        timestamp_format = "%a %b %d %H:%M:%S %Y"  # Adjusted format with the year
        timestamp = datetime.strptime(timestamp_str_with_year, timestamp_format)

        # Check if the timestamp is within the last 30 minutes
        if timestamp >= time_threshold:
            # Get the image URL and complete it
            img_tag = card.find('img')
            img_url = img_tag.get('src')
            img_full_url = 'https:' + img_url

            # Download the image with proper headers
            req = urllib.request.Request(img_full_url, headers=headers)
            img_name = f"tuas_images/tuas_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"

            with urllib.request.urlopen(req) as response, open(img_name, 'wb') as out_file:
                out_file.write(response.read())

            print(f'Downloaded {img_name}')

from IPython.display import Image, display

# Assuming you have downloaded the images into the 'tuas_images' folder

# List all files in the tuas_images folder
import os
for filename in os.listdir('tuas_images'):
  if filename.endswith(".jpg"):  # Assuming your images are JPGs
    image_path = os.path.join('tuas_images', filename)
    display(Image(filename=image_path))

pip install tensorflow keras opencv-python matplotlib

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()

# Fix for output layers retrieval to avoid IndexError
try:
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
except:
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Load COCO classes (which includes vehicles like car, truck, bus)
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Colors for different classes
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Vehicle classes as per COCO (cars, buses, trucks)
vehicle_classes = ["car", "bus", "truck", "motorbike", "bicycle"]

# Directory with images
image_folder = 'tuas_images'

# Loop through all images in the folder
for image_file in os.listdir(image_folder):
    if image_file.endswith('.jpg'):
        img_path = os.path.join(image_folder, image_file)

        # Load the image
        img = cv2.imread(img_path)
        img = cv2.resize(img, None, fx=0.4, fy=0.4)
        height, width, channels = img.shape

        # Prepare the image for the YOLO model
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        # Initialize lists to hold detected bounding boxes, confidences, and class IDs
        class_ids = []
        confidences = []
        boxes = []

        # Process the output from YOLO
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                # Filter out weak predictions (confidence threshold)
                if confidence > 0.3:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Non-maximum suppression to remove overlapping boxes
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.6)

        # Count vehicles
        vehicle_count = 0

        # Draw bounding boxes and count vehicles
        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = confidences[i]

                # Count if the detected object is a vehicle
                if label in vehicle_classes:
                    vehicle_count += 1
                    color = colors[class_ids[i]]
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(img, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Display the count of vehicles
        print(f"Image: {image_file}, Vehicle count: {vehicle_count}")

        # Display the image with detected vehicles
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title(f"Vehicle Count: {vehicle_count}")
        plt.show()

        # Save the processed image with detected vehicles
        output_path = os.path.join('output_images', f"detected_{image_file}")
        cv2.imwrite(output_path, img)
