# need to install following --> pip install opencv-python-headless numpy matplotlib
# before running the code

# this code gets in image from the web wikipedia page

import cv2  # OpenCV for computer vision
import numpy as np
import matplotlib.pyplot as plt
import requests
from io import BytesIO

url = "https://en.wikipedia.org/wiki/Road#/media/File:Erlangen_Bundesautobahn_73_Auffahrt_Luftbild-20230422-RM-165734.jpg"
response = requests.get(url)
image = np.array(bytearray(response.content), dtype=np.uint8)

#print(response)
#print(image)

if image is None:
    print("Error: Unable to load image.")
else:
    print("Image loaded successfully.")

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for display

# For video (uncomment to use)
# video = cv2.VideoCapture('road_video.mp4')
gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

blur_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

edges = cv2.Canny(blur_image, threshold1=50, threshold2=150)

def region_of_interest(img):
    height, width = img.shape
    mask = np.zeros_like(img)

    # Define a triangular polygon for the mask
    polygon = np.array([[
        (0, height),
        (width, height),
        (width // 2, height // 2)
    ]], np.int32)

    cv2.fillPoly(mask, polygon, 255)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

cropped_edges = region_of_interest(edges)
lines = cv2.HoughLinesP(cropped_edges, rho=1, theta=np.pi/180, threshold=50,
                        minLineLength=50, maxLineGap=150)

def draw_lines(img, lines):
    line_image = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image

line_image = draw_lines(image, lines)
combo_image = cv2.addWeighted(image, 0.8, line_image, 1, 1)

plt.imshow(combo_image)
plt.show()

# Uncomment if using video
# while video.isOpened():
#     ret, frame = video.read()
#     if not ret:
#         break

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     blur = cv2.GaussianBlur(gray, (5, 5), 0)
#     edges = cv2.Canny(blur, 50, 150)
#     cropped_edges = region_of_interest(edges)
#     lines = cv2.HoughLinesP(cropped_edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=150)
#     line_image = draw_lines(frame, lines)
#     combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)

#     cv2.imshow("Lane Detection", combo_image)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# video.release()
# cv2.destroyAllWindows()
