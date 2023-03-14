import cv2
import numpy as np

# Load the image
img = cv2.imread('satelite_image_1.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply thresholding using Otsu's method
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Apply a morphological operation to remove small noise
kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

# Find contours in the binary image
contours, hierarchy = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw the contours on the original image
for cnt in contours:
    cv2.drawContours(img, [cnt], 0, (0,255,0), 2)

# Display the image with the detected buildings
cv2.imshow('Detected Buildings', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
