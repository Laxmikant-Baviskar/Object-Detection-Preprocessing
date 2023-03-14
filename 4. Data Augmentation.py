import cv2
import numpy as np

# Load the image
img = cv2.imread('satellite_image.jpg')

# Define the augmentation parameters
rotation_angle = 30
scale_factor = 1.5

# Rotate the image by the specified angle
rotated_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

# Scale the image by the specified factor
scaled_img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

# Display the original and augmented images
cv2.imshow('Original Image', img)
cv2.imshow('Rotated Image', rotated_img)
cv2.imshow('Scaled Image', scaled_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
