# import cv2
# import numpy as np

# # Load the image
# img = cv2.imread('satellite_image.jpg')

# # Convert the image to grayscale
# gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # Apply thresholding to create a binary image
# thresh_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# # Apply morphological operations to remove noise and fill in gaps
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
# closed_img = cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, kernel)

# # Apply connected component analysis to identify objects
# num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed_img)

# # Convert the label image to an 8-bit unsigned integer array
# label_img = np.uint8(labels / num_labels * 255)

# # Display the segmented image
# cv2.imshow('Segmented Image', label_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



# ============================================


# import cv2
# import numpy as np

# # Load the image
# img = cv2.imread('satellite_image.jpg')

# # Convert the image to grayscale
# gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # Apply thresholding to create a binary image
# thresh_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# # Apply morphological operations to remove noise and fill in gaps
# # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
# # closed_img = cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, kernel)

# # Apply connected component analysis to identify objects
# num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh_img) # <-- Change : from closed img to thresh_img 

# # Convert the label image to an 8-bit unsigned integer array
# label_img = np.uint8(labels / num_labels * 255)

# # Display the segmented image
# cv2.imshow('Segmented Image', label_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



# ============ With Color Extraction  =====================

import cv2
import numpy as np

# Load the image
# img = cv2.imread("roof_image.jpg")
img = cv2.imread("satelite_image_1.jpg")

# Convert the image to HSV color space
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Define color ranges for the building
lower_color = np.array([0, 0, 50])
upper_color = np.array([255, 255, 255])

# Threshold the image to extract the building
mask = cv2.inRange(hsv_img, lower_color, upper_color)

# Apply morphology operations to remove noise and fill holes
kernel = np.ones((5, 5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# Apply the mask to the original image to extract the building
building = cv2.bitwise_and(img, img, mask=mask)

# Display the result
cv2.imshow("Building", building)
cv2.waitKey(0)
cv2.destroyAllWindows()
