# ===> Gives Poor result 

import cv2

# Load the image
img = cv2.imread('satellite_image.jpg')

# Apply a median filter with a 5x5 kernel
filtered_img = cv2.medianBlur(img, 5)



# Display the original and filtered images
cv2.imshow('Original Image', img)
cv2.imshow('Filtered Image', filtered_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
