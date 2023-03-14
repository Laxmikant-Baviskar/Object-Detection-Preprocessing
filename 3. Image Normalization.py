import cv2

# Load the image
img = cv2.imread('satellite_image.jpg')

# Normalize the pixel values to the range [0, 1]
normalized_img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

# Display the original and normalized images
cv2.imshow('Original Image', img)
cv2.imshow('Normalized Image', normalized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
