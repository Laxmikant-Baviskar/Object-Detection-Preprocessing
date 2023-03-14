import cv2

# Load the image
img = cv2.imread('satellite_image.jpg')

# Convert the image to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply histogram equalization to enhance the contrast
eq_img = cv2.equalizeHist(gray_img)

# Display the original and enhanced images
cv2.imshow('Original Image', img)
cv2.imshow('Enhanced Image', eq_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
