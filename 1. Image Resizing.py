import cv2

# Load the image
img = cv2.imread('satellite_image.jpg')


# Resize the image to half its original size
resized_img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

# Display the original and resized images
cv2.imshow('Original Image', img)
cv2.imshow('Resized Image', resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

