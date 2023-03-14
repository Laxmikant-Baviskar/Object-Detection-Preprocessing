import cv2

# Load the image
img = cv2.imread('satellite_image.jpg')

# Define the object bounding box
x1, y1, x2, y2 = 100, 100, 200, 200

# Draw the bounding box on the image
annotated_img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Display the annotated image
cv2.imshow('Annotated Image', annotated_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
