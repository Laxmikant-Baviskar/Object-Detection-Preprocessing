# import cv2
# import numpy as np

# # Load the two images
# img1 = cv2.imread('ap_image1.jpg')
# img2 = cv2.imread('ap_image2.jpg')

# # Convert the images to grayscale
# gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# # Compute the absolute difference between the two grayscale images
# diff = cv2.absdiff(gray1, gray2)

# # Apply a threshold to the difference image to remove small differences
# threshold = 50
# mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]

# # Convert the mask to a color heatmap
# heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_HOT)

# # Display the result
# cv2.imshow('Difference Heatmap', heatmap)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# ===============================================================

# import cv2
# import numpy as np

# # Load the two images
# img1 = cv2.imread('ap_image1.jpg')
# img2 = cv2.imread('ap_image2.jpg')

# # Resize the images to a common size
# img1 = cv2.resize(img1, (640, 480))
# img2 = cv2.resize(img2, (640, 480))

# # Convert the images to grayscale
# gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# # Compute the absolute difference between the two grayscale images
# diff = cv2.absdiff(gray1, gray2)

# # Apply a threshold to the difference image to remove small differences
# threshold = 50
# mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]

# # Convert the mask to a color heatmap
# heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_HOT)

# # Display the result
# cv2.imshow('Difference Heatmap', heatmap)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# =============================================================

# import cv2
# import numpy as np

# # Load the two images
# img1 = cv2.imread('ap_image1.jpg')
# img2 = cv2.imread('ap_image2.jpg')

# # Resize the images to a common size
# img1 = cv2.resize(img1, (640, 480))
# img2 = cv2.resize(img2, (640, 480))

# # Convert the images to grayscale
# gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# # Compute the difference between the two grayscale images
# diff = gray2 - gray1

# # Create a custom color map for the heatmap
# color_map = np.zeros((256, 1, 3), dtype=np.uint8)
# color_map[:, :, 0] = np.where(diff < 0, 255, 0)
# color_map[:, :, 1] = np.where(diff > 0, 255, 0)
# color_map[:, :, 2] = np.abs(diff)

# # Apply the custom color map to the difference image
# heatmap = cv2.applyColorMap(diff, color_map)

# # Display the result
# cv2.imshow('Difference Heatmap', heatmap)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# ================================================================

# import cv2
# import numpy as np

# # Load the two images
# img1 = cv2.imread('ap_image1.jpg')
# img2 = cv2.imread('ap_image2.jpg')

# # Resize the images to a common size
# img1 = cv2.resize(img1, (640, 480))
# img2 = cv2.resize(img2, (640, 480))

# # Convert the images to grayscale
# gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# # Compute the difference between the two grayscale images
# diff = gray2 - gray1

# # Scale the difference to the range (0, 255)
# diff = (diff - np.min(diff)) / (np.max(diff) - np.min(diff)) * 255

# # Create a custom color map for the heatmap
# color_map = np.zeros((256, 1, 3), dtype=np.uint8)
# color_map[:, :, 0] = np.where(diff < 0, 255, 0)
# color_map[:, :, 1] = np.where(diff > 0, 255, 0)
# color_map[:, :, 2] = np.abs(diff)

# # ==============

# # color_map = np.zeros((256, 1, 3), dtype=np.uint8)
# # color_map[:, :, 2] = np.where(diff > 0, diff, 0)
# # color_map[:, :, 0] = np.where(diff < 0, -diff, 0)

# # ==============


# # Apply the custom color map to the difference image
# heatmap = cv2.applyColorMap(diff.astype(np.uint8), color_map)

# # Display the result
# cv2.imshow('Difference Heatmap', heatmap)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



# ==========================

import cv2
import numpy as np

# Load the images
img1 = cv2.imread('ap_image1.jpg')
img2 = cv2.imread('ap_image2.jpg')

# Compute the difference between the images
diff = cv2.absdiff(img1, img2)
gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

# Resize the difference array to match the target shape
gray_resized = cv2.resize(gray, (256, 256))

# Create a color map of the difference
color_map = np.zeros((256, 256, 3), dtype=np.uint8)
color_map[:, :, 2] = np.where(gray_resized > 0, 255, 0)
color_map[:, :, 0] = np.where(gray_resized < 0, 255, 0)

# Display the result
cv2.imshow('Color map', color_map)
cv2.waitKey(0)
cv2.destroyAllWindows()
