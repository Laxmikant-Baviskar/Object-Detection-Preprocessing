# import cv2
# import numpy as np

# # Load the images
# img1 = cv2.imread('ap_image1.jpg')
# img2 = cv2.imread('ap_image2.jpg')


# # Compute the mean squared error (MSE)
# mse = np.mean((img1 - img2) ** 2)

# # Compute the structural similarity index (SSIM)
# ssim = cv2.compareSSIM(img1, img2, multichannel=True)

# # Print the results
# print(f"MSE: {mse}")
# print(f"SSIM: {ssim}")


# =======================================



# import cv2
# import numpy as np

# # Load the images
# img1 = cv2.imread('ap_image1.jpg')
# img2 = cv2.imread('ap_image2.jpg')


# # Resize img2 to match the size of img1
# img2_resized = cv2.resize(img2, (570, 574))

# # Compute the mean squared error (MSE)
# mse = np.mean((img1 - img2_resized) ** 2)

# # Print the result
# print(f"MSE: {mse}")


# ===========================================


import cv2
import numpy as np

# Load the images
img1 = cv2.imread('ap_image1.jpg')
img2 = cv2.imread('ap_image2.jpg')
# Resize the images to the same dimensions
img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]))

# Convert the images to grayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Compute the absolute difference between the two grayscale images
diff = cv2.absdiff(gray1, gray2)

# Create a color map to highlight the differences
color_map = np.zeros((diff.shape[0], diff.shape[1], 3), dtype=np.uint8)
color_map[:, :, 0] = np.where(diff > 0, 255, 0)
color_map[:, :, 1] = np.where(diff < 0, 255, 0)

# Show the color map
cv2.imshow('Comparison', color_map)
cv2.waitKey(0)