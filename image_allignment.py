# import cv2

# import numpy as np


# # Load the two images
# img1 = cv2.imread('image1.jpg')
# img2 = cv2.imread('image2.jpg')

# # Convert the images to grayscale
# gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# # Initialize the ORB feature detector
# orb = cv2.ORB_create()

# # Find the keypoints and descriptors in the two images
# kp1, des1 = orb.detectAndCompute(gray1, None)
# kp2, des2 = orb.detectAndCompute(gray2, None)

# # Initialize the brute-force matcher
# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# # Match the descriptors
# matches = bf.match(des1, des2)

# # Sort the matches by distance
# matches = sorted(matches, key=lambda x:x.distance)

# # Get the best matches
# best_matches = matches[:50]

# # Get the keypoints from the best matches
# src_pts = np.float32([kp1[m.queryIdx].pt for m in best_matches]).reshape(-1, 1, 2)
# dst_pts = np.float32([kp2[m.trainIdx].pt for m in best_matches]).reshape(-1, 1, 2)

# # Find the transformation matrix using RANSAC
# M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# # Warp the second image to align with the first image
# aligned_img = cv2.warpPerspective(img2, M, (img1.shape[1], img1.shape[0]))

# # Display the aligned image
# cv2.imshow('Aligned Image', aligned_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# ==================================================================

# import cv2
# import numpy as np

# # Load the two images
# img1 = cv2.imread('image1.jpg')
# img2 = cv2.imread('image2.jpg')

# # Convert the images to grayscale
# gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# # Initialize the SIFT feature detector and FLANN matcher
# sift = cv2.SIFT_create()
# flann = cv2.FlannBasedMatcher({'algorithm': 0, 'trees': 5}, {'checks': 50})

# # Find the keypoints and descriptors in the two images
# kp1, des1 = sift.detectAndCompute(gray1, None)
# kp2, des2 = sift.detectAndCompute(gray2, None)

# # Match the descriptors using FLANN matcher
# matches = flann.knnMatch(des1, des2, k=2)

# # Filter the matches using Lowe's ratio test
# good_matches = []
# for m, n in matches:
#     if m.distance < 0.75 * n.distance:
#         good_matches.append(m)

# # Get the keypoints from the good matches
# src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
# dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# # Find the transformation matrix using RANSAC
# M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# # Warp the second image to align with the first image
# aligned_img = cv2.warpPerspective(img2, M, (img1.shape[1], img1.shape[0]), flags=cv2.INTER_LINEAR)

# # Display the aligned image
# cv2.imshow('Aligned Image', aligned_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# ================================================================

import cv2
import numpy as np

# Load the two images
img1 = cv2.imread('NL_image1.jpg')
img2 = cv2.imread('NL_image2.jpg')

# Convert the images to grayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Initialize the SIFT feature detector
sift = cv2.SIFT_create()

# Find the keypoints and descriptors in the two images
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

# Initialize the brute-force matcher
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# Match the descriptors using brute-force matching
matches = bf.match(des1, des2)

# Sort the matches by distance
matches = sorted(matches, key=lambda x: x.distance)

# Keep only the top matches (adjust this parameter as needed)
num_matches = 100
matches = matches[:num_matches]

# Extract the keypoints from the matches
pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

# Compute the homography matrix using RANSAC
H, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC)

# Apply the homography matrix to the second image
aligned_img = cv2.warpPerspective(img2, H, (img1.shape[1], img1.shape[0]))

# Display the result
cv2.imshow('Aligned Image', aligned_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
