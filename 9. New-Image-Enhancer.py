# import cv2

# # Load the image
# image = cv2.imread('anime-blur.jpg')

# # Convert the image to LAB color space
# lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

# # Apply contrast enhancement to the L channel
# clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
# lab[:,:,0] = clahe.apply(lab[:,:,0])

# # Apply color correction to the A and B channels
# lab[:,:,1] = cv2.equalizeHist(lab[:,:,1])
# lab[:,:,2] = cv2.equalizeHist(lab[:,:,2])

# # Convert the LAB image back to BGR color space
# result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

# # Save the result
# cv2.imwrite('result.jpg', result)


# =============================================================


# import cv2
# import numpy as np

# # Load the blurred image
# blur = cv2.imread('anime-blur.jpg')

# # Convert the image to grayscale
# gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

# # Compute the Fourier transform of the blurred image
# f = np.fft.fft2(gray)

# # Estimate the power spectrum of the blur kernel
# psf = np.ones_like(gray)
# psf[psf.shape[0]//2, psf.shape[1]//2] = 0
# psf = cv2.GaussianBlur(psf, (5, 5), 0)
# psf /= psf.sum()

# # Compute the Wiener filter
# snr = 0.1
# wiener_filter = np.conj(psf) / (np.abs(psf)**2 + snr)

# # Apply the Wiener filter to the Fourier transform of the blurred image
# f_deconvolved = f * wiener_filter

# # Compute the inverse Fourier transform of the filtered image
# deconvolved = np.fft.ifft2(f_deconvolved).real

# # Normalize the deblurred image to the range [0, 255]
# deconvolved = cv2.normalize(deconvolved, None, 0, 255, cv2.NORM_MINMAX)

# # Convert the deblurred image to uint8 datatype
# deconvolved = deconvolved.astype(np.uint8)

# # Save the result
# cv2.imwrite('deblurred_image.jpg', deconvolved)


# ==========================================================

# import cv2
# import numpy as np

# # Load the blurred image
# blur = cv2.imread('anime-blur.jpg')

# # Convert the image to grayscale
# gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

# # Compute the Fourier transform of the blurred image
# f = np.fft.fft2(gray)

# # Estimate the power spectrum of the blur kernel
# psf = np.ones_like(gray)
# psf[psf.shape[0]//2, psf.shape[1]//2] = 0
# psf = cv2.GaussianBlur(psf, (5, 5), 0)
# psf = psf.astype(np.float64)
# psf /= psf.sum()

# # Compute the Wiener filter
# snr = 0.1
# wiener_filter = np.conj(psf) / (np.abs(psf)**2 + snr)

# # Apply the Wiener filter to the Fourier transform of the blurred image
# f_deconvolved = f * wiener_filter

# # Compute the inverse Fourier transform of the filtered image
# deconvolved = np.fft.ifft2(f_deconvolved).real

# # Normalize the deblurred image to the range [0, 255]
# deconvolved = cv2.normalize(deconvolved, None, 0, 255, cv2.NORM_MINMAX)

# # Convert the deblurred image to uint8 datatype
# deconvolved = deconvolved.astype(np.uint8)

# # Save the result
# cv2.imwrite('deblurred_image.jpg', deconvolved)


# =============================================================


# import cv2
# import numpy as np
# from skimage import restoration

# # Load the blurred image
# blur = cv2.imread('anime-blur.jpg')

# # Convert the image to grayscale
# gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

# # Compute the Richardson-Lucy deconvolution
# psf = np.ones((5, 5)) / 25
# deconvolved = restoration.richardson_lucy(gray, psf, iterations=30)

# # Normalize the deblurred image to the range [0, 255]
# deconvolved = cv2.normalize(deconvolved, None, 0, 255, cv2.NORM_MINMAX)

# # Convert the deblurred image to uint8 datatype
# deconvolved = deconvolved.astype(np.uint8)

# # Save the result
# cv2.imwrite('new_deblurred_image.jpg', deconvolved)

# =============================================================

# import cv2
# import numpy as np
# from skimage import restoration

# # Load the blurred image
# blur = cv2.imread('anime-blur.jpg')

# # Convert the image to grayscale
# gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

# # Compute the Richardson-Lucy deconvolution
# psf = np.ones((5, 5)) / 25
# deconvolved = restoration.richardson_lucy(gray, psf, num_iter=30)

# # Normalize the deblurred image to the range [0, 255]
# deconvolved = cv2.normalize(deconvolved, None, 0, 255, cv2.NORM_MINMAX)


# # Convert the deblurred image to uint8 datatype
# deconvolved = deconvolved.astype(np.uint8)

# # Save the result
# cv2.imwrite('new_deblurred_image.jpg', deconvolved)


# =============================================================
import cv2
import numpy as np
from skimage import restoration

# Load the blurred image
blur = cv2.imread('anime-blur.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

# Apply Wiener deconvolution
psf = np.ones((5, 5)) / 25
deconvolved = restoration.wiener(gray, psf, 0.1)

# Normalize the deblurred image to the range [0, 255]
deconvolved = cv2.normalize(deconvolved, None, 0, 255, cv2.NORM_MINMAX)

# Convert the deblurred image to uint8 datatype
deconvolved = deconvolved.astype(np.uint8)

# Save the result
cv2.imwrite('new1_deblurred_image.jpg', deconvolved)
