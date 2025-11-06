import cv2
import numpy as np
from google.colab.patches import cv2_imshow

# Load the image
img = cv2.imread('imgCarro_Km.png')

# Check if the image was loaded successfully
if img is None:
    print("Error: Could not load image.")
else:
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to smooth the image and reduce noise
    # Kernel size (5,5) and standard deviation 0 are common choices
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Perform Canny edge detection
    # The two threshold values are crucial for determining which edges are detected
    # Experiment with these values (e.g., 50, 150 or 100, 200) for different images
    edges = cv2.Canny(blurred, 50, 150)

    # Display the original and edge-detected images
    cv2_imshow(img)
    cv2_imshow(edges)

    # Wait for a key press and then close all windows
    # cv2.waitKey(0) # waitKey() is not needed with cv2_imshow
    # cv2.destroyAllWindows() # destroyAllWindows() is not needed with cv2_imshow