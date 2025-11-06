import numpy as np
import cv2
from google.colab.patches import cv2_imshow # Import cv2_imshow

# Read the input image
img = cv2.imread('imgTeste.png') # Replace 'image.jpg' with your image file

# Check if the image was loaded successfully
if img is None:
    print("Error: Could not load image.")
else:
    # Reshape the image to a 2D array of pixels and 3 color channels (RGB/BGR)
    # Each row represents a pixel, and columns represent B, G, R values
    Z = img.reshape((-1, 3))

    # Convert to np.float32 for K-Means algorithm
    Z = np.float32(Z)

    # Define criteria for K-Means termination
    # (type, max_iter, epsilon)
    # cv2.TERM_CRITERIA_EPS: stop if specified accuracy (epsilon) is reached
    # cv2.TERM_CRITERIA_MAX_ITER: stop if specified number of iterations (max_iter) is reached
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    # Number of clusters (K)
    K = 8 # You can adjust this value to control the number of colors

    # Apply K-Means clustering
    # ret: compactness measure (sum of squared distances from each point to its corresponding center)
    # label: array of labels for each pixel, indicating its cluster
    # center: array of cluster centers (the new color values)
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Convert cluster centers back to uint8
    center = np.uint8(center)

    # Map the labels back to the new color values from the centers
    res = center[label.flatten()]

    # Reshape the result back to the original image dimensions
    res2 = res.reshape((img.shape))

    # Display the original and quantized images
    cv2_imshow(img) # Use cv2_imshow
    cv2_imshow(res2) # Use cv2_imshow
    # cv2.waitKey(0) # waitKey() is not needed with cv2_imshow
    # cv2.destroyAllWindows() # destroyAllWindows() is not needed with cv2_imshow