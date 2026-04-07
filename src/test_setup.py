import cv2
import numpy as np

print("OpenCV version:", cv2.__version__)
print("NumPy version:", np.__version__)

# Create a simple black image
img = np.zeros((300, 300, 3), dtype=np.uint8)

cv2.putText(img, "Setup OK", (50,150),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

cv2.imshow("Test Window", img)
cv2.waitKey(2000)
cv2.destroyAllWindows()
