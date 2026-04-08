import cv2

#Camera → Frame → Face Detection → Eye Detection → AI Model

#VideoCapture() is an OpenCV class used to:
#Connect to a camera
#Capture video frames

# Open webcam (0 = default camera)
# Other indices can be used for external cameras, for example 1 = external USB camera
# "video.mp4" -> video file instead of webcam

#cap is an object (handle) that represents your camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot access camera")
    exit()

while True:
    ret, frame = cap.read()

    #ret: Boolean True/False 
    # True - Frame captured successfully
    # False - Failed to capture frame (e.g., Permission issues(OS), camera busy, hardware failure/glitch)

    #frame: The actual image captured from the camera, represented as a NumPy array (matrix of pixel values)
    #numpy.ndarray : (height, width, channels) : (1080, 1920, 3) : Color image (BGR)
    #   print("Frame size:", frame.shape)

    if not ret:
        print("Error: Failed to capture frame")
        break

    # Display the frame
    #cv2.imshow("Window Name", frame)
    cv2.imshow("Driver Monitoring - Camera Feed", frame)


    # Press 'q' to exit
    #cv2.waitKey(delay) is a function that waits for a key event for a specified amount of time (in milliseconds).
    # If a key is pressed during that time, it returns the ASCII code of the key. If no key is pressed, it returns -1.
    # cv2.waitKey(1) : Waits for 1ms and then continues loop. This allows the video feed to update smoothly while still allowing for user input.
    
    # Without waitKey() : Window may freeze, Image may not display and No keyboard interaction possible (e.g., to quit the program)
    # Pressing 'q' will break the loop and exit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()