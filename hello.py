import cv2
# print(cv2.getBuildInformation())
video = cv2.VideoCapture('lumin.mp4')

# Initialize frame count
count = 0

# Read the first frame
success, image = video.read()

# Loop through the video file
while success:
    # Save the frame as an image file
    cv2.imwrite("frame%d.jpg" % count, image)     
  
    # Read the next frame
    success, image = video.read()

    # Increment frame count
    count += 1
    
    