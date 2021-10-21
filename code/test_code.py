import json
import glob
import cv2
import time

a = glob.glob('./dataset/info/*.json')
b = glob.glob('./dataset/videos/*.mov')
print(a[:10])
print()

index = 7

with open(a[index]) as json_file:  
    data = json.load(json_file)
    print(data['gyro'][0:10])
    print()
    print(len(data['gyro']))
    print()
    print(data['accelerometer'][0:10])
    print()
    print(len(data['accelerometer']))

cap = cv2.VideoCapture(b[index])

n = 0
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
 
    # Display the resulting frame
    cv2.imshow('Frame',frame)
    print(data['gyro'][n])
    print(data['accelerometer'][n])
    print()
    time.sleep(2)
    
 
    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
 
  # Break the loop
  else: 
    break

  n += 1
# When everything done, release the video capture object
cap.release()
 
# Closes all the frames
cv2.destroyAllWindows()
