import numpy as np
import cv2
import scipy
from scipy.ndimage.filters import gaussian_filter

#Initialize x and y displacement array
smoothed_x_trajectory = []
smoothed_y_trajectory = []
smoothed_a1_trajectory = []
smoothed_a2_trajectory = []
x_transformation = []
y_transformation = []
a1_transformation = []
a2_transformation = []
x_trajectory = []
y_trajectory = []
a1_trajectory = []
a2_trajectory = []


fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out_raw = cv2.VideoWriter('output_raw.avi',fourcc, 30.0, (1920,1080))
out_smooth = cv2.VideoWriter('shaky4_smooth.mp4',fourcc, 30.0, (1920,1080))
cap = cv2.VideoCapture('shaky4.mp4')

ret, frame1 = cap.read()
#out_raw.write(frame1)

while(cap.isOpened()):
    ret, frame2_color = cap.read()
    frame2 = frame2_color
    if ret == True:

        # Get video width and height
        width = frame1.shape[1]
        height = frame1.shape[0]

        # Turn to grayscale
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        #cv2.imshow("image1", frame1)
        #cv2.imshow("image2", frame2)

        frame1Corner = cv2.goodFeaturesToTrack(frame1, 200, 0.01, 30)
        #print frame1Corner
        opticalFlowResult = cv2.calcOpticalFlowPyrLK(frame2, frame1, frame1Corner, None)
        #print opticalFlowResult[1]

        transResult = cv2.estimateRigidTransform(frame1Corner, opticalFlowResult[0], False)

        #frame2_trans = cv2.warpAffine(frame2, transResult, (width, height))
        #cv2.imshow("image2", frame2_trans)
        #cv2.waitKey()

        a1_transformation.append(transResult[0][0])
        a2_transformation.append(transResult[0][1])
        x_transformation.append(transResult[0][2])
        y_transformation.append(transResult[1][2])

        #print transResult
        print("One frame written.")
        frame1 = frame2_color

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
print "x_transformation length = ", len(x_transformation)
print "y_transformation length = ", len(y_transformation)

print x_transformation

x_trajectory.append(x_transformation[0])
y_trajectory.append(y_transformation[0])
a1_trajectory.append(a1_transformation[0])
a2_trajectory.append(a2_transformation[0])

#Accumulate the transformations
for i in range(1, len(x_transformation)):
    x_trajectory.append(x_transformation[i] + x_trajectory[len(x_trajectory) - 1])
    y_trajectory.append(y_transformation[i] + y_trajectory[len(y_trajectory) - 1])
    a1_trajectory.append(a1_transformation[i] + a1_trajectory[len(a1_trajectory) - 1])
    a2_trajectory.append(a2_transformation[i] + a2_trajectory[len(a2_trajectory) - 1])
#print x_trajectory

#Smooth the trajectory
radius = 30
for i in range(0, len(x_trajectory)):
    sum_x = 0
    sum_y = 0
    sum_a1 = 0
    sum_a2 = 0
    count = 0

    for j in range(-radius, radius):
        if(i+j >= 0 and i+j < len(x_trajectory)):
            sum_x = sum_x + x_trajectory[i+j]
            sum_y = sum_y + y_trajectory[i+j]
            sum_a1 = sum_a1 + a1_trajectory[i + j]
            sum_a2 = sum_a2 + a2_trajectory[i + j]

            count = count + 1

    smoothed_x_trajectory.append(sum_x / count)
    smoothed_y_trajectory.append(sum_y / count)
    smoothed_a1_trajectory.append(sum_a1 / count)
    smoothed_a2_trajectory.append(sum_a2 / count)

cap.release()

#print x_transformation
#print x_trajectory
#print smoothed_x_trajectory
#print len(a1_trajectory)
#print a1_trajectory.append
print
print len(smoothed_a1_trajectory)
print smoothed_a1_trajectory.append

cap = cv2.VideoCapture('shaky4.mp4')
ret, frame1 = cap.read()
out_smooth.write(frame1)
for frame_num in range(0, len(smoothed_x_trajectory)):
    ret, frame2 = cap.read()
    if ret == True:
        # Translate the image
        #M = np.float32([[1, 0, smoothed_x_trajectory[frame_num] - x_trajectory[frame_num]],[0, 1, smoothed_y_trajectory[frame_num] - y_trajectory[frame_num]]])
        #M = np.float32([[1, 0, x_transformation[frame_num]+smoothed_x_trajectory[frame_num]-x_trajectory[frame_num]], [0, 1, y_transformation[frame_num]+smoothed_y_trajectory[frame_num]-y_trajectory[frame_num]]])
        #M = np.float32([[1, 0, smoothed_x_trajectory[frame_num]],[0, 1, smoothed_y_trajectory[frame_num]]])
        M = np.float32([[(a1_trajectory[frame_num] - smoothed_a1_trajectory[frame_num])*2, (a2_trajectory[frame_num] - smoothed_a2_trajectory[frame_num])*2, x_trajectory[frame_num] - smoothed_x_trajectory[frame_num]],[(-a2_trajectory[frame_num] + smoothed_a2_trajectory[frame_num])*2, (a1_trajectory[frame_num] - smoothed_a1_trajectory[frame_num])*2, y_trajectory[frame_num] - smoothed_y_trajectory[frame_num]]])
        frame2_trans = cv2.warpAffine(frame2, M, (width, height))

        # Draw circles
        '''
        print "start"
        frame2_trans_BW = cv2.cvtColor(frame2_trans, cv2.COLOR_BGR2GRAY)
        corner = cv2.goodFeaturesToTrack(frame2_trans_BW, 200, 0.01, 30)
        if corner != None:
            print corner
            for x in corner:
                cv2.circle(frame2_trans, corner[x], 5, (0, 0, 255), 2)
        '''
        # Write the translated image to output video
        out_smooth.write(frame2_trans)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cv2.waitKey()
# When everything done, release the capture
cap.release()
#out_raw.release()
out_smooth.release()
cv2.destroyAllWindows()