import numpy as np
import cv2
import scipy
import math
from scipy.ndimage.filters import gaussian_filter

# Key parameters
radius = 35
source_file_name = "shaky4.mp4"

#Initialize x, y, a, s displacement array
smoothed_x_trajectory = []
smoothed_y_trajectory = []
smoothed_a_trajectory = []
smoothed_s_trajectory = []
x_transformation = []
y_transformation = []
a_transformation = []
s_transformation = []
x_trajectory = []
y_trajectory = []
a_trajectory = []
s_trajectory = []

# Parameters
# Shitoma corners params
feature_params = dict( maxCorners = 200,
                       qualityLevel = 0.05,
                       minDistance = 30,
                       blockSize = 10 )
# LK Optical params
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Video Input
cap = cv2.VideoCapture(source_file_name)

# Initial frame
ret, frame1 = cap.read()
frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

# Get video width and height
width = frame1.shape[1]
height = frame1.shape[0]

good_frame2 = np.array([])

while(cap.isOpened()):
    ret, frame2_color = cap.read()
    if not ret:
        print "[ERROR] Cannot read more frames."
        break

    # Turn to grayscale
    frame2 = cv2.cvtColor(frame2_color, cv2.COLOR_BGR2GRAY)

    # Get features to track if not enough
    if len(good_frame2) < 150:
        frame1Corner = cv2.goodFeaturesToTrack(frame1, mask = None, **feature_params)
    else:
        # Reuse previous points
        frame1Corner = good_frame2.reshape(-1,1,2)

    opticalFlowResult, st, err = cv2.calcOpticalFlowPyrLK(frame1, frame2, frame1Corner, None, **lk_params)


    # Use good points only
    good_frame2 = opticalFlowResult[np.where(st == 1)]
    good_frame1 = frame1Corner[np.where(st == 1)]

    transResult = cv2.estimateRigidTransform(good_frame1, good_frame2, False)

    if transResult is not None:
        # Decompose Transformation
        # for [a c e]
        #     [b d f]
        # e and f is displacement
        # atan2(b, a) gives the rotation
        # a = sx * sinA, d = sy * cosA,
        # Assume sx=sy, then sqrt((a^2 + d^2)/2) gives scales
        dx = transResult[0, 2]
        dy = transResult[1, 2]
        da = math.atan2(transResult[1, 0], transResult[0, 0])
        ds = math.sqrt((math.pow(transResult[0, 0], 2) + math.pow(transResult[1, 1], 2)) / 2)

        last_dx = dx
        last_dy = dy
        last_da = da
        last_ds = ds

    else:
        dx = last_dx
        dy = last_dy
        da = last_da
        ds = last_ds

    # Push all transformation DOF into array
    x_transformation.append(dx)
    y_transformation.append(dy)
    a_transformation.append(da)
    s_transformation.append(ds)

    # Print transResult
    print(len(x_transformation), " frames read.")

    # Process to next frame
    frame1 = frame2

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print "x_transformation length = ", len(x_transformation)
print "y_transformation length = ", len(y_transformation)

x_trajectory.append(x_transformation[0])
y_trajectory.append(y_transformation[0])
a_trajectory.append(a_transformation[0])
s_trajectory.append(s_transformation[0])

# Accumulate the transformations
for i in range(1, len(x_transformation)):
    x_trajectory.append(x_transformation[i] + x_trajectory[len(x_trajectory) - 1])
    y_trajectory.append(y_transformation[i] + y_trajectory[len(y_trajectory) - 1])
    a_trajectory.append(a_transformation[i] + a_trajectory[len(a_trajectory) - 1])
    s_trajectory.append(s_transformation[i] * s_trajectory[len(s_trajectory) - 1])

# Smoothen the trajectory
sum_x = 0
sum_y = 0
sum_a = 0
sum_s = 0
count = 0

# Initialize sums
for i in range(0, radius-1):
    if(i < len(x_trajectory)):
        sum_x = sum_x + x_trajectory[i]
        sum_y = sum_y + y_trajectory[i]
        sum_a = sum_a + a_trajectory[i]
        sum_s = sum_s + s_trajectory[i]
        count = count + 1

for i in range(0, len(x_trajectory)):
    if (i + radius < len(x_trajectory)):
        sum_x = sum_x + x_trajectory[i+radius]
        sum_y = sum_y + y_trajectory[i+radius]
        sum_a = sum_a + a_trajectory[i+radius]
        sum_s = sum_s + s_trajectory[i+radius]
        count = count + 1
    
    if (i - radius >= 0):
        sum_x = sum_x - x_trajectory[i-radius]
        sum_y = sum_y - y_trajectory[i-radius]
        sum_a = sum_a - a_trajectory[i-radius]
        sum_s = sum_s - s_trajectory[i-radius]
        count = count - 1

    smoothed_x_trajectory.append(sum_x / count)
    smoothed_y_trajectory.append(sum_y / count)
    smoothed_a_trajectory.append(sum_a / count)
    smoothed_s_trajectory.append(sum_s / count)

# Video Output
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_name = source_file_name.split('.')[0] + '_smooth.mp4'
out_smooth = cv2.VideoWriter(output_name,fourcc, 30.0, (width, height))

# Second pass
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
for frame_num in range(0, len(smoothed_x_trajectory)):
    ret, frame = cap.read()
    if not ret:
        break
    dx = smoothed_x_trajectory[frame_num] - x_trajectory[frame_num]
    dy = smoothed_y_trajectory[frame_num] - y_trajectory[frame_num]
    da = smoothed_a_trajectory[frame_num] - a_trajectory[frame_num]
    ds = smoothed_s_trajectory[frame_num] / s_trajectory[frame_num]

    x = dx + x_transformation[frame_num]
    y = dy + y_transformation[frame_num]
    a = da + a_transformation[frame_num]
    s = ds * s_transformation[frame_num]
    
    cos = math.cos(a)
    sin = math.sin(a)
    M = np.float32([
        [
            s * cos,
            -s * sin,
            x
        ],
        [
            s * sin, 
            s * cos, 
            y
        ]
    ])
    frame_trans = cv2.warpAffine(frame, M, (width, height))

    # Write the translated image to output video
    out_smooth.write(frame_trans)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    print(frame_num , " frames written.")

# When everything done, release the capture
cap.release()
out_smooth.release()
cv2.destroyAllWindows()