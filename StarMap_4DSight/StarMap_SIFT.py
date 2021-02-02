
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import numpy as np
from skimage.measure import ransac
from skimage.transform import warp, AffineTransform

#Initialize
list_kp1 = []
list_kp2 = []

starMap = cv2.imread("StarMap.png") # Big image
starCluster = cv2.imread("Small_area.png") # Cropped image

starMap_bw = cv2.cvtColor(starMap, cv2.COLOR_BGR2GRAY)
starCluster_bw = cv2.cvtColor(starCluster , cv2.COLOR_BGR2GRAY)


#SIFT
sift = cv2.xfeatures2d.SIFT_create()

keypoints_1, descriptors_1 = sift.detectAndCompute(starMap_bw,None)
keypoints_2, descriptors_2 = sift.detectAndCompute(starCluster_bw,None)


# feature matching
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

matches = bf.match(descriptors_1, descriptors_2)
matches = sorted(matches , key = lambda x:x.distance)

# Finding coordinates of keypoints
for mat in matches:
	#Getting matching keypoints for both images
	starMap_idx = mat.queryIdx
	starCluster_idx = mat.trainIdx

	# x for columns
	# y for rows
	(x1,y1) = keypoints_1[starMap_idx].pt
	(x2,y2) = keypoints_2[starCluster_idx].pt

	list_kp1.append((x1,y1))
	list_kp2.append((x2,y2))

# transform lists to np array for RANSAC
list_kp1 = np.array(list_kp1)
list_kp2 = np.array(list_kp2)


# Finding transform units with RANSAC
model_robust, inliers = ransac((list_kp1, list_kp2), AffineTransform, min_samples=3,
                               residual_threshold=2, max_trials=100)


# # Scale , Translation , Rotation info from RANSAC
# model_robust.translation : Coordinates of the top left corner
# model_robust.scale : scale of the small image to the big
# model_robust.rotation : rotation amount in radyan


# Finding coodinates of Top Left corner with 0 rotation (x0,y0)
x0 = abs((model_robust.translation[0]*math.cos(model_robust.rotation*-1) 
	- model_robust.translation[1]*math.sin(model_robust.rotation*-1)))
y0 = abs((model_robust.translation[1]*math.cos(model_robust.rotation*-1) 
	+ model_robust.translation[0]*math.sin(model_robust.rotation*-1)))

# find dimentions of the small image
clusterH = starCluster.shape[0]
clusterW = starCluster.shape[1]

# Scale the small image dimentions for the big image
scaleW = 1/model_robust.scale[0]
scaleH = 1/model_robust.scale[1]

# Corner coordinates of the smaller image
corTL = [x0 , y0] # Top Left Corner
corTR = [(x0+scaleW*clusterW) , y0] # Top Right Corner
corBL = [x0 , (y0+scaleH*clusterH)] # Bottom Left Corner
corBR = [(x0+scaleW*clusterW) , (y0+scaleH*clusterH)]

coor = [corTL, corTR, corBL, corBR]

cv2.rectangle(starMap , (int(corTL[0]) , int(corTL[1])),
				(int(corBR[0]) , int(corBR[1])),
				(0,255,0),
				thickness= 2,
				lineType= cv2.LINE_8)

## markers
font = cv2.FONT_HERSHEY_SIMPLEX
for item in coor:
	cv2.putText(starMap ,  f"({int(item[0])}, {int(item[1])})", 
		(int(item[0]),int(item[1])-10) ,
	 	font , 0.5 , (0,255,0), 1, cv2.LINE_AA)

cv2.imshow("rectangle" , starMap)
cv2.waitKey(0)


## Uncomment this section to see matched keypoints 
# compare_img = cv2.drawMatches(starMap , keypoints_1 ,starCluster , keypoints_2 , matches[:50] , starCluster , flags=2)
# plt.imshow(compare_img)
# plt.show()


