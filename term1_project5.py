from lesson_functions import *
import os
import glob
import time
import numpy as np
from sklearn.preprocessing import StandardScaler
# NOTE: the next import is only valid for scikit-learn version <= 0.17
from sklearn.cross_validation import train_test_split
# for scikit-learn >= 0.18 use:
# from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import matplotlib.image as mpimg
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip

# Images are divided up into vehicles and non-vehicles folders (each of which contains subfolders)
# First locate vehicle images
basedir = 'vehicles/'
# Different folders represent different sources for images e.g. GTI, Kitty, generated by Udacity etc.
image_types = os.listdir(basedir)
car_imgs = []
for imtype in image_types:
    car_imgs.extend(glob.glob(basedir+imtype+'/*'))
print('Number of Vehicle Images found:', len(car_imgs))

# Tweak these parameters and see how the results change.
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9 # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 32 # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
ystart = 400
ystop = 656
scale = 1.5
threshold = 1.5

t=time.time()

car_features = extract_features(car_imgs, color_space=color_space,
				spatial_size=spatial_size, hist_bins=hist_bins,
				orient=orient, pix_per_cell=pix_per_cell,
				cell_per_block=cell_per_block,
				hog_channel=hog_channel, spatial_feat=spatial_feat,
				hist_feat=hist_feat, hog_feat=hog_feat)

# Do the same thing for non-vehicle images
basedir = 'non-vehicles/'
image_types = os.listdir(basedir)
notcar_imgs = []
for imtype in image_types:
    notcar_imgs.extend(glob.glob(basedir+imtype+'/*'))
print('Number of Non-Vehicle Images found:', len(notcar_imgs))

notcar_features = extract_features(notcar_imgs, color_space=color_space,
				   spatial_size=spatial_size, hist_bins=hist_bins,
				   orient=orient, pix_per_cell=pix_per_cell,
				   cell_per_block=cell_per_block,
				   hog_channel=hog_channel, spatial_feat=spatial_feat,
				   hist_feat=hist_feat, hog_feat=hog_feat)

t2 = time.time()
print(round(t2-t, 2), 'Seconds to extract features...')

X = np.vstack((car_features, notcar_features)).astype(np.float64)
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)

print('Using:', orient, 'orientations', pix_per_cell, 'pixels per cell and', cell_per_block, 'cells per block')
print('Feature vector length:', len(X_train[0]))
# Use a linear SVC
svc = LinearSVC()
# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

# Localize cars on test images
images = glob.glob('test_images/*')
for idx, fname in enumerate(images):
    test_img = mpimg.imread(fname)
    # Find_cars on test images
    box_list, find_cars_img = find_cars(test_img, ystart=ystart, ystop=ystop, color_space=color_space,
					scale=scale, svc=svc, X_scaler=X_scaler, orient=orient,
					pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
					spatial_size=spatial_size, hist_bins=hist_bins)
    mpimg.imsave('./output_images/find_cars_' + str(idx) + '.jpg', find_cars_img)
    # Add heat to each box in box list
    heat = np.zeros_like(test_img[:,:,0]).astype(np.float)
    heat = add_heat(heat, box_list)
    # Visualize the heatmap when saving
    heatmap = np.clip(heat, 0, 255)
    mpimg.imsave('./output_images/add_heat_' + str(idx) + '.jpg', heatmap, cmap='hot')
    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, threshold=threshold)
    # Visualize the heatmap when saving
    heatmap = np.clip(heat, 0, 255)
    mpimg.imsave('./output_images/apply_threshold_' + str(idx) + '.jpg', heatmap, cmap='hot')
    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_labeled_bboxes_img = draw_labeled_bboxes(np.copy(test_img), labels)
    mpimg.imsave('./output_images/draw_labeled_bboxes_' + str(idx) + '.jpg', draw_labeled_bboxes_img)

# Process video
# clip = VideoFileClip('./test_video.mp4')
clip = VideoFileClip('./project_video.mp4')
output_clip = clip.fx(process_video, None, ystart=ystart, ystop=ystop, color_space=color_space,
                      scale=scale, svc=svc, X_scaler=X_scaler, orient=orient,
                      pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                      spatial_size=spatial_size, hist_bins=hist_bins, threshold=threshold)
# output_clip.write_videofile('./test_video_output.mp4', audio=False)
output_clip.write_videofile('./project_video_output.mp4', audio=False)