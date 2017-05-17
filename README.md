##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/find_cars_2.jpg
[image2]: ./output_images/find_cars_3.jpg
[image3]: ./output_images/find_cars_4.jpg
[image4]: ./output_images/add_heat_4.jpg
[image5]: ./output_images/apply_threshold_4.jpg
[image6]: ./output_images/draw_labeled_bboxes_4.jpg
[video1]: ./project_video_output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in lines 16 through 67 of the file called `term1_project5.py`.  
I started by setting up two lists (`car_imgs` and `notcar_imgs`) of file names of all `vehicle` (lines 18 through 24) and `non-vehicle` (lines 52 through 57) images.  
I  applied the `extract_features()` function to extract `car_features` and `notcar_features` from these images (lines 44 and 59).  
You can find this function in the file called `lesson_functions.py` (line 67). In this file I collected all the useful code from the lessons for this project.  
In the `extract_features()` function I iterated through the list of image names and read in each image one by one (line 77) and I applied `color_space` conversion to it (line 79).  
If `hog_feat == True` I extracted the HOG features from each color converted image with the `get_hog_features()` function. When the color channel `hog_channel` parameter value is `'ALL'` (line 90), I call this `get_hog_features()` function successively for all the color channels (0, 1 and 2).  
The `get_hog_features()` function (line 11) applies the `skimage.feature.hog()` function (lines 14 and 22) to return the HOG features.

####2. Explain how you settled on your final choice of HOG parameters.

In line 26 through 31 of `term1_project5.py` I played with this `color_space`, `hog_channel` and  the `skimage.feature.hog()` parameters (`orientations`, `pixels_per_cell` and `cells_per_block`) to get a feel for what combination of parameters give the best accuracy of the linear SVC (line 92).

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a Linear Support Vector Classification (LinearSVC) classifier in lines 69 through 92 of `term1_project5.py` using the extracted binned color, color histogram and HOG features.  
To combine and normalize the `car_features` and `notcar_features` with the `StandardScaler()` method I created a vertical array stack `X` (line 69) where each row is a single feature vector. `StandardScaler()` expects np.float64. Then I fit a scaler to `X` and scaled it (lines 71 and 73). Now, `scaled_X` contains the normalized feature vectors.   
I defined the labels vector `y` (line 76) by horizontally (column wise) stacking two one-dimensional arrays. One  array of `car_features` length filled with ones and the other of `notcar_features` length filled with zeros.  
In line 80 I split up the features (`scaled_X`) and labels (`y`) into randomized training (`X_train` and `y_train`) and (`X_test` and `y_test`) test sets.  
In line 85 I created the LinearSVC and in line 88 I trained it on the training set.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

In lines 195 through 225 (`lesson_functions.py`) of the `find_cars()` function I implemented a sliding window search.
Each window is defined by a scaling factor where a scale of 1 would result in a window that's 8 x 8 cells then the overlap of each window is in terms of the cell distance. This means that a cells_per_step = 2 would result in a search window overlap of 75% (1 - 2/8).

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

![alt text][image1]

![alt text][image2]

![alt text][image3]

To optimize the performance of my classifier I extracted HOG features just once for the entire region of interest (i.e. lower half of each frame of video) (lines 189 through 191 of `lesson_functions.py`) and subsampled that array to get features for each sliding window (lines 201 through 203), instead of extracting the HOG features from each individual window as you searched across the image.

---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_output.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video by adding the bounding box coordinates to a list `box_list` (line 266 of `lesson_functions.py`: `find_cars()`).  From the positive detections I created a heatmap (line 272: `add_heat()`) and then thresholded that map to filter for false positives (line 274: `apply_threshold()`).  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap (line 276).  I then assumed each blob corresponded to a vehicle.  I combined overlapping bounding boxes to cover the area of each blob detected (line 277: `draw_labeled_bboxes()`).  

Here are the results of successively applying the `find_cars()`, `add_heat()`, `apply_threshold()` and `draw_labeled_bboxes()` functions to the same frame of the video:  

Output of the `find_cars()` function:
![alt text][image3]

Output of the `add_heat()` function:
![alt text][image4]

Output of the `apply_threshold()` function:
![alt text][image5]

Output of the `draw_labeled_bboxes()` function:
![alt text][image6]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

- I hard-coded limited the sliding window search in the y direction for each frame of the video by setting `ystart = 400` and `ystop = 656` (lines 37 and 38 of `term1_project5.py`). These values could be different for other videos.
- For this project we used the so called traditional computer vision techniques, I would also like to tackle this problem with the deep learning for computer vision techniques.

