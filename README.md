# P5 Vehicle Detection and Tracking -- Reinhard Steffens

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)


[hog_01]: ./pics/output_6_0.png "HOG Features 01"
[hog_02]: ./pics/output_6_1.png "HOG Features 02"
[hog_03]: ./pics/output_6_2.png "HOG Features 03"
[hog_04]: ./pics/output_6_3.png "HOG Features 04"

[hog_YCrCb_Veh_01]: ./pics/output_7_0.png "HOG Features YCrCb Veh 01"
[hog_YCrCb_Veh_02]: ./pics/output_7_1.png "HOG Features YCrCb Veh 02"

[hog_YCrCb_No_01]: ./pics/output_8_0.png "HOG Features YCrCb No Veh 01"
[hog_YCrCb_No_02]: ./pics/output_8_1.png "HOG Features YCrCb No Veh 02"

[hog_HLS_Veh_01]: ./pics/output_9_0.png "HOG Features HLS Veh 01"
[hog_HLS_Veh_02]: ./pics/output_9_1.png "HOG Features HLS Veh 02"

[hog_HLS_No_01]: ./pics/output_10_0.png "HOG Features HLS No Veh 01"
[hog_HLS_No_02]: ./pics/output_10_1.png "HOG Features HLS No Veh 02"

[hog_HSV_Veh_01]: ./pics/output_11_0.png "HOG Features HSV Veh 01"
[hog_HSV_Veh_02]: ./pics/output_11_1.png "HOG Features HSV Veh 02"

[hog_HSV_No_01]: ./pics/output_12_0.png "HOG Features HSV No Veh 01"
[hog_HSV_No_02]: ./pics/output_12_1.png "HOG Features HSV No Veh 02"

[feat_Veh_01]: ./pics/output_15_0.png "Normalized Features Veh 01"
[feat_Veh_02]: ./pics/output_15_1.png "Normalized Features Veh 02"

[feat_No_01]: ./pics/output_16_0.png "Normalized Features No Veh 01"
[feat_No_02]: ./pics/output_16_1.png "Normalized Features No Veh 02"


[find_boxes_01]: ./pics/output_13_0.png "Find Boxes 01"
[find_boxes_02]: ./pics/output_13_1.png "Find Boxes 02"
[find_boxes_03]: ./pics/output_13_2.png "Find Boxes 03"
[find_boxes_04]: ./pics/output_13_3.png "Find Boxes 04"
[find_boxes_05]: ./pics/output_13_4.png "Find Boxes 05"
[find_boxes_06]: ./pics/output_13_5.png "Find Boxes 06"

[heat_01]: ./pics/output_17_0.png "Heatmap 01"
[heat_02]: ./pics/output_17_1.png "Heatmap 02"
[heat_03]: ./pics/output_17_2.png "Heatmap 03"
[heat_04]: ./pics/output_17_3.png "Heatmap 04"
[heat_05]: ./pics/output_17_4.png "Heatmap 05"
[heat_06]: ./pics/output_17_5.png "Heatmap 06"

[video]: ./out_test.mp4 "Video"

### General note on the Jupyter Notebook

I reused code from the lecture wherever possible to speed up my work. I ended up writing only very few lines additionally to bring the given pieces together.  I also experimented with different parameter settings a lot, but the parameters used in the lecture examples turned out to be pretty close to optimal for me.

##Histogram of Oriented Gradients (HOG)

The relevant code can be found in the notebook "./P5.ipynb" in the section titled "Feature Extract Functions"

I started by reading in all the `vehicle` and `non-vehicle` images.  

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here are examples using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`.  In each line you see the original image and the HOG of each color channel:


![alt text][hog_01]
![alt text][hog_02]
![alt text][hog_03]
![alt text][hog_04]

You can see that the HOGs for each channel are relatively similar.  As discussed below (see table with comparison of accuracy results) I decided to use all HOG channels for the training and prediction since it reduced the false positives in my tests.  This is however decreasing performance.

Choosing less pixels per cell (I tried `4`) lead to worse recognition results, while increasing this to `16` made the classifier training far too slow. So I kept it at `8`.  I did similar experiments with the number of orientations and the cells per block.  Choosing `9` orientations and `2` cells per block lead to satisfying test results for me. 

### Investigation of other color channels

We saw above already some examples for car pictures with `YCrCb` color channel. Here are 2 more:

![alt text][hog_YCrCb_Veh_01]
![alt text][hog_YCrCb_Veh_02]

It is also interesting to look at non-car images of the `YCrCb` HOG channels:

![alt text][hog_YCrCb_No_01]
![alt text][hog_YCrCb_No_02]

To investigate more I also tried the `HLS` color channel:

![alt text][hog_HLS_Veh_01]
![alt text][hog_HLS_Veh_02]

![alt text][hog_HLS_No_01]
![alt text][hog_HLS_No_02]

And here is the same analysis for `HSV`:

![alt text][hog_HSV_Veh_01]
![alt text][hog_HSV_Veh_02]

![alt text][hog_HSV_No_01]
![alt text][hog_HSV_No_02]

Furthermore, here is a visualization of the full feature vectors and their normalization for car and non-car pictures

![alt text][feat_Veh_01]
![alt text][feat_Veh_02]

![alt text][feat_No_01]
![alt text][feat_No_02]


##Classifier

The relevant code can be found in the notebook "./P5.ipynb" in the section titled "Train the Classifier"

I took the vehicle and non-vehicle pictures and shuffled their order.  Then I took `6000` samples from each set to train the model with.  Using more samples made the training go slower and didn't lead to greater accuracy.  I suspect the repetitive nature of the car pictures is to blame for that.

To train the classifier, we first use the `extract_features` function to get all features for the car and non-car pictures.  The feature vectors are then normalized using a `StandardScaler`.  The training and test sets are separated with the `train_test_split` function from `sklearn.cross_validation`. 

As classifier I chose a linear support vector machine. It took about 2.5 seconds to train the SVM and the test accuracy was about 99%.  Again, the repetitive nature of the car image examples might skew the accuracy to high.  When testing manually the number of test errors was indeed quite low though.

After the svm is trained, I save it in a pickle file.  Along with the model I saved the parameter settings and the scaler.  This simplified testing the trained model when I interrupted my work on this project for a day or two.

### Comparison of Classifier Accuracy for Different Parameter Settings

Here is a table of my results:

| acc    | sample_size | color_space | hog_channel | hog_feat | spatial_feat | hist_feat | total_time | n_features |
| ------ | ----------- | ----------- | ----------- | -------- | ------------ | --------- | ---------- | ---------- |
| 0.9788 | 1000        | YUV         | ALL         | True     | False        |False      |36.3        |5292        |
| 0.9773 | 1000        | HSV         | 2           | True     | True         |True       |41.8        |2580        |
| 0.8818 | 1000        | YCrCb       | 2           | True     | False        |False      |34.1        |1764        |
| 0.8348 | 1000        | HLS         | 2           | True     | False        |False      |26.3        |1764        |
| 0.9197 | 1000        | HSV         | ALL         | False    | True         |True       |20.3        |816         |
| 0.9788 | 1000        | YUV         | ALL         | True     | True         |True       |36.1        |6108        |
| 0.9364 | 1000        | HSV         | 2           | True     | False        |False      |27.8        |1764        |
| 0.9864 | 1000        | HSV         | ALL         | True     | True         |True       |32.9        |6108        |
| 0.9682 | 1000        | HSV         | ALL         | True     | False        |False      |41.1        |5292        |
| 0.9879 | 1000        | YCrCb       | ALL         | True     | True         |True       |30.3        |6108        |
| 0.9455 | 1000        | LUV         | 2           | True     | True         |True       |45.2        |2580        |
| 0.95 	 | 1000        | YCrCb       | 2           | True     | True         |True       |37.0        |2580        |
| 0.9394 | 1000        | YCrCb       | ALL         | False    | True         |True       |21.6        |816         |
| 0.9227 | 1000        | RGB         | 2           | True     | False        |False      |41.7        |1764        |
| 0.9364 | 1000        | HLS         | 2           | True     | True         |True       |38.8        |2580        |
| 0.9773 | 1000        | YCrCb       | ALL         | True     | False        |False      |37.8        |5292        |
| 0.8182 | 1000        | LUV         | 2           | True     | False        |False      |51.0        |1764        |
| 0.9667 | 1000        | RGB         | 2           | True     | True         |True       |108.7       |2580        |

The optimal setting was `YCrCb` with all hog channels and all spatial and hist features.


##Sliding Window Search

The relevant code can be found in the notebook "./P5.ipynb" in the section titled "The find_cars Function".

The `find_cars` function takes as input the image to search in, parameters for the sliding windows search (`ystart`, `ystop`, and `scale`), the classifier and the scaler, and parameters for the feature extraction.

To speed up the search the function applies the HOG feature extraction on the entire (scaled) region of interest of the picture (instead of applying it in each search window). 

The sliding windows are then computed, given the parameters `pix_per_cell` and `cells_per_step`.  Essentially we start in the top left of the region of interest and look at cells of size `8*8`.  In each step we move  `cells_per_step` to the right, resp. downwards.  Since `cells_per_step` was set to be `2` this makes a `75%` overlap.  We could probably work with a `50%` overlap to make the detection a bit speedier but more lossy. I decided to keep the not-so-fast but accurate approach. 

In each window, we get the HOG features (for all channels), the spatial features and the color histogram.  This is combined to one scaled feature vector and fed to the classifier.  Depending on the prediction of the classifier (`1` for car and `0` for non-car) the given window is stored and later returned.

Below you see `6` images where I applied the `find_cars` method with sliding windows in the y-strip `400` to `656` pixels and with a scaling factor of `1.5`:

![alt text][find_boxes_01]
![alt text][find_boxes_02]
![alt text][find_boxes_03]
![alt text][find_boxes_04]
![alt text][find_boxes_05]
![alt text][find_boxes_06]

You can see that the actual cars are detected well while false positives exist but are quite rare.

In my actual pipeline I will call the `find_cars` function multiple times.  Each time with a slightly different region of interest and scale factor.  The idea behind that is the following:  Cars which are further away will appear more in the middle (height-wise) of the picture and require less scaling while cars which are close to the camera will be lower in the picture (heigh `y`-value) and require a larger scaling.  So I call the `find_cars` function `3` times  with the following search window parameters (e.g. see section "Test the Heatmap Tools on Example Images" and "Pipeline for One Image").

```python
ystart=350, ystop=606, scale=1.0,
ystart=400, ystop=656, scale=1.5
ystart=450, ystop=706, scale=2.0
```


###Applying Heatmaps

The relevant code can be found in the notebook "./P5.ipynb" in the section titled "Heatmap Tools".

To deal with false positive I implemented some heatmap tools (just like in the lecture).  The heat maps basically combine the returned boxes from `find_cars`, and, given a threshold, filter out regions with a sufficient amount of detections.

In the section "Test the Heatmap Tools on Example Images" I plotted the heatmaps and the original image with the combined box for `6` example images.  See the resulting images here:

![alt text][heat_01]
![alt text][heat_02]
![alt text][heat_03]
![alt text][heat_04]
![alt text][heat_05]
![alt text][heat_06]

## Pipeline for One Image

The relevant code can be found in the notebook "./P5.ipynb" in the section titled "Pipeline for One Image".

First, to simplify smoothing between frames, I added a class `HeatMapBuffer` which stores the last `N` heat-maps in a queue.  In each frame of the pipeline I will use this to average over the last `10` frames.

For each image we first call the `find_cars` function for `3` different regions and scales (as described above). The resulting detected boxes are the combined in a heat map and smoothed over the last 10 frames. We then apply a threshold (`1` was enough to keep out false positives) and clip the resulting heat-map.  This heat-map is used to find the combined boxes that finally mark the complete cars.

## Video Implementation

The relevant code can be found in the notebook "./P5.ipynb" in the section titled "Video".

We apply the pipeline to the example video as in the previous projects.

Here's a [link to my output video result](./out_test.mp4)

---

##Discussion

Generally the vehicle detection works quite well.  The cars in the output video are almost always detected and no false positives can been seen.

One issue with the detection is that cars in the distance are not recognized.  I assume that we could add another search window layer which scans only the horizon.  On the other hand, the cars in the far distance are not very relevant to us so investing more computational time to find them is in my opinion a waste of resources.

My biggest concern about my solution is the performance.  It took about 45 min. to run the code on the example video which is far from 'real-time'.  I think I could use less features for the classification (e.g. only one HOG channel) and I also could limit the use of the search windows to smaller regions or less overlap. 

Another issue seems to be that sometimes, two cars close to each other are recognized as one.  This I assume could lead to issues for the final goal of a self-driving car.  
