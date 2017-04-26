
**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

If you want to see all the functions used for this project please view the python notebook.  I deleted them from this README.md file to prevent overrun and keep this as concise as possible.  Other then those functions though this file is the same as the notebook.


```python
cars = []
non_cars = []

car_files = os.listdir('vehicles/')
non_car_files = os.listdir('non-vehicles/')

for imgtype in car_files:
    cars.extend(glob.glob('vehicles/' + imgtype + '/*'))

for imgtype in non_car_files:
    non_cars.extend(glob.glob('non-vehicles/' + imgtype + '/*'))

print('cars: ', len(cars))
print('non-cars: ', len(non_cars))
```

    cars:  8792
    non-cars:  8968



```python
figures = {}

car_i = np.random.randint(0, len(cars))
non_car_i = np.random.randint(0, len(non_cars))

car_image = cv2.imread(cars[car_i])
RGB_car_image = cv2.cvtColor(car_image, cv2.COLOR_BGR2RGB)
figures[0] = RGB_car_image

non_car_image = cv2.imread(non_cars[non_car_i])
RGB_non_car_image = cv2.cvtColor(non_car_image, cv2.COLOR_BGR2RGB)
figures[1] = RGB_non_car_image

plot_figures(figures, 1, 2)
```


![png](output_4_0.png)


This cell above shows an example of the car and non-car images I loaded for use.


```python
#lets see if we can see any differences in color spaces that might prove useful
rgb_car_image = cv2.cvtColor(car_image, cv2.COLOR_BGR2RGB)
rgb_non_car_image = cv2.cvtColor(non_car_image, cv2.COLOR_BGR2RGB)

hsv_car_image = cv2.cvtColor(car_image, cv2.COLOR_BGR2HSV)
hsv_non_car_image = cv2.cvtColor(non_car_image, cv2.COLOR_BGR2HSV)

luv_car_image = cv2.cvtColor(car_image, cv2.COLOR_BGR2LUV)
luv_non_car_image = cv2.cvtColor(non_car_image, cv2.COLOR_BGR2LUV)

hls_car_image = cv2.cvtColor(car_image, cv2.COLOR_BGR2HLS)
hls_non_car_image = cv2.cvtColor(non_car_image, cv2.COLOR_BGR2HLS)

yuv_car_image = cv2.cvtColor(car_image, cv2.COLOR_BGR2YUV)
yuv_non_car_image = cv2.cvtColor(non_car_image, cv2.COLOR_BGR2YUV)

ycrcb_car_image = cv2.cvtColor(car_image, cv2.COLOR_BGR2YCrCb)
ycrcb_non_car_image = cv2.cvtColor(non_car_image, cv2.COLOR_BGR2YCrCb)

rgb_image2 = cv2.cvtColor(car_image, cv2.COLOR_BGR2RGB)/255.

#lets use the plot3d method from the lessons
plot3d(rgb_car_image, rgb_image2)
plt.title('RGB Car')
plt.show()

plot3d(rgb_non_car_image, rgb_image2)
plt.title('RGB Non-Car')
plt.show()

plot3d(hsv_car_image, rgb_image2, axis_labels=list("HSV"))
plt.title('HSV Car')
plt.show()

plot3d(hsv_non_car_image, rgb_image2, axis_labels=list("HSV"))
plt.title('HSV Non-Car')
plt.show()

plot3d(luv_car_image, rgb_image2, axis_labels=list("LUV"))
plt.title('LUV Car')
plt.show()

plot3d(luv_non_car_image, rgb_image2, axis_labels=list("LUV"))
plt.title('LUV Non-Car')
plt.show()

plot3d(hls_car_image, rgb_image2, axis_labels=list("HLS"))
plt.title('HLS Car')
plt.show()

plot3d(hls_non_car_image, rgb_image2, axis_labels=list("HLS"))
plt.title('HLS Non-Car')
plt.show()

plot3d(yuv_car_image, rgb_image2, axis_labels=list("YUV"))
plt.title('YUV Car')
plt.show()

plot3d(yuv_non_car_image, rgb_image2, axis_labels=list("YUV"))
plt.title('YUV Non-Car')
plt.show()

plot3d(ycrcb_car_image, rgb_image2, axis_labels=['Y', 'Cr', 'Cb'])
plt.title('YCrCb Car')
plt.show()

plot3d(ycrcb_non_car_image, rgb_image2, axis_labels=['Y', 'Cr', 'Cb'])
plt.title('YCrCb Non-Car')
plt.show()
```


![png](output_6_0.png)



![png](output_6_1.png)



![png](output_6_2.png)



![png](output_6_3.png)



![png](output_6_4.png)



![png](output_6_5.png)



![png](output_6_6.png)



![png](output_6_7.png)



![png](output_6_8.png)



![png](output_6_9.png)



![png](output_6_10.png)



![png](output_6_11.png)


Looks like YCrCb is the most clustered. It may yield better results.

<h1>Histogram of Oriented Gradients (HOG)</h1>

The parameters used below were most effective given time constraints. Tested several, but settled with these.  
color_space = 'YCrCb'

orient = 9

pix_per_cell = 8

cell_per_block = 2


hog_channel = 'ALL'

spatial_size = (16, 16)

hist_bins = 32

To extract the HOG features, I used the hog fucntion from skimage.feature library.


```python
# Define feature parameters
color_space = 'YCrCb'  # RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9
pix_per_cell = 8
cell_per_block = 2

hog_channel = 'ALL'  # 0, 1, 2, "ALL"
spatial_size = (16, 16)
hist_bins = 32
spatial_feat = True
hist_feat = True
hog_feat = True

figures = {}

figures[0] = rgb_car_image
hog_features, car_hog_image = get_hog_features(ycrcb_car_image[:,:,1], orient, pix_per_cell, cell_per_block, vis=True, feature_vec=True)
figures[1] = car_hog_image

figures[2] = rgb_non_car_image
hog_features, non_car_hog_image = get_hog_features(ycrcb_non_car_image[:,:,1], orient, pix_per_cell, cell_per_block, vis=True, feature_vec=True)
figures[3] = non_car_hog_image

plot_figures(figures, 2, 2)
```


![png](output_9_0.png)


Here we extract features for car and non-car images and then train our classifier.

To train the classifier I used Linear SVC.  


```python
car_features = extract_features(cars, color_space=color_space, spatial_size=spatial_size,
            hist_bins=hist_bins, orient=orient, pix_per_cell=pix_per_cell,
            cell_per_block=cell_per_block, hog_channel=hog_channel,
            spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)

non_car_features = extract_features(non_cars, color_space=color_space, spatial_size=spatial_size,
            hist_bins=hist_bins, orient=orient, pix_per_cell=pix_per_cell,
            cell_per_block=cell_per_block, hog_channel=hog_channel,
            spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)

X = np.vstack((car_features, non_car_features)).astype(np.float64)
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(non_car_features))))

# Split up data into randomized training and test sets
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.10, random_state=np.random.randint(0, 100))

print('Using: ', orient, 'orientations, ', pix_per_cell, 'pixels per cell, ', cell_per_block, 'cells per block, ',
      hist_bins, 'histogram bins, and ', spatial_size, 'spatial sampling')
print('Feature vector length: ', len(X_train[0]))

# Linear SVC
# svc = SVC(kernel='rbf')
svc= LinearSVC(C=0.00001)

t = time.time()
svc.fit(X_train, y_train)

print(round(time.time()-t, 2), 'Seconds to train SVC...)')
print('Test Accuracy: ', round(svc.score(X_test, y_test), 4))
```

    Using:  9 orientations,  8 pixels per cell,  2 cells per block,  32 histogram bins, and  (16, 16) spatial sampling
    Feature vector length:  6156
    3.9 Seconds to train SVC...)
    Test Accuracy:  0.987


In order to find the cars I use a sliding window search or multiple convolutions of different sizes.  The scales were 1.5 to 2 to simplify things.  I could of use more, but this seemed good enough to capture the most important cars.  I also defined a region of interst to speed up detection since we don't have to detect any flying cars as of yet.  After that I tried to detect the areas with the most activity.  I did this using the label method from scipy.ndimage.measurements.  It helps calculating points of interests. After that I applied a threshold to set values under a specific amount will be set to zero.


```python
test_images = glob.glob('test_images/*')
y_start_stop = (400, 650)
scales = [1.5, 2]

figures = {}
count = 0
for img in test_images:
    img = cv2.imread(img)
    output_img, windows = find_cars(img, y_start_stop[0], y_start_stop[1], scales, svc, X_scaler, orient,
                                    pix_per_cell, cell_per_block, spatial_size, hist_bins)

    heat_map = np.zeros_like(img[:, :, 0])
    heat_map = add_heat(heat_map, windows)
    heat_map = np.clip(heat_map, 0, 255)
    heat_map = apply_threshold(heat_map, 0)

    labels = label(heat_map)
    figures[count] = draw_labeled_bboxes(np.copy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), labels)
    count += 1
    figures[count] = heat_map
    count += 1

plot_figures(figures, 8, 2)
```


![png](output_13_0.png)


<h1>Video Implementation</h1>


```python
past_frames = []

def process_image(img):
    global past_frames

    output_img, windows = find_cars(img, y_start_stop[0], y_start_stop[1], scales, svc, X_scaler, orient,
                                    pix_per_cell, cell_per_block, spatial_size, hist_bins)

    past_frames.append(windows)
    past_frames = past_frames[-15:]

    heat_map = np.zeros_like(img[:, :, 0])
    heat_map = add_heat(heat_map, [window for windows in past_frames for window in windows])
    heat_map = np.clip(heat_map, 0, 255)
    heat_map = apply_threshold(heat_map, 15)

    labels = label(heat_map)
    draw_img = draw_labeled_bboxes(np.copy(img), labels)
    return draw_img
```

My output video is can be created and found by name "project2_vid.mp4" It was created using the code below.  Here is a link to the video output : https://youtu.be/_V4ykY_Keb8


```python
from moviepy.editor import VideoFileClip
from IPython.display import HTML

project_output = 'project2_vid.mp4'
clip1 = VideoFileClip('project_video.mp4')
project_clip = clip1.fl_image(process_image)
%time project_clip.write_videofile(project_output, audio=False)
```

    [MoviePy] >>>> Building video project2_vid.mp4
    [MoviePy] Writing video project2_vid.mp4


    100%|█████████▉| 1260/1261 [10:12<00:00,  2.06it/s]


    [MoviePy] Done.
    [MoviePy] >>>> Video ready: project2_vid.mp4

    CPU times: user 18min 19s, sys: 1min 55s, total: 20min 14s
    Wall time: 10min 12s


<h1>Discussion</h1>
Ultimately this project showed how to do some basic detection, but I still have noise and false positives I would have to deal with in order to get a better detector.  I think if I stepped through the video frame by frame I could better tune the parameters to give better results.  


```python

```
