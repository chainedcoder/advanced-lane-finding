## Advanced Lane Finding

### T his project is a simple solution to finding lanes for the Advance Lane Finding Project in the udacity nano degree. The steps followed are:

1. Calibrate camera
2. Undistort Image
3. Color gradient
4. Alter perspective to "birds-eye view"
5. Find lane pixels
6. Use sliding window
7. skip sliding window if previous best fit exists
8. Fit polinomial
9. Output results to original image

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/calibration_output.png "Calibration"
[image2]: ./output_images/undistorted_output.png "Undistorted"
[image3]: ./output_images/color_gradient_output.png "COlor Gradient"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Calibration.

The code for this step is contained in the first code cell of the IPython notebook located in "./examples/example.ipynb" (or in lines # through # of the file called `some_file.py`).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  

![alt text][image1]

### Pipeline (single images)

#### 1. Distortion-correction.

I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 
![alt text][image2]

#### 2. Color and Gradient.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at cell titled step3). While in the end I used lab and hsv channel, I retained the code I had used with the hsl channel as had been in demostrated in class. The lab channel help me pick up the yellow lanes better and avoid the darker cracks and lines of the road in the challenge video. While I didn't do a perfect job, and while I got a tiny glitch in the testVideo after this modification, I found the approach more useful. 

![alt text][image3]

#### 3. Perspective Transform.

The code for my perspective transform includes a function called `perspective_transform()`, in the cell title Alter Perspective (This is also in the 2nd Last code cell in the peiple class).  The `perspective_transform()` function takes as inputs an image (`img`) abd (`edges`).  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32([(540, 450), (740, 450), (-450, h), (1730, h)])
dst = np.float32([(100, 0), (1180, 0), (100, h), (1180, h)])  
```

I chose a wide base because I realized I was able to, even though for just a few seconds, mark out the lanes in the harder challenge video. Here's an example of the result I got

![alt text][image4]

#### 4. Identifying the lane lines and fitting a polynomial

After getting the lane marks appearing a lot more clearing(using the steps above), I mapped out a histogram to locate the lane mark concetration on the image. This was especially hard in the harder challenge where the road curved a lot (I didn't get to figure it out). 

I then split the histogram into the left and right sections and used max function to get the bases in either side. I eliminated regions to close to the centre to or too far of:

``` python
    binary_warped = altered_img
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    midpoint = np.int(histogram.shape[0]//2)
    q_point = np.int(midpoint//2)
    eighth = np.int(q_point//2)
    leftx_base = np.argmax(histogram[eighth:int(q_point+eighth)])+eighth
    rightx_base = np.argmax(histogram[midpoint:(midpoint+q_point)]) + midpoint
```

Once I got the starting points of the sliding window, I wound my way up in each iteration(of 10) adjust the width based on the average of the concetration of the points. To account for sharper bends, I adjust the position of the search windows in both the x and y axis. This looked a bit messy when mapped out(some pixels were being checked multiple time) but It got the job done better than without - espcially with the challenge and harder challenge. The code is in the section titled "Find Lane Pixels". The function is called `sliding_window_search`. 

For subsequent images frames for the video, I used fitted polynomial to avoid sliding the window approach. The function is called `fit_polynomial()` (you can find the code in the final pipeline class(in the 3rd last code cell).

![alt text][image5]

#### 5. Calculating the Radius and distance from centre

I dused the radius equation to calculate the radius of the curve. The method `measure_curvature_real(self)` in the Pipeline Class (3rd last code cel) demostrates this. I then calculated the distance from the centre in the methods `calc_centre_dist(self)`

```python
    def measure_curvature_real():
        '''
        Calculates the curvature of polynomial functions in meters.
        '''
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension

        y_eval = np.max(ploty) * ym_per_pix

        left_fit_scaled = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
        right_fit_scaled = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)


        left_curverad = ((1 + (2 * left_fit_scaled[0] * y_eval  + left_fit_scaled[1])**2)**1.5)/np.absolute(2*left_fit_scaled[0])  ## Implement the calculation of the left line here
        right_curverad =  ((1 + (2 * right_fit_scaled[0] * y_eval + right_fit_scaled[1])**2)**1.5)/np.absolute(2*right_fit_scaled[0])  ## Implement the calculation of the right line here

        right_pos = (left_fit[0]*y_eval**2 + left_fit[1]*y_eval + left_fit[2])
        left_pos = (right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2])
        centre_pos = (right_pos + left_pos)/2
        centre_diff = ((result_img.shape[1]/2) - centre_pos) * xm_per_pix

        return left_curverad, right_curverad, centre_diff 
```

#### 6. Mapping and Projecting result unto original image.

The final phase was to draw this into the original image. I reversed the perspective using the inverse matrix and then drew out the result. THe final code version is `draw_lane(self)` in the pipeline class(3rd last code cell).
```python
    def draw_lane(warped):        
        M = Mrec
        
        try:
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        except TypeError:
            # Avoids an error if `left` and `right_fit` are still none or incorrect
            print('The function failed to fit a line!')
            left_fitx = 1*ploty**2 + 1*ploty
            right_fitx = 1*ploty**2 + 1*ploty
            

        # Create an image to draw the lines on
        warped_binary = np.zeros_like(warped).astype(np.uint8)
        out_img = warped

        # Recast the x and y points into usable format for cv2.fillPoly()
        left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        left_right = np.hstack((left, right))

        # Draw the lane onto the warped blank image
        cv2.polylines(out_img, np.int32([left]), False, (255,0,255), 15)
        cv2.polylines(out_img, np.int32([right]), False, (0,0,255), 15)
        cv2.fillPoly(out_img, np.int_([left_right]), (0,255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        out = cv2.warpPerspective(out_img, M, (img.shape[1], img.shape[0])) 
        
        left_curve_txt = "Left Radius: "+ str(int(left_curverad))+ "m"
        right_curve_txt = "Right Radius: "+ str(int(right_curverad))+ "m"
        centre_txt = "Car is: {:.2f}".format(centre_diff) + "m off centre"

        cv2.putText(img, left_curve_txt, (100,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(img, right_curve_txt, (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255), 2)
        cv2.putText(img, centre_txt, (100,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2 )

        # Combine the result with the original image
        marked_lane_img = cv2.addWeighted(img, 1, out, 0.3, 0)
        return marked_lane_img

```

![alt text][image6]

---

### Pipeline (video)

#### Sample video results.

Here's the [link to my video result](./project_video_output.mp4)
Here's the [link to my challenge video result](./challenge_video_output.mp4)

---

### Discussion

#### Issues and challenges

I could definately improve the sliding window algorithm by avoiding checking same area when subsequent window cells overlap. 

In the challenge video, I especially hard difficulty with the areas of the road that are same color as the lanes where the lane curve. This made it especiall diffuclt to detect the start points using the histogram approach. While I improved this with the lab and hsv channel the result were not optimal. I haven't figure how to solve this properly.

Also the polinomial fit didn't work reliably for the challenge video. It almost always didn't work for the harder challenge. There must be a way better mark out the curves on these difficult roads.
