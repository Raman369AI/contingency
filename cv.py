import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np


def readAndWritePixelRegion():
     root = os.getcwd()
     imgPath = os.path.join(root,'rhm.jpg')
     img = cv.imread(imgPath)
     #matplotlib takes RGB, the imread by default takes BGR
     # cvtColor is used to conver from one colorscheme to another.
     imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
     #y axis comes first and then the X axis
     eyeRegion = imgRGB[26:127,126:200]
     dx = 200 - 126
     dy = 127 - 26
     startX = 220 
     startY = 100
     imgRGB[startY:startY+dy,startX:startX+dx] = eyeRegion
     plt.figure()
     plt.imshow(imgRGB)
     plt.show()
     debug = 1


'''RGB color channels - Image has 3 channels, Blue Green and Red, the order is flipped for OpenCV - BGR, and matplotlib is for RGB
Channels is used for image segmentation, Green is used to segment out.
'''

def pureColors():
     zeroes = np.zeros((100,100))
     ones = np.ones((100,100))
     # Gives red image given the Matplotlib uses RGB.
     bImg = cv.merge((255*ones,zeroes, zeroes))
     gImg = cv.merge((zeroes,255*ones, zeroes))
     rImg = cv.merge((zeroes,zeroes,255*ones))
     blackImg = cv.merge((zeroes,zeroes,zeroes))
     whiteImg = cv.merge((255*ones,255*ones,255*ones))
     plt.figure()
     plt.subplot(231)
     plt.imshow(bImg)
     plt.subplot(232)
     plt.imshow(gImg)
     plt.subplot(233)
     plt.imshow(rImg)
     plt.subplot(234)
     plt.imshow(blackImg)
     plt.subplot(235)
     plt.imshow(whiteImg)
     plt.show()
     debug = 1

def bgrChannelGrayScale():
     root = os.getcwd()
     imgPath = os.path.join(root,'rhm.jpg')
     img = cv.imread(imgPath)
     b,g,r = cv.split(img) # each one is a 2D array
     zeros = np.zeros_like(b)
    #Without color mapping it just shows Grayscale
     blue_img = cv.merge([b, zeros, zeros])
     green_img = cv.merge([zeros, g, zeros])
     red_img = cv.merge([zeros, zeros, r])
     blue_img_rgb = cv.cvtColor(blue_img, cv.COLOR_BGR2RGB)    
     green_img_rgb = cv.cvtColor(green_img, cv.COLOR_BGR2RGB)    
     red_img_rgb = cv.cvtColor(red_img, cv.COLOR_BGR2RGB)    
     ones = np.ones_like(b)
     plt.figure()
     plt.subplot(231)
     plt.imshow(blue_img_rgb)
     plt.subplot(232)
     plt.imshow(green_img_rgb)
     plt.subplot(233)
     plt.imshow(red_img_rgb)
     plt.show()


'''Greyscale channel is a single channel, Reduces the amount of data due to single channel - processing power is less - preprocessing is efficient.
GrayScale pixel = 0.299R + 0.587G + 0.114B
'''

def grayScale():
     root = os.getcwd()
     imgPath = os.path.join(root,'rhm.jpg')
     img = cv.imread(imgPath)
     imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
     cv.imshow('gray',imgGray)
     cv.waitKey(0)

def readAsGray():
     root = os.getcwd()
     imgPath = os.path.join(root,'rhm.jpg')
     img = cv.imread(imgPath, cv.IMREAD_GRAYSCALE)
     cv.imshow('gray',img)
     cv.waitKey(0)


'''HSV Color space: Hue(0-180)-only openCV others use 360, Saturation(0-255) and Value(0-255)
H- Type of the color
S - Saturation of the color -diluting the color with water.
V - Intensity of the color

Color based segmentation - More intuitive ways of describing the colors.

Normalize RGB, cmax and cmin and then the difference, calculate the Hue calculation.
Saturation and Value calculation = cmax*100.
'''


def hscColorSegmentation():
     root = os.getcwd()
     imgPath = os.path.join(root,'rhm.jpg')
     img = cv.imread(imgPath)
     imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)

     imgHSV = cv.cvtColor(imgRGB, cv.COLOR_BGR2HSV)
     h,s,v = cv.split(imgHSV)
     lowerBound = np.array([130,50,30])#, HSV values
     upperBound = np.array([180,70,35])
     mask = cv.inRange(imgHSV,lowerBound,upperBound)

     plt.figure()
     plt.subplot(231)
     plt.imshow(imgRGB)
    #  plt.subplot(232)
    #  plt.imshow(s)
    #  plt.subplot(233)
    #  plt.imshow(v)
     plt.show()
     cv.imshow('mask',mask)
     cv.waitKey(0)


'''Image Resizing - Making the image smaller and bigger - Reduce the data , proprocessing step for Image, Improve the resolution.
Nearest Neighbour, Linear, Cubic, 2d Nearest, Bilinear and BiCubic.
Interpolation Method	OpenCV Constant	Title	Best Use Case
Nearest Neighbor	    cv.INTER_NEAREST	nearest	Fast, low-quality resizing
Bilinear	              cv.INTER_LINEAR	     linear	General resizing
Bicubic	              cv.INTER_CUBIC	     bicubic	High-quality upscaling
Area-based	         cv.INTER_AREA	     area	     High-quality downscaling
Lanczos	              cv.INTER_LANCZOS4	lanczos4	Best quality upscaling
'''
def imgResize():
     root = os.getcwd()
     imgPath = os.path.join(root,'rhm.jpg')
     img = cv.imread(imgPath)
     imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
     imgHSV = cv.cvtColor(imgRGB, cv.COLOR_BGR2HSV)
     img = img[143:1148,1213:2570]
     height,weight,_ = img.shape
     scale = 1/4

     interpMethods = [cv.INTER_NEAREST, cv.INTER_LINEAR, cv.INTER_CUBIC, cv.INTER_AREA, cv.INTER_LANCZOS4]
     interpTitle = ['area', 'linear', 'bicubic', 'nearest', 'lanczos4']     
     plt.figure()
     plt.subplot(231)
     plt.imshow(img)
     for i  in range(len(interpMethods)):
          plt.subplot(2,3,i+2)
          imgResize = cv.resize(img, (int(weight*scale),int(height*scale)), interpolation = interpMethods[i])
          plt.imshow(imgResize)
          plt.title(interpTitle[i])
     plt.show()



'''Image Histogram : Shows the distribution of pixel intensity.
Thresholding, Equalisation and enhancement, color analysis and color segmentation
'''


def greyHistogram():
     root = os.getcwd()
     imgPath = os.path.join(root,'rhm.jpg')
     img = cv.imread(imgPath, cv.IMREAD_GRAYSCALE)
     imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
     plt.figure()
     plt.imshow(imgRGB)
     hist = cv.calcHist([img],[0],None,[256],[0,256]) #([list of images],[channels],mask,#Bins, #Pixel value range)
     plt.figure()
     plt.plot(hist)
     plt.xlabel('bins')
     plt.ylabel('number of pixels')
     plt.show()

def colorHistogram():
     root = os.getcwd()
     imgPath = os.path.join(root,'rhm.jpg')
     img = cv.imread(imgPath)
     imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
     plt.figure()
     plt.imshow(imgRGB)
     colors = ['b','g','r']
     plt.figure()
     for i in range(len(colors)):
          hist = cv.calcHist([imgRGB],[i],None,[256],[0,256]) #([list of images],[channel],mask,#Bins, #Pixel value range)
          plt.plot(hist,colors[i])
     plt.xlabel('pixel intensity')
     plt.ylabel('number of pixels')
     plt.show()

def histogramRegion():
     root = os.getcwd()
     imgPath = os.path.join(root,'rhm.jpg')
     img = cv.imread(imgPath)
     imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
     imgRGB = imgRGB[143:1148,1213:2570]
     height,weight,_ = imgRGB.shape
     plt.figure()
     plt.imshow(imgRGB)
     colors = ['b','g','r']
     plt.figure()
     for i in range(len(colors)):
          hist = cv.calcHist([imgRGB],[i],None,[256],[0,256]) #([list of images],[channel],mask,#Bins, #Pixel value range)
          plt.plot(hist,colors[i])
     plt.xlabel('pixel intensity')
     plt.ylabel('number of pixels')
     plt.show()   


'''2D Convolution, kernel matrix and do Feature Extraction, Image Enhancment , edge detection, kernel should not be larger than the image'''      
def convolution2d():
     root = os.getcwd()
     imgPath = os.path.join(root,'rhm.jpg')
     img = cv.imread(imgPath)
     imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
     n = 100
     kernel = np.ones((n,n),np.float32)/(n*n)
     imgConv = cv.filter2D(img,-1,kernel)
     plt.figure()
     plt.imshow(imgRGB)
     plt.figure()
     plt.imshow(imgConv)
     plt.show()
def callback(input):
     pass
'''Average filtering, do 2D convolution to get the average value of the neighbours, smoothing and noise reduction'''
def avgFiltering():
     root = os.getcwd()
     imgPath = os.path.join(root,'rhm.jpg')
     img = cv.imread(imgPath)
     imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
     winName = 'avg filter'
     cv.namedWindow(winName)
     cv.createTrackbar('n',winName,1,100,callback)
     height,weight,_ = imgRGB.shape
     while True:
          if cv.waitKey(1) == ord('q'):
               break
          n = cv.getTrackbarPos('n',winName)
          imgFilter = cv.blur(imgRGB,(n,n))
          cv.imshow(winName,imgFilter)
     cv.destroyAllWindows()

'''Median filtering, do 2D convolution to get the median
 value of the neighbours, Noise reduction, Edge preservation'''

def medianFiltering():
     root = os.getcwd()
     imgPath = os.path.join(root,'rhm.jpg')
     img = cv.imread(imgPath)
     imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
     noisyImg = imgRGB.copy()
     noiseProb = 0.05
     noise = np.random.rand(noisyImg.shape[0],noisyImg.shape[1])
     noisyImg[noise < noiseProb/2] = 0
     noisyImg[noise > 1 - noiseProb/2] = 255
    
     imgFilter = cv.medianBlur(noisyImg,3)
     plt.figure()
     plt.subplot(121)
     plt.imshow(noisyImg)
     plt.subplot(122)
     plt.imshow(imgFilter)
     plt.show()
     # n = 100
     # kernel = np.ones((n,n),np.float32)/(n*n)
     # imgConv = cv.filter2D(img,-1,kernel)
     # plt.figure()
     # plt.imshow(imgRGB)
     # plt.figure()
     # plt.imshow(imgConv)
     # plt.show()


def gaussianKernel(size, sigma):
     kernel = cv.getGaussianKernel(size, sigma)
     kernel = np.outer(kernel, kernel.T)
     return kernel

'''Gaussian Filtering: Uses Gaussian filtering, noise reduction and preprocessing
Kernel size n must be odd, always n should be a minimum of 6 sigma outside +1, 
to cover 2 std 
Sigma - Controls the strength of the blur and Higher sigma means more blur
Kernel size - large to cover most of the gaussian area

'''
def gaussianFiltering():
     root = os.getcwd()
     imgPath = os.path.join(root,'rhm.jpg')
     img = cv.imread(imgPath)
     imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
     n =51
     fig = plt.figure()
     plt.subplot(121)
     kernel = gaussianKernel(n,8)
     plt.imshow(kernel)
     plt.show()

     winName = 'gaussian filter'
     cv.namedWindow(winName)
     cv.createTrackbar('sigma',winName,1,100,callback)
     height,width,_ = imgRGB.shape
     scale = 1/4
     width = int(width*scale)
     height = int(height*scale)
     img = cv.resize(imgRGB,(width,height))
     while True:
          if cv.waitKey(1) == ord('q'):
               break
          sigma = cv.getTrackbarPos('sigma',winName)
          imgFilter = cv.GaussianBlur(img,(n,n),sigma)
          cv.imshow(winName,imgFilter)
     cv.destroyAllWindows()


'''Image Thresholding: Techniques converts parts or all of the image to binary (0 or 255)
Image segmentation, Feature extraction and object detection, used for preprocessing
'''

def imgThresholding():
     root = os.getcwd()
     imgPath = os.path.join(root,'rhm.jpg')
     img = cv.imread(imgPath)
     imgRGB = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
     hist = cv.calcHist([imgRGB],[0],None,[256],[0,256])
     plt.figure()
     plt.plot(hist)
     plt.xlabel('bins')
     plt.ylabel('number of pixels')
     plt.show()

     threshOpt = [cv.THRESH_BINARY, cv.THRESH_BINARY_INV, cv.THRESH_TRUNC, cv.THRESH_TOZERO, cv.THRESH_TOZERO_INV]
     threshnames  = ['binary', 'binary_inv', 'trunc', 'tozero', 'tozero_inv']
     plt.figure()
     plt.subplot(231)
     plt.imshow(imgRGB,cmap = 'gray')
     for i in range(len(threshOpt)):
          plt.subplot(2,3,i+2)
          _,imgThresh = cv.threshold(imgRGB,125,255,threshOpt[i])
          plt.imshow(imgThresh,cmap = 'gray')
          plt.title(threshnames[i])

     plt.show()


'''Adaptive Thresholding: Finds a threshold value based on the local region
Uneven lighting, varying contract, obtain outline of the image or segment.
Apply threshold value for the region with the constant offset C

Mean Adaptive method
Gaussian Adaptive method
'''
"""
In adaptive thresholding (OpenCV), the so-called "three thresholds" refer to:

- adaptiveMethod: The algorithm used to calculate the local threshold for each pixel (mean or Gaussian).
- blockSize: The size of the neighborhood area used to calculate the local threshold.
- C (offset): A constant subtracted from the computed threshold to fine-tune the result.

These are not three separate cutoff values, but parameters that together control how the threshold for each pixel is determined.
"""

def adaptiveThresholding():
     root = os.getcwd()
     imgPath = os.path.join(root,'rhm.jpg')
     img = cv.imread(imgPath, cv.IMREAD_GRAYSCALE)
     plt.figure()
     plt.subplot(241)
     plt.imshow(img,cmap = 'gray')
     plt.title('original')


     plt.subplot(242)
     _,imgThresh = cv.threshold(img,125,255,cv.THRESH_BINARY)
     plt.imshow(imgThresh,cmap = 'gray')
     plt.title('global threshold')

     plt.subplot(243)
     maxValue = 255
     blockSize = 7
     offset = 2 
     imgThresh = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,blockSize,offset)
     plt.imshow(imgThresh,cmap = 'gray')
     plt.title('mean adaptive')
     
     plt.subplot(244)
     maxValue = 255
     blockSize = 7
     offset = 2 
     imgThresh = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,blockSize,offset)
     plt.imshow(imgThresh,cmap = 'gray')
     plt.title('Gaussian adaptive')
     plt.show()


'''OTSU Binarization: Thresholding technique that tries to seperate the foreground 
and background, tries to seperate the histogram into 2 classes. Automatic way to threshold
1. Compute histogram
2. Iterate through all threshold values
3. Find the threshold values that minimize the weighted in class variances for the two classes
4. Thrshold the image with that value.
'''

def otsuBinarization():
     root = os.getcwd()
     imgPath = os.path.join(root,'rhm.jpg')
     img = cv.imread(imgPath, cv.IMREAD_GRAYSCALE)
     plt.figure()
     plt.subplot(131)
     plt.imshow(img,cmap = 'gray')
     plt.title('original')


     plt.subplot(132)
     thresh = 70
     maxVal = 255
     _,imgThresh = cv.threshold(img,thresh,maxVal,cv.THRESH_BINARY)
     plt.imshow(imgThresh,cmap = 'gray')
     plt.title('global threshold')

     plt.subplot(133)
     arbthresh = 700
     maxVal = 255
     _,imgThresh = cv.threshold(img,thresh,maxVal,cv.THRESH_BINARY+cv.THRESH_OTSU)
     plt.imshow(imgThresh,cmap = 'gray')
     plt.title('OTSU threshold')
     plt.show()


'''Morphological Transformations: Kernel based operation on BINARY IMAGES
Emphasize the foreground or background by changing the size/shape of the image
Good for noise reduction, Image enhancement, segmentation.


Operation	Definition	          Main Effect	                              Typical Use Case
Erosion	Minimum under kernel	Shrinks objects, removes noise	          Remove small white spots, thin objects
Dilation	Maximum under kernel	Expands objects, fills gaps	               Fill holes, connect objects
Opening	Erosion → Dilation	     Removes small noise, smooths contours	     Remove small objects/noise
Closing	Dilation → Erosion	     Fills small holes/gaps, smooths contours	Fill holes, connect cracks
Gradient	Dilation - Erosion	     Highlights edges	                         Edge detection
Tophat	Image - Opening	     Extracts small bright features	          Enhance bright spots
Blackhat	Closing - Image	     Extracts small dark features	               Enhance dark spots
How they affect images:

Erosion/Dilation change the size and connectivity of objects.

Opening/Closing clean up noise and fill small gaps or holes.

Gradient finds edges.

Tophat/Blackhat highlight small features relative to their surroundings.

These operations are fundamental for preprocessing, noise removal, feature extraction, and shape analysis in image processing tasks.

Related
How do erosion and dilation influence the shape of objects in an image
In what ways do opening and closing help reduce noise or fill gaps in images
How can gradient, Tophat, and Blackhat transformations highlight specific features
What effects do these morphological operations have on image details and boundaries
Why are these transformations important for improving image analysis accuracy

'''

def morphTans():
     root = os.getcwd()
     imgPath = os.path.join(root,'rhm.jpg')
     img = cv.imread(imgPath, cv.IMREAD_GRAYSCALE)
     maxValue = 255
     blockSize = 7
     offset = 3
     plt.subplot(241)
     """
cv.ADAPTIVE_THRESH_GAUSSIAN_C:
    Specifies the adaptive method for threshold calculation.
    Uses a Gaussian-weighted sum of the neighborhood to determine the local threshold for each pixel.

cv.THRESH_BINARY:
    Specifies the type of thresholding to apply.
    Sets pixels above the threshold to maxValue (white), others to 0 (black).

Both parameters are used together in cv.adaptiveThreshold() to define how the local threshold is computed and how the image is binarized.
"""
     ellipseKernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(7,7))
     crossKernel = cv.getStructuringElement(cv.MORPH_CROSS,(7,7))
     imgGauss = cv.adaptiveThreshold(img, maxValue, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, blockSize, offset)
     imgGauss = cv.GaussianBlur(imgGauss,(7,7),2)
     plt.imshow(imgGauss,cmap = 'gray')
     plt.title('Gaussian adaptive')


     kernel = np.ones((7,7),np.uint8)
     erosion = cv.erode(imgGauss,ellipseKernel,iterations = 1)
     plt.subplot(242)
     plt.imshow(erosion,cmap = 'gray')
     plt.title('Erosion')

     dilation = cv.dilate(imgGauss,ellipseKernel,iterations = 1)
     plt.subplot(243)
     plt.imshow(dilation,cmap = 'gray')
     plt.title('Dilation')


     morphtypes = [cv.MORPH_OPEN, cv.MORPH_CLOSE, cv.MORPH_GRADIENT, cv.MORPH_TOPHAT, cv.MORPH_BLACKHAT]
     morphnames = ['open', 'close', 'gradient', 'tophat', 'blackhat']
     for i in range(len(morphtypes)):
          plt.subplot(2,4,i+4)
          imgMorph = cv.morphologyEx(imgGauss,morphtypes[i],ellipseKernel)
          plt.imshow(imgMorph,cmap = 'gray')
          plt.title(morphnames[i])    
     plt.show()


     


'''Histogram Backprojection: Histogram of a region to identify 
the parts of the image with similar histogram distribution

1. Object Trakcing
2. Image Detection


a. Compute the histogram of region and normalise 
Calculate histogram of main image
Find the probability of each pixel in the image 


The histBackprojection function performs a histogram backprojection to 
identify regions in an image that are similar 
in color to a pre-defined region of interest. Here's a breakdown of the code:
'''

def histBackprojection():
     root = os.getcwd()
     imgPath = os.path.join(root,'rhm.jpg')
     img = cv.imread(imgPath, cv.IMREAD_GRAYSCALE)
     imgRGB = cv.imread(imgPath)
     #imgRGB = cv.cvtColor(img,cv.COLOR_BGR2RGB)
     hist = cv.calcHist([imgRGB],[0],None,[256],[0,256])
     plt.figure()
     plt.subplot(231)
     plt.imshow(imgRGB,cmap = 'gray')

     imgRegion = imgRGB[0:1500,1200:2800]
     plt.subplot(232)
     plt.imshow(imgRegion,cmap = 'gray')
     imgRegionHSV = cv.cvtColor(imgRegion, cv.COLOR_BGR2HSV)
     imgRegionHist = cv.calcHist([imgRegionHSV],[0,1],None,[180,256],[0,180,0,256])
     cv.normalize(imgRegionHist,imgRegionHist,0,255,cv.NORM_MINMAX)
     imgHSV = cv.cvtColor(imgRGB, cv.COLOR_BGR2HSV)

     out = cv.calcBackProject([imgHSV],[0,1],imgRegionHist,[0,180,0,256],1)
     plt.subplot(233)
     plt.imshow(out)

     #Smoothing out.
     ellipseKernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(15,15))
     cv.filter2D(out,-1,ellipseKernel,out)
     plt.subplot(234)
     plt.imshow(out)

     _,mask = cv.threshold(out,70,255,cv.THRESH_BINARY)
     plt.subplot(235)
     plt.imshow(mask)
     maskAllChannels = cv.merge((mask,mask,mask))
     imgSeg = cv.bitwise_and(imgRGB,maskAllChannels)
     plt.subplot(236)
     plt.imshow(imgSeg)
     plt.show()

#Gradients: First / Second Derivative of the image
#Edge Detection, Feature Extraction and Image Enhancement.
#Convolve with a derivative kernel, SobelX, SobelY  - 1st order Derivative, laplacian 2nd Order Derivative

def imageGradient():
     root = os.getcwd()
     imgPath = os.path.join(root,'rhm.jpg')
     img = cv.imread(imgPath,cv.IMREAD_GRAYSCALE)
     imgRGB = cv.cvtColor(img,cv.COLOR_BGR2RGB)
     plt.figure()
     plt.subplot(231)
     plt.imshow(imgRGB)

     laplacian = cv.Laplacian(img,cv.CV_64F,ksize = 3)
     plt.subplot(232)
     plt.imshow(laplacian,cmap='gray')
     plt.title('laplacian')

     # kx , ky = cv.getDerivKernels(1,0,15)
     # print(ky@kx.T)

     sobelX = cv.Sobel(img,cv.CV_64F,1,0,ksize = 3) #(img, high_resolution, )
     plt.subplot(233)
     plt.imshow(sobelX,cmap='gray')
     plt.title('sobelX')

     sobelY = cv.Sobel(img,cv.CV_64F,0,1,ksize = 3) #(img, high_resolution, )
     plt.subplot(234)
     plt.imshow(sobelY,cmap='gray')
     plt.title('sobelY')


     plt.show()



'''Canny Edge Detection: More robust than gradient menthods, 
1. Object Detection.
2. Image Segmentation.
3. Feature Extraction.


Smooth image with gaussian filter
Sobel and Non-maximum supression to find local suppression.
Double thresholding to keep edges that go above max threshold
'''
def cannyEdge():

     root = os.getcwd()
     imgPath = os.path.join(root,'rhm.jpg')
     img = cv.imread(imgPath,cv.IMREAD_GRAYSCALE)
     #imgRGB = cv.cvtColor(img,cv.COLOR_BGR2RGB)
     scale = 1/5
     heightScale =  int(img.shape[0]*scale)
     widthScale = int(img.shape[1]*scale)
     imgRGB = cv.resize(img,(widthScale,heightScale))
     plt.figure()
     plt.subplot(231)
     plt.imshow(imgRGB)
     imgCanny = cv.Canny(imgRGB,200,255)
     plt.subplot(232)
     plt.imshow(imgCanny,cmap = 'gray')
     plt.show()


'''Image pyramid blending: Laplacian blending and Gaussian blending
Image - Downsize it using Gaussian then expand and subtract the difference is laplacian.
'''
# Gaussian Pyramid (low-frequency features)

# Laplacian Pyramid (high-frequency details)

# Alpha Mask (blending weights)
def imgPyramidBlending():
     root = os.getcwd()
     imgPath = os.path.join(root,'rhm.jpg')
     img = cv.imread(imgPath,cv.IMREAD_GRAYSCALE)
     imgRGB = cv.cvtColor(img,cv.COLOR_BGR2RGB)
     plt.figure()
     #plt.imshow(imgRGB)
     downsamp = imgRGB.copy()
     BGR_Gauss_pyramid = [downsamp]
     plt.subplot(231)
     plt.imshow(downsamp)
     for i in range(5):
          downsamp = cv.pyrDown(downsamp)
          BGR_Gauss_pyramid.append(downsamp)
          plt.subplot(2,3,i+2)
          plt.imshow(downsamp)

     # plt.figure()
     BGR_lapPyramidList = [BGR_Gauss_pyramid[5]]
     for i in range(5):
          upsamp = cv.pyrUp(BGR_Gauss_pyramid[i])
          laplacian = cv.subtract(BGR_Gauss_pyramid[i-1],upsamp)
          BGR_lapPyramidList.append(laplacian)
          plt.subplot(2,3,i)
          plt.imshow(laplacian)
     plt.show()

'''houghLineTransform: Extract lines and segmented lines from images'''
def houghLineTransform():
     
    root = os.getcwd()
    imgPath = os.path.join(root,'rhm.jpg')
    img = cv.imread(imgPath, cv.IMREAD_GRAYSCALE)
    blurred = cv.GaussianBlur(img, (5,5), 0)
    canny = cv.Canny(blurred, 50,180)
    edges = cv.convertScaleAbs(canny)
    
    plt.figure()
     
    plt.subplot(231)
    plt.imshow(canny)
    #imgCanny = cv.Canny(canny,200,255)
    distResol = 0.5
    angleResol = np.pi/60
    thresh = 300
    lines = cv.HoughLines(edges,distResol,angleResol,thresh)
    k =3000
    for line in lines:
        rho,theta = line[0]
        dhat = np.array([[np.cos(theta),np.sin(theta)]])
        d = rho*dhat
        lhat = np.array([[-np.sin(theta),np.cos(theta)]])
        p1 = d + k*lhat
        p2 = d - k*lhat
        p1 = p1.astype(int)
        p2 = p2.astype(int)
        
        cv.line(edges, (p1[0][0],p1[0][1]), (p2[0][0],p2[0][1]), (255,255,255), 10)

    plt.subplot(232)
    plt.imshow(edges)


   
    plt.show()

'''Feature Detection/Corner Detection: Calibration patterns, 
Harris corner detection - Maximize the fucntion that 
gives the most change in all directions'''

def harrisCorner():
     root = os.getcwd()
     imgPath = os.path.join(root,'rhm.jpg')
     img = cv.imread(imgPath)
     imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
     imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
     imgGray = np.float32(imgGray)
     plt.figure()
     plt.subplot(231)
     plt.imshow(imgGray,cmap = 'gray')
     blockSize =5 #Size of the neighbourhood around the pixel
     sobelSize = 3 #aperture parameter used to calculate gradients.
     k = 0.04
     harris = cv.cornerHarris(imgGray,2,3,0.04)
     plt.subplot(232)
     plt.imshow(harris)

     plt.show()


'''SIFT - Scale Invariant Feature Transform: Invariant to scale, 
Robust to variations, Feature detection. Finding blobs by using LoG and DoG
laplaccian of Gaussian and Difference of Gaussian.

Keypoint Localization: Remove points with low contrast
Find orientation.
Keypoint descriptor: L2 distance and neighbours
'''

def sift():
     root = os.getcwd()
     imgPath = os.path.join(root,'rhm.jpg')
     img = cv.imread(imgPath)
     imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
     imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
     sift = cv.SIFT_create()
     keypoints = sift.detect(imgGray,None)
     imgGray = cv.drawKeypoints(imgGray,keypoints,imgGray,flags = cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
     plt.figure()
     plt.subplot(231)
     plt.imshow(imgRGB)
     plt.subplot(232)
     plt.imshow(imgGray)
     plt.show()

     





if __name__ == "__main__":
    #bgrChannelGrayScale()
    #grayScale()
    morphTans()