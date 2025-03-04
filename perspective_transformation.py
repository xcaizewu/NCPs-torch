import cv2
import numpy as np


'''
DEFINE CAMERA PARAMETERS
'''
#normal vector of the ground plane relative to the camera
NORMAL = 0.5*np.array([[0, 1, 0.04 ]])

# intrinsic camera matrix (https://en.wikipedia.org/wiki/Camera_resectioning#Intrinsic_parameters)
INTRINSIC_MATRIX = np.array([[ 942.21001829,    0.        ,  982.06345972],
                             [   0.        ,  942.91483399,  617.04862261],
                             [   0.        ,    0.        ,    1.        ]])
RAW_SIZE = (1208, 1920) #raw image size, used to scale the camera matrix if we try to transform images of different size

'''
Transforms an image to a new camera perspective given a translation
and rotation
Inputs:
    @image: (n,m,3) numpy array representing the input image
    @t:     scalar distance in meters to laterally translate by
    @theta: scalar angle in radians to rotate by
Outputs:
    @out:   (n,m,3) numpy array representing the transformed image
'''
def perspective_transform(image, t, theta):
    translation = np.array([[t, 0, 0]]) #translate only on the x-axis

    c = np.cos(theta)
    s = np.sin(theta)
    R = np.array([[c,0,-s], [0,1,0], [s,0,c]]) #only rotate about the z-axis
    H = R - np.dot(translation.T, NORMAL) # homography matrix H
    K = compute_intrinsic(INTRINSIC_MATRIX, image) #compute the intrinsic matrix K
    # compute K * H * K^{-1}
    KHK = np.matmul(np.matmul(K, H), np.linalg.inv(K))
    # perform the perspective transform
    warped_image = cv2.warpPerspective(image, KHK, (image.shape[1], image.shape[0]), cv2.INTER_NEAREST)
    return warped_image

'''
Rescale a camera intrinsic matrix given a difference image size (ie. through resizing)
'''
def compute_intrinsic(K, img):
    IMG_SCALE_Y = RAW_SIZE[0] / img.shape[0]
    IMG_SCALE_X = RAW_SIZE[1] / img.shape[1]

    IMG_SCALE = np.array([[1./IMG_SCALE_X,           0, 0],
                             [       0, 1./IMG_SCALE_Y, 0],
                             [       0,           0, 1.]])

    scaled_K = np.matmul(IMG_SCALE, K)

    return scaled_K


''' crop image to bounding box
box = [x1,y1,x2,y2]
-------- x2,y2
|          |
|          |
|          |
x1,y1-------
where all coordinates are scaled by the height and width of the image
'''
def crop(image, box):
    (h,w) = image.shape[:2]
    (x1,y1,x2,y2) = [int(round(box[0]*w)), int(round(box[1]*h)),
                     int(round(box[2]*w)), int(round(box[3]*h))]

    return image[y2:y1, x1:x2]

def draw_box(image, box):
    (h,w) = image.shape[:2]
    (x1,y1,x2,y2) = [int(round(box[0]*w)), int(round(box[1]*h)),
                     int(round(box[2]*w)), int(round(box[3]*h))]

    return cv2.rectangle(image,(x1,y2),(x2,y1),(0,255,0),3)

def flip(img):
    return cv2.flip(img, 1)

def flip_all(images):
    for i in range(images.shape[0]):
        images[i] = flip(images[i])
    
    return images

def crop_only(img):
    img=img.astype('float32')
    # patch_box = [500/1920.0, 1000/1208.0, 1420/1920.0, 600/1208.0] #bounding box for the patch
    patch_box = [520/1920.0, 950/1208.0, 1440/1920.0, 590/1208.0] #bounding box for the patch

    patch = crop(img, patch_box)
    # return cv2.resize(patch, (200,60))
    new_img = cv2.resize(patch, (200,78))
    return new_img.astype('uint8')

def crop_all(images):
    img_buffer = np.empty([images.shape[0],78,200,3],dtype=np.uint8)
    for i in range(images.shape[0]):
        img_buffer[i] = crop_only(images[i])
    
    return img_buffer

def transform_and_crop(img,shift,rotate):
    img=img.astype('float32')
    # patch_box = [500/1920.0, 1000/1208.0, 1420/1920.0, 600/1208.0] #bounding box for the patch
    patch_box = [520/1920.0, 950/1208.0, 1440/1920.0, 590/1208.0] #bounding box for the patch

    new_img = perspective_transform(img, shift, rotate)
    patch = crop(new_img, patch_box)
    # return cv2.resize(patch, (200,60))
    return cv2.resize(patch, (200,78))

def correct_inverse_r(inverse_r,t,theta,speed=60):
    recovery_time = 4
    lookahead_distance = recovery_time * speed # lookahead_distance is perpendicular to axis of translation

    correction_rotation = theta / lookahead_distance
    correction_translation = 2*t/(np.square(lookahead_distance) + np.square(t))

    new_inverse_r = inverse_r + correction_rotation + correction_translation
    return new_inverse_r

'''
####################################
############ MAIN CODE #############
####################################
'''

def test():
    #read in a test image
    img = cv2.imread('raw_image2.png')
    (h,w,d) = img.shape
    patch_box = [500/1920.0, 1000/1208.0, 1420/1920.0, 600/1208.0] #bounding box for the patch

    # loop through translations from 1 meter to the left to 1 meter to the right
    print('translation demo')
    for t in np.linspace(-2,2,200):
        new_img = perspective_transform(img, t, 0)
        new_image_with_box = draw_box(new_img, patch_box)
        cv2.imshow('full frame', cv2.resize(new_image_with_box,None, fx=0.4, fy=0.4))

        patch = crop(new_img, patch_box)
        cv2.imshow('patch', cv2.resize(patch, (200,60)))
        cv2.waitKey(30)


    print('rotation demo')
    for theta in np.linspace(-np.pi/10., np.pi/10.,200):
        new_img = perspective_transform(img, 0, theta)
        new_image_with_box = draw_box(new_img, patch_box)
        cv2.imshow('full frame', cv2.resize(new_image_with_box,None, fx=0.4, fy=0.4))

        patch = crop(new_img, patch_box)
        cv2.imshow('patch', cv2.resize(patch, (200,60)))
        cv2.waitKey(30)
