import numpy as np
import matplotlib.pyplot as plt
# import scipy
import os
from darfix.core.dataset import Dataset
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as patches
import cv2

'''
Description
    Decode edf file obtained from ESRF to darfix dataset

Input parameter
    file_dir: the directory where the edf files are saved
    saved_dir: the directory where the returned files are saved

Return
    dataset: Darfix dataset type files
'''
def import_dataset(file_dir, saved_dir):
    filenames = []
    print(f"There are {len(os.listdir(file_dir))} files in the directory") # Print out number of files in the directory
    
    for file in os.listdir(file_dir):
        if file.endswith(".edf"):
            filenames.append(file)
            # count_edf += 1
    # required to ensure messed up order will still load all files
    for single_file in filenames:
        if single_file.endswith("00.edf"):
            first_filename = single_file
    print("First filename is %s."%first_filename)
    print(f"The size of the resulting dataset is {len(filenames)}")
    dataset = Dataset(_dir = saved_dir, first_filename = file_dir + '/'+first_filename, in_memory = True)
    return dataset

'''
Description 
    Take log 2 of the intensity value of rockinglayer (x,y,rockingstep) over rockingsteps

Input Parameters
    rockinglayer: the 3D numpy rockinglayer (x,y,rockingsteps)

Return
    log2(intensity) 3D numpy rockinglayer

'''
def Log2Rockinglayer(rockinglayer):
    log2_rockinglayer = np.zeros(rockinglayer.shape)
    # take log 2 
    for i in range(rockinglayer.shape[-1]):
        log2_rockinglayer[:,:,i][:,:] = np.log2(rockinglayer[:,:,i][:,:])
    # remove base intensity
    print("Finish taking log 2")
    return log2_rockinglayer
'''
Description
    Get the center of mass in terms of rockingstep

Input Parameters
    3D numpy rockinglayer array (x,y,rockingstep)

Return
    2D center of mass colormap (x,y)
'''
def rockingstep_COM_colormap(rockinglayer):
    # remove base intensity
    rockinglayer -= np.percentile(rockinglayer, 1)
    dim1, dim2 = rockinglayer.shape[:2]
    colormap = np.zeros(([dim1, dim2]))
    sum = np.sum(rockinglayer, axis = 2)
    for index in range(rockinglayer.shape[-1]):
        colormap += rockinglayer[:,:,index] * index
    colormap = cv2.blur((colormap/sum),(9,9))
    return colormap

'''
* Need to optimize this, use K-Mean
Description:
    Correct the crystallographic offset by using thresholding and contour sizing

Input Parameters:
    colormap: the colormap result obtained from rockingstep_COM_colormap function
    thres_val: threshold value for the largest crystallographic offset
    mode: 1 or 0 to optimize the offset region

Return
    
'''
def shift_correction(rockinglayer, colormap, thres_val, mode = 1):
    if mode == 1:
        thres_colormap = np.zeros(colormap.shape) + 1
        thres_colormap[colormap > thres_val] = 0
    elif mode == 0:
        thres_colormap = np.zeros(colormap.shape)
        thres_colormap[colormap > thres_val] = 1
    thres_img = np.uint8(thres_colormap)
    area_val = []
    contours, _ = cv2.findContours(thres_img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    seg_colormap = np.zeros(colormap.shape)

    for c in range(len(contours)):
        area_val.append(cv2.contourArea(contours[c]))

    min_area = np.percentile(area_val, 99.5)

    for index in range(len(area_val)):
        if area_val[index] > min_area:
            cont_img = cv2.drawContours(seg_colormap, contours, index, color = (255,255,255), thickness = -1)
    
    mask = np.where(cont_img > 0)
    corrected_rockinglayer = np.copy(rockinglayer)
    corrected_rockinglayer[mask[0],mask[1],:] = np.roll(rockinglayer[mask[0],mask[1],:], shift = 1, axis = -1)
    print(f"Shape of corrected rockinglayer:{corrected_rockinglayer.shape}")
    
    return corrected_rockinglayer

'''
Description
    Plot a single image with a patch 

Input Parameters
    image_array: 3 dimensional array [x, y, rocking frame]
    rocking_frame: the designated rocking frame
    roi: the roi value where the patch will be plotted [x1, x2, y1, y2]
    title: title of the image
    color: color of the patch box, default as orange
    max_colorbar: maximum value on the colorbar, default as 700
    min_colorbar: minimum value on the colorbar, default as 0

Return
    none
'''
def plot_roi_patch_image(image_array, rocking_frame, roi, title, color = 'red', max_colorbar = 700, min_colorbar = 0):
    # Debug checking
    print(f'Image array shape is: {image_array.shape}')
    print(f'ROI value is {roi}')
    
    # Plot figure
    fig, ax = plt.subplots(1,1,figsize = (5,5))
    # if len(image_array.shape) == 2:
    #     im = ax.imshow(image_array, vmin = min_colorbar, vmax = max_colorbar, cmap = 'jet')
    # else:
    im = ax.imshow(image_array[:,:,rocking_frame],vmin = min_colorbar, vmax = max_colorbar, cmap = 'jet')
    ax.set_xlabel("Pixel")
    ax.set_ylabel("Pixel")
    ax.set_title(title)
    
    # define the patch for ROI value
    if type(roi[0]) == type(1):
        patch_roi = patches.Rectangle((roi[2],roi[0]),np.abs(roi[3]-roi[2]),np.abs(roi[1]-roi[0]),linewidth = 2, edgecolor= color, facecolor = 'none')
        ax.add_patch(patch_roi)
    else:
        for index in range(len(roi)):
            print(f'Add {index+1} patch')
            patch_roi = patches.Rectangle((roi[index][2],roi[index][0]),np.abs(roi[index][3]-roi[index][2]),np.abs(roi[index][1]-roi[index][0]),linewidth = 2, edgecolor= color[index], facecolor = 'none')
            ax.add_patch(patch_roi)
    

'''
Description
    Return the cropped image array with input roi value

Input Parameters
    original_img_arr: the original 2D image array to be cropped, should be np array
    roi: region of interest to be cropped [x1,x2,y1,y2]

Return
    cropped image array (2D numpy)
'''
def get_img_roi_array(original_img_arr, roi):
    region = np.empty((roi[1]-roi[0],roi[3]-roi[2],original_img_arr.shape[-1]))
    for i in range(original_img_arr.shape[-1]):
        region[:,:,i] = np.copy(original_img_arr[roi[0]:roi[1],roi[2]:roi[3],i])
    return region

'''
Description
    Convert input darfix dataset to 3D numpy array

Input Parameters
    dataset: darfix dataset type

Return
    img_arr: 3D numpy array [x,y,rocking frame]
    
'''

def Dataset2Numpy(dataset):
    sh = dataset.__dict__['_data'].shape
    img_arr = np.empty((sh[-2], sh[-1],sh[0]))
    print(f"The size of image array is {img_arr.shape}")
    for i in range(img_arr.shape[-1]):
         img_arr[:,:,i] = dataset.get_data(i)
    return img_arr

'''
Description
    Get the mean intensity of each rocking frame

Input Parameters
    array: numpy array for calculating mean intensity [x,y,rockingframe]

Return
     mean_intensity: list of the size of rocking frames    
'''
def get_mean_intensity_list(array):
    mean_intensity = np.zeros((array.shape[-1]))
    for i in range(array.shape[-1]):
        mean_intensity[i] = np.mean(array[:,:,i])
    return mean_intensity

'''
Description
    Unit normalize (not maximum-normalize) an image array

Input parameter
    array: numpy array to be normalize, no dimension limit

Return
    array: normalized numpy array
'''
def get_normalized_intensity_list(list):
    list = (list - np.min(list)) # subtract min value
    list_norm = np.sqrt(np.dot(list,list)) # compute norm
    list /= list_norm # divide intensity by its norm
    print(f"The inner product of the list is {np.dot(list,list)}")
    return list

'''
Description 
    Remove the component of removed_arr from target_arr

Input parameter
    target_arr: target numpy array [x, y, rockingframe]
    remove_arr: normalized mean intensity list 

Return
    result_arr: orthogonalized target array from remove_arr
'''
def remove_component(target_arr, remove_arr):
    result_arr_coef = np.zeros((target_arr.shape[0], target_arr.shape[1]))
    result_arr = np.zeros((target_arr.shape))
    for i in range(target_arr.shape[0]):
        for j in range(target_arr.shape[1]):
            result_arr_coef[i,j] = np.inner(target_arr[i,j,:], remove_arr)
            result_arr[i,j,:] = target_arr[i,j,:] - result_arr_coef[i,j] * remove_arr[:]
    return result_arr

'''
Description
    Apply Gram-Schmidt orthogonalization method on a set of arrays

Input parameter
    arrays: list of 3D numpy arrays (x,y,rockingsteps), numpy arrays should be done with log base (increase the feature) and corrected with crystallography orientation

Return
    arrays: normalized list of 2D numpy arrays (intensity, rockingstep) after Gram-Schmidt
'''
def Gram_Schmidt_list(arrays):
    return_list = []

    for index in range(len(arrays)):
        mean_list = get_mean_intensity_list(arrays[index])
        print(f"In {index+1}th array")
        for j in range(index):
            print(f"{j+1}th array is being removed...")
            mean_list = mean_list - np.dot(return_list[j], mean_list)* return_list[j]

        if np.array_equal(mean_list, np.zeros(mean_list.shape)):
            raise np.linalg.LinAlgError("The column vectors are not linearly independent")

        mean_norm_list = mean_list / np.sqrt(np.dot(mean_list,mean_list))

        return_list.append(mean_norm_list)
        print(f"{index+1}th array is taken mean intensity and normalized\n")
    return return_list 

'''
Description
    Project rockinglayer to GS components

Input Parameter
    rockinglayer: the rockinglayer we are projected from
    GScomponents: Gram-Schmidt components for decomposition

Return
    rockinglayer_SB
    rockinglayer_WBminus
    rockinglayer_WBplus
'''
def projectGS(rockinglayer, GScomponents):
    rockinglayer_SB = np.zeros((rockinglayer.shape[:2]))
    rockinglayer_WBminus = np.zeros((rockinglayer.shape[:2]))
    rockinglayer_WBplus = np.zeros((rockinglayer.shape[:2]))


    for i in range(rockinglayer.shape[0]):
        for j in range(rockinglayer.shape[1]):
            rockinglayer_SB[i,j] = np.sum(np.multiply(GScomponents[0][:], rockinglayer[i,j,:]))
            rockinglayer_WBminus[i,j] = np.sum(np.multiply(GScomponents[1][:], rockinglayer[i,j,:]))
            rockinglayer_WBplus[i,j] = np.sum(np.multiply(GScomponents[2][:], rockinglayer[i,j,:]))
    
    return rockinglayer_SB, rockinglayer_WBminus, rockinglayer_WBplus