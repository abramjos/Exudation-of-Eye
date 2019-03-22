import cv2
import glob
import numpy as np
import os
import argparse


#delta is the additional thickness on sides of image to preserve exudations on edges
global delta
delta=10

def test(im_list,mask_list):
    for i in range(len(im_list)):
        im=cv2.imread(im_list[i])
        mask=cv2.imread(mask_list[i])
        if im.shape!=mask.shape:
            print('for {}, {} image   :::   {} mask '.format(i,im.shape,mask.shape))
        print(im.shape)

    plt.figure()
    plt.subplot(4,2,1) 
    plt.imshow(cv2.imread(lst[0][3]))
    plt.subplot(4,2,2)
    plt.imshow(cv2.imread(lst[1][3]))
    plt.show()

def fetch(im_list,mask_list):
    w_max=0
    contour_list=[]     
    for i in range(len(im_list)):
        image=cv2.imread(im_list[i],0)
        mask=cv2.imread(mask_list[i],0)
        name=im_list[i]

        kernel = np.ones((5,5),np.float32)/25
        image = cv2.blur(image,(4,4))
        image = cv2.filter2D(image,-1,kernel)
        
        ret,thresh = cv2.threshold(image,15,255,cv2.THRESH_BINARY)
        _,contours, hierarchy = cv2.findContours(thresh.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        max_area=cv2.contourArea(contours[0])
        for i in contours:
            area = cv2.contourArea(i)
            if area>=max_area:
                max_area=area
                x,y,w,h = cv2.boundingRect(i)
        print('{}  gives ::: {}'.format(name,str(w)))
        contour_list.append([x,y,w,h])
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
        if w_max<w:
            print('\t overwriting {} by {}'.format(w_max,w))
            w_max=w
        elif w<10:
            plt.imshow(thresh)
            plt.show()
    return(contour_list)


def resize_rgb(image,contour,width):
    x,y,w,h=contour
    image_roi=image[y:y+h,x:x+w,:]
    image_bod=cv2.copyMakeBorder(image_roi,top=int((w-h)/2)+delta, bottom=int((w-h)/2)+delta,left=delta,right=delta,borderType= cv2.BORDER_CONSTANT)
    image_resized=cv2.resize(image_bod,(width,width),interpolation = cv2.INTER_CUBIC)
    return(image_resized)

def resize_mask(image,contour,width):
    x,y,w,h=contour
    image_roi=image[y:y+h,x:x+w]
    image_bod=cv2.copyMakeBorder(image_roi,top=int((w-h)/2)+delta, bottom=int((w-h)/2)+delta,left=delta,right=delta,borderType= cv2.BORDER_CONSTANT)
    image_resized=cv2.resize(image_bod,(width,width),interpolation = cv2.INTER_CUBIC)
    return(image_resized)



if __name__ == '__main__':

    #input dataset can be data/train or data/test
    #based on contour detected, the pixel diameter of smallest contour is taken to normalize all the images and masks 
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input_path',type=str,default='data/train',
                        help='preprocess directory')
    parser.add_argument('-o','--output_path',type=str,default='preprocess/train',
                        help='preprocess directory')
    args = parser.parse_args()
    # import ipdb;ipdb.set_trace()

    im_list=glob.glob(args.input_path+'/images/*')
    mask_list=glob.glob(args.input_path+'/masks/*') 

    im_list.sort()
    mask_list.sort()


    contour_list=fetch(im_list,mask_list)
    contour_list=np.array(contour_list)

    w_list=contour_list[:,2]    
    w_list=np.array(w_list) 



    op_width=[int((np.average(w_list)+w_list.max())/2),w_list.max()]

    width=256

    folder_im=args.output_path+'/images'
    folder_msk=args.output_path+'/masks'


    try:
        os.stat(folder_im)
        os.stat(folder_msk)
    except:
        os.mkdir(args.output_path.split('/')[0])
        os.mkdir(args.output_path)
        os.mkdir(folder_im)
        os.mkdir(folder_msk)
    image=[]
    mask=[]
    for i in range(len(im_list)): 
        im=cv2.imread(im_list[i])
        msk=cv2.imread(mask_list[i])
        im_resize=resize_rgb(im,contour_list[i],width)
        mask_resize=resize_rgb(msk,contour_list[i],width)

        im_loc=im_list[i].replace(args.input_path,args.output_path)
        mask_loc=mask_list[i].replace(args.input_path,args.output_path)
        cv2.imwrite(im_loc,im_resize)
        cv2.imwrite(mask_loc,mask_resize)
        image.append(im_resize)
        mask.append(mask_resize)

    images=np.array(image)
    masks=np.array(mask)

    im_name=args.input_path.split('/')[-1]+'_X.npy'
    mask_name=args.input_path.split('/')[-1]+'_Y.npy'

    np.save(im_name,image)
    np.save(mask_name,mask)
