import cv2
from PIL import Image
import numpy as np
#from keras.preprocessing.image import ImageDataGenerator
import glob
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from skimage.transform import rotate 
import os

def load(train='./preprocess/train/'):
	print("loading the dataset..\n")
	x=glob.glob(train+'images/*')
	y=glob.glob(train+'masks/*')
	x.sort()
	y.sort()


	xim=[]
	yim=[]
	for i in x:
		tmp=cv2.imread(i)
		xim.append(tmp)

	for i in y:
		tmp=cv2.imread(i)
		yim.append(tmp)

	xim=np.array(xim)
	yim=np.array(yim)
    print("loading dataset is complete..\n")
	return(xim,yim)
	


def display1(x,xx):
     plt.subplot(211)
     plt.imshow(x)
     plt.subplot(212)
     plt.imshow(xx,cmap='Greys')
     plt.show()

def display2(x,y,xx,yy):
     plt.subplot(221)
     plt.imshow(x)
     plt.subplot(222,cmap='Greys')
     plt.imshow(y)
     plt.subplot(223)
     plt.imshow(xx)
     plt.subplot(224,cmap='Greys')
     plt.imshow(yy)
     plt.show()

def aug(x,y,process,val):
    xx,yy=process(x,y,val)
    return(xx,yy)

##gamma value suitable gamma_max=1.5
def gamma(x,y, gamma_max=None):
    if gamma_max==None:
        return (x,y)

    x=x.astype(np.uint8)
    gamma=np.random.uniform(.75,gamma_max)
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
    for i in np.arange(0, 256)]).astype("uint8")
    xx=cv2.LUT(x, table).astype(np.uint8)
    return (xx,y)

##can use any value btw 0-180
def rot(x,y,r_max=None):
    if r_max==None:
        return (x,y)
    r=np.random.uniform(-r_max,r_max)
    x=rotate(x,r,mode='constant')
    y=rotate(y,r,mode='constant')
    xx=(x*255).astype(np.uint8)
    yy=(y*255).astype(np.uint8)
    # import ipdb; ipdb.set_trace()
    return(xx,yy)

##ideally alpha_max =2 (from >.5) beta_max=10  
def cont(x,y,c_max=None):
    if c_max==None:
        return (x,y)

    alpha_max,beta_max=c_max

    beta=int(np.random.uniform(-beta_max,beta_max))
    alpha=np.random.uniform(.6,alpha_max)

    xx=cv2.addWeighted(x, alpha, x, 0, beta)
    return(xx,y)

##Ideally input h_max=5 for med
def clr_shift(x,y,h_max=None):
    if h_max==None:
        return (x,y)
    x=x.astype(np.uint8)
    hsv=cv2.cvtColor(x,cv2.COLOR_BGR2HSV)

    h_shift=int(np.random.normal(-h_max,h_max))
    h=hsv[:,:,0]
    h_shift=h+h_shift
    h_rot=h_shift+180
    h=h_rot%180
    hsv[:,:,0]=h

    x_shift=cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    return(x_shift,y)

##generator for image and mask
def generate(x,y,op_len,rotation=None,contrast=None,clr_space=None,gamma_max=None):
    xim,yim=shuffle(x,y)    
    value=[rotation,contrast,clr_space,gamma_max]
    print("Augmentation based on:\n \tRotation-{}\n \tContrast-{}\n \tHue_space-{}\n \tGamma-{}".format(rotation,contrast,clr_space,gamma_max))
    process=[rot,cont,clr_shift,gamma]
    ratio=float(len(xim))/op_len
    print('Augmneted data({}) / training data({}) split = {}'.format(op_len,len(xim),ratio))
    while True:
        im=int(np.random.uniform(0,len(xim)))
        x=xim[im]
        y=yim[im]
        if np.random.rand()>.2:
            for i in range(len(process)):
                if np.random.rand()>.2:
                    x,y=aug(x,y,process[i],value[i])
        # in case generator creates smaller sets of images with negligible mask detections
        # if cv2.countNonZero(x)<=250:
        #     continue
        yy=((y>20).astype(np.uint8))*255
        yield(x,yy)

if __name__ == '__main__':
    try:
        os.stat('image.npy')
        xim=np.load('image.npy')
        yim=np.load('mask.npy')
    except:
        xim,yim=load()
        np.save('image.npy',xim)
        np.save('mask.npy',yim)


    aug_folder='./augmented/'

    try:
        os.stat(aug_folder)
    except:
        os.mkdir(aug_folder)
        os.mkdir(aug_folder+'images')
        os.mkdir(aug_folder+'masks')

    aug_len=1000

    count=0
    for x,y in generate(xim,yim,op_len=aug_len,rotation=180,contrast=(2,10),clr_space=3,gamma_max=1.3):
    	print("Generation of data stamped @ %d"%count)
    	cv2.imwrite('./augmented/images/%s.jpg'%str(count).zfill(4),x)
    	cv2.imwrite('./augmented/masks/%s.jpg'%str(count).zfill(4),y)
    	count+=1
    	if count>aug_len:
    		break

    print("+++++++++++++++++++++DATA GENERTATION COMLETE+++++++++++++++++++")
