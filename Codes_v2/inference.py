import tensorflow as tf
import os
import numpy as np
import cv2 as cv

from models import H_estimator
from utils import DataLoader, load, save
import constant
import skimage
import glob
tf.compat.v1.disable_eager_execution()

os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = constant.GPU
test_folder = constant.TEST_FOLDER
snapshot_dir =  constant.SNAPSHOT_DIR + '/model.ckpt-500000'
batch_size = constant.TEST_BATCH_SIZE

height, width = 512, 512



# define dataset
with tf.compat.v1.name_scope('dataset'):
    ##########testing###############
    
    test_inputs_clips_tensor = tf.compat.v1.placeholder(shape=[batch_size, height, width, 3 * 2], dtype=tf.float32)
    test_inputs = test_inputs_clips_tensor
    print('test inputs = {}'.format(test_inputs))
    



# depth is not needed in the inference process, 
#we assign "test_depth" arbitrary values such as an all-one map
test_depth = tf.ones_like(test_inputs[...,0:1])
print("test_depth.shape")
print(test_depth.shape)
with tf.compat.v1.variable_scope('generator', reuse=None):
    print('testing = {}'.format(tf.compat.v1.get_variable_scope().name))
    test_warp2_depth, test_mesh, test_warp2_H1, test_warp2_H2, test_warp2_H3, test_one_warp_H1, test_one_warp_H2, test_one_warp_H3,test_H1_mat,test_H2_mat = H_estimator(test_inputs, test_inputs, test_depth)
    


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True      
with tf.compat.v1.Session(config=config) as sess:
    # dataset
    data_loader = DataLoader(test_folder, height, width)

    # initialize weights
    sess.run(tf.compat.v1.global_variables_initializer())
    print('Init global successfully!')

    # tf saver
    saver = tf.compat.v1.train.Saver(var_list=tf.compat.v1.global_variables(), max_to_keep=None)

    restore_var = [v for v in tf.compat.v1.global_variables()]
    loader = tf.compat.v1.train.Saver(var_list=restore_var)

    def inference_func(ckpt):
        print("============")
        print(ckpt)
        load(loader, sess, ckpt)
        print("============")
        length = len(glob.glob(os.path.join(test_folder, 'input1/*.jpg')))
        psnr_list = []
        ssim_list = []

        out_path = "../output/testing/"
        
        if not os.path.exists("../fusion/"):
            os.makedirs("../fusion/")
        if not os.path.exists(out_path+"mask1"):
            os.makedirs(out_path+"mask1")
        if not os.path.exists(out_path+"mask2"):
            os.makedirs(out_path+"mask2")
        if not os.path.exists(out_path+"mask3"):
            os.makedirs(out_path+"mask3")
        if not os.path.exists(out_path+"warp1"):
            os.makedirs(out_path+"warp1")
        if not os.path.exists(out_path+"warp2"):
            os.makedirs(out_path+"warp2")
        if not os.path.exists(out_path+"warp3"):
            os.makedirs(out_path+"warp3")


        for i in range(0, length):
            input_clip = np.expand_dims(data_loader.get_data_clips(i), axis=0)
            #chop input1 left,input2 right 60%
            ratio = 0.8
            chop_input_clip = input_clip.copy()
            input1 = (input_clip[0,...,0:3]+1) * 127.5  
            input2 = cv.resize((input_clip[0,...,3:6]+1) * 127.5,(int(512/(1-ratio)),512))  

            chop_input_clip[0,...,0:3] = cv.resize(chop_input_clip[0,...,int(512*ratio):,0:3],(512,512))
            chop_input_clip[0,...,3:6] = cv.resize(chop_input_clip[0,...,:int(512*(1-ratio)),3:6],(512,512))

            chop_input1 = (chop_input_clip[0,...,0:3]+1) * 127.5
            chop_input2 = (chop_input_clip[0,...,3:6]+1) * 127.5
            
            #Attention: both inputs and outpus are the types of numpy, that is :(preH, warp_gt) and (input_clip,h_clip)
            _, mesh, warp_H1, warp_H2, warp_H3, warp_one_H1, warp_one_H2, warp_one_H3,H1_mat,H2_mat = sess.run([test_warp2_depth, test_mesh, test_warp2_H1, test_warp2_H2, test_warp2_H3, test_one_warp_H1, test_one_warp_H2, test_one_warp_H3,test_H1_mat,test_H2_mat], 
                    feed_dict={test_inputs_clips_tensor: chop_input_clip})
            


            # warp  = warp_H3
            final_warp = (warp_H3+1) * 127.5    
            final_warp = final_warp[0] 
            # warp_one  = warp_one_H3
            final_warp_one = warp_one_H3[0]
            
            # calculate psnr/ssim
            psnr = skimage.metrics.peak_signal_noise_ratio(chop_input1*final_warp_one, final_warp*final_warp_one, data_range=255)
            ssim = skimage.metrics.structural_similarity(chop_input1*final_warp_one, final_warp*final_warp_one, data_range=255, multichannel=True)
            
            # image fusion
            img1 = cv.copyMakeBorder(input1, 0,0,0,512, cv.BORDER_CONSTANT, value=0)
            img2 = final_warp*final_warp_one
            fusion = np.zeros((512,1024,3), np.uint8)
            mask1 = np.ones((512,1024,1), np.uint8)*255
            mask2 = final_warp_one*255

            H1_mat = np.linalg.inv(H1_mat[0])
            H2_mat = np.linalg.inv(H2_mat[0])

            border = np.array([[[0.,0.],[0.,512.],[512./(1-ratio),512.],[512./(1-ratio),0.]]])
            border1 = cv.perspectiveTransform(border,H2_mat)[0]
            xmin,ymin=np.min(border1,axis=0)
            xmax,ymax=np.max(border1,axis=0)
            size = (int(max(xmax,512)),int(max(ymax,512)))
            input2_warp_H1 = cv.warpPerspective(input2, H2_mat, size)
            input2_warp_one_H1 = cv.warpPerspective(np.ones((512,int(512/(1-ratio)),3))*255, H2_mat, size)
            #reszie to original size
            img2 = cv.resize(img2,(int(512*(1-ratio)),512))
            mask2 = cv.resize(mask2,(int(512*(1-ratio)),512))
            input2_warp_H1 = cv.resize(input2_warp_H1,(int(input2_warp_H1.shape[1]*(1-ratio)),input2_warp_H1.shape[0]))
            input2_warp_one_H1 = cv.resize(input2_warp_one_H1,(int(input2_warp_H1.shape[1]*(1-ratio)),input2_warp_H1.shape[0]))

            #padding
            pad_img2 = np.zeros((512,1024,3))
            pad_img2[:,int(512*ratio):int(512*ratio)+img2.shape[1],:] = img2
            pad_mask2 = np.zeros((512,1024,3))
            pad_mask2[:,int(512*ratio):int(512*ratio)+img2.shape[1],:] = mask2
            right = 1024-input2_warp_H1.shape[1]-int(512*ratio)
            if right<0:
                right=0
            input2_warp_H1=cv.copyMakeBorder(input2_warp_H1, 0,0,int(512*ratio),right, cv.BORDER_CONSTANT, value=0)
            input2_warp_one_H1=cv.copyMakeBorder(input2_warp_one_H1, 0,0,int(512*ratio),right, cv.BORDER_CONSTANT, value=0)
            

            # image for other model
            cv.imwrite(out_path+"mask1/"+str(i+1).zfill(6) + ".jpg", mask1)
            cv.imwrite(out_path+"mask2/"+str(i+1).zfill(6) + ".jpg", pad_mask2)
            cv.imwrite(out_path+"mask3/"+str(i+1).zfill(6) + ".jpg", input2_warp_one_H1[:512,:1024,:])
            cv.imwrite(out_path+"warp1/"+str(i+1).zfill(6) + ".jpg", img1)
            cv.imwrite(out_path+"warp2/"+str(i+1).zfill(6) + ".jpg", pad_img2)
            cv.imwrite(out_path+"warp3/"+str(i+1).zfill(6) + ".jpg", input2_warp_H1[:512,:1024,:])
            #better fusion not needed
                #img2[gray2<=1]=img1[gray2<=1]
                #cv.imwrite('img2/'+ str(i+1).zfill(6) + ".jpg", img2)
                #alpha = 0.5
                #fusion = cv.addWeighted(img1, alpha, img2, 1-alpha, 0.0)

            fusion[...,0] = pad_img2[...,0] 
            fusion[...,1] = img1[...,1]*0.5 +  pad_img2[...,1]*0.5
            fusion[...,2] = img1[...,2]
            path = "../fusion/" + str(i+1).zfill(6) + ".jpg"
            cv.imwrite(path, fusion)
            
            
            print('i = {} / {}, psnr = {:.6f}'.format( i+1, length, psnr))
            
            psnr_list.append(psnr)
            ssim_list.append(ssim)
            
            
        print("===================Results Analysis==================")   
        print("psnr")
        psnr_list.sort(reverse = True)
        data_30 = int(length*0.3)
        data_60 = int(length*0.6)
        psnr_list_30 = psnr_list[0 : data_30]
        psnr_list_60 = psnr_list[data_30: data_60]
        psnr_list_100 = psnr_list[data_60: -1]
        print("top 30%", np.mean(psnr_list_30))
        print("top 30~60%", np.mean(psnr_list_60))
        print("top 60~100%", np.mean(psnr_list_100))
        print('average psnr:', np.mean(psnr_list))
        
        ssim_list.sort(reverse = True)
        ssim_list_30 = ssim_list[0 : data_30]
        ssim_list_60 = ssim_list[data_30: data_60]
        ssim_list_100 = ssim_list[data_60: -1]
        print("top 30%", np.mean(ssim_list_30))
        print("top 30~60%", np.mean(ssim_list_60))
        print("top 60~100%", np.mean(ssim_list_100))
        print('average ssim:', np.mean(ssim_list))
        
    inference_func(snapshot_dir)

    






