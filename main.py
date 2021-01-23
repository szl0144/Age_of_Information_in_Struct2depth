import time
import pdb
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker
import cv2
import shutil
import os
from imp import reload
from matplotlib.pyplot import MultipleLocator
from params import args
import main
from torch.utils.data import DataLoader, Dataset, random_split

from DispNetS import DispNetS     # Depth Estimation using DispNets Module
from PoseExpNet import PoseExpNet # The pose network (returns the pose as a six-element vector)


# This part is a reference for you to change the learning rate (Step size for each iteration)
#   0-100  1e-4
# 100-200  2e-5
# 200-end  1e-5


def get_lr(epoch, learning_rate):
    if epoch >= 30 and epoch < 40:
        lr = learning_rate / 2
    elif epoch >= 40:
        lr = learning_rate / 4
    else:
        lr = learning_rate
    return lr


# Learn from the break point，Need to load the previous parameters such as model, optimizer，epoch，best_valid_loss
# In this code, we temporarily don't use it
#save_path = './ckpt/monodepth_epoch_%s.pth' % epoch
#torch.save(save(model, optimizer, epoch, best_valid_loss), save_path)

def save(model, optimizer, epoch, best_valid_loss):
    save_dict = dict()
    
    save_dict['model'] = model.state_dict()
    save_dict['optimizer'] = optimizer.state_dict()
    save_dict['epoch'] = epoch
    save_dict['best_valid_loss'] = best_valid_loss
    
    return save_dict


def load(dict_path, model, optimizer):
    adict = torch.load(dict_path)
    
    model.load_state_dict(adict['model'])
    
    # https://github.com/pytorch/pytorch/issues/2830
    optimizer.load_state_dict( adict['optimizer'] )
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()
            
    epoch = adict['epoch']
    best_valid_loss = adict['best_valid_loss']
    
    return model, optimizer, epoch, best_valid_loss



def set_lr(optimizer, learning_rate):
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate



def main(T,AoI):
    import data_loader
    ds_path = './dataset' 
    ds = data_loader.SequenceFolder()
    ds.init(ds_path, AoI,0,3,4)
    train_size = int(0.9 * len(ds))   # 方便的按比例分割数据集
    valid_size = len(ds) - train_size

    train_dataset, valid_dataset = random_split(ds, [train_size, valid_size]) # 

    #print(args.batchsize)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batchsize, 
                                  shuffle=True, num_workers=4)
    
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batchsize, 
                                  shuffle=True, num_workers=4)

    #print(len(train_dataloader))
    print("train_size:", train_size)
    print("valid_size:", valid_size)
    seq_length = args.seq_length
    num_scales = 4
    
    torch.manual_seed(0)

    device = args.device
    
    disp_net = DispNetS().to(device) 
    disp_net.init_weights()

    pose_exp_net = PoseExpNet(nb_ref_imgs=seq_length-1, output_exp=False).to(device)
    pose_exp_net.init_weights()
    
    args_lr = args.learning_rate
    optim_params = [
        {'params': disp_net.parameters(), 'lr': args_lr},
        {'params': pose_exp_net.parameters(), 'lr': args_lr}
    ]

    args_momentum = 0.9
    args_beta = 0.999
    args_weight_decay = 0

    optimizer = torch.optim.Adam(optim_params,
                                 betas=(args_momentum, args_beta),
                                 weight_decay = args_weight_decay
                                 )


    start_epoch = 0

    # continue_train = 1  # Whether to learn from break point  
    # if continue_train:  # If learn from break point

    #     model, optimizer, epoch = load(args.previous_ckpt_path, model, optimizer)
    #     start_epoch = epoch + 1
    #     model = model.to(device)
    #     set_lr(optimizer, get_lr(start_epoch, args.learning_rate))
    #     print("\n")
    #     print("load previous checkpoint successfully!")
    #     print("start_epoch:", start_epoch)
    #     print("\n")
    # else:  # Learn from beginning
    #     model = model.to(device)
    #     print("\n")
    #     print("train from scrach!")
    #     print("start_epoch:", start_epoch)
    #     print("\n")

    # cudnn.benchmark = True
    
    best_loss = float('Inf')
    optimal_epoch = 0
    optimal_valid_epoch = 0
    best_loss = float('Inf')
    valid_list = []
    loss_list = []
    valid_loss = 0.0
    best_valid_loss = float('Inf')
    args_epochs = 100

    print("============================== Trainning start")
    # for epoch in range(args_epochs):
    for epoch in range(start_epoch, args_epochs):  # 
        
        
        disp_net.train()
        pose_exp_net.train()
        
        c_time = time.time()
        # Start a epoch

        running_loss = 0.0

        for loader_idx, (image_stack, image_stack_norm, intrinsic_mat, _) in enumerate(train_dataloader):
            #pdb.set_trace()
            image_stack = [img.to(device) for img in image_stack]
            image_stack_norm = [img.to(device) for img in image_stack_norm]
            intrinsic_mat = intrinsic_mat.to(device) # 1 4 3 3
            disp = {}
            depth = {}
            depth_upsampled = {}
            
            for seq_i in range(seq_length):
                multiscale_disps_i, _ = disp_net(image_stack[seq_i])
                # [1,1,128,416], [1,1,64,208],[1,1,32,104],[1,1,16,52]

                # if seq_i == 1:
                #     dd = multiscale_disps_i[0]
                #     dd = dd.detach().cpu().numpy()
                #     np.save( "./rst/" + str(loader_idx) + ".npy", dd)
                
                multiscale_depths_i = [1.0 / d for d in multiscale_disps_i]
                disp[seq_i] = multiscale_disps_i
                depth[seq_i] = multiscale_depths_i
                
                depth_upsampled[seq_i] = []
                
                for s in range(num_scales):
                    depth_upsampled[seq_i].append( nn.functional.interpolate(multiscale_depths_i[s],
                                   size=[128, 416], mode='bilinear', align_corners=True) )
                    
            egomotion = pose_exp_net(image_stack_norm[2], [ image_stack_norm[0], image_stack_norm[1] ])  #change from midle to last
            # torch.Size([1, 2, 6])

            # build loss======================================
            from loss_func import calc_total_loss

            total_loss, reconstr_loss, smooth_loss, ssim_loss = \
            calc_total_loss(image_stack, disp, depth, depth_upsampled, egomotion, intrinsic_mat)
            # total loss  ================================

            if loader_idx % (200/args.batchsize) == 0:
                print("idx: %4d reconstr: %.5f  smooth: %.5f  ssim: %.5f  total: %.5f" % \
                    (loader_idx, reconstr_loss, smooth_loss, ssim_loss, total_loss) )

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()
        running_loss /= (train_size/args.batchsize)
        loss_list.append(running_loss)
        if running_loss < best_loss:
            best_loss = running_loss
            optimal_epoch = epoch
            #print("* best loss:", best_loss )

            torch.save(disp_net.state_dict(),     './disp_net_best.pth' )
            torch.save(pose_exp_net.state_dict(), './pose_exp_net_best.pth' )

        for loader_idx_val, (image_stack_val, image_stack_norm_val, intrinsic_mat_val, _) in enumerate(valid_dataloader):
            
            image_stack_val = [img.to(device) for img in image_stack_val]
            image_stack_norm_val = [img.to(device) for img in image_stack_norm_val]
            intrinsic_mat_val = intrinsic_mat_val.to(device) # 1 4 3 3
            
            disp_val = {}
            depth_val = {}
            depth_upsampled_val = {}
            
            for seq_i in range(seq_length):
                multiscale_disps_val_i, _ = disp_net(image_stack_val[seq_i])
                # [1,1,128,416], [1,1,64,208],[1,1,32,104],[1,1,16,52]                
                multiscale_depths_val_i = [1.0 / d for d in multiscale_disps_val_i]
                disp_val[seq_i] = multiscale_disps_val_i
                depth_val[seq_i] = multiscale_depths_val_i
                depth_upsampled_val[seq_i] = []
        
                for s in range(num_scales):
                    depth_upsampled_val[seq_i].append( nn.functional.interpolate(multiscale_depths_val_i[s],
                                   size=[128, 416], mode='bilinear', align_corners=True) )
                    
            egomotion_val = pose_exp_net(image_stack_norm_val[2], [ image_stack_norm_val[0], image_stack_norm_val[1] ]) #change from middle to last
            # torch.Size([1, 2, 6])

            # build loss======================================
            from loss_func import calc_total_loss

            total_loss_val, reconstr_loss_val, smooth_loss_val, ssim_loss_val = \
            calc_total_loss(image_stack_val, disp_val, depth_val, depth_upsampled_val, egomotion_val, intrinsic_mat_val)
            #pdb.set_trace() 
            # total loss  ================================
            valid_loss += total_loss_val.item()
        valid_loss /= (valid_size/args.batchsize)
        valid_list.append(valid_loss)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            optimal_valid_epoch = epoch
        #############epoch ends
    print("============================== Training End")
    print ('time:', round(time.time() - c_time, 3), 's',
        'Best training loss:', best_loss,
        'Optimal training epoch is:', optimal_epoch,
        'Best validation loss:', best_valid_loss,
        'Optimal validation epoch is:', optimal_valid_epoch,
        'AOI', AoI)
    valid_list.sort()       
    '''
    x = np.arange(0,args_epochs)
    plt.plot(x,loss_list,'r--',label='Training Loss') 
    plt.plot(x,valid_list,'g--',label='Validation Loss')
    plt.title('Training Loss vs Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')   
    plt.ylim(0,1)
    plt.xlim(0,110)
    plt.xticks(range(len(loss_list)))
    x_major_locator=MultipleLocator(5)
    ax=plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.legend()
    plt.show()
    '''
    
    fw = open("loss_new.txt",'a')
    fw.write('{} {} {}\n'.format(valid_list[0],best_loss,T)) 

    fw.close()  
    reload(data_loader)

def split():
    shutil.rmtree('./dataset/video1')
    os.mkdir('./dataset/video1')
    shutil.rmtree('./dataset/video2')
    os.mkdir('./dataset/video2')
    i = 0
    j=0
    frame_count = 0
    frame_count_2 = 0

    cap = cv2.VideoCapture("./video/3.mp4")
    while True:
        ret, frame = cap.read()
        if ret is False:
            break
        frame_count = frame_count + 1

    cap_2 = cv2.VideoCapture("./video/1.mp4")
    while(True):
        ret_2, frame_2_2 = cap_2.read()
        if ret_2 is False:
            break
        frame_count_2 = frame_count_2 + 1
    print('video_1 frame number is: ',frame_count)   
    print('video_2 frame number is: ',frame_count_2)


    cap = cv2.VideoCapture("./video/3.mp4")
    while True:
        i += 1
        ret, frame = cap.read()
        if i == frame_count+1:
            break
            #print("i:", i)
        if i%5 == 0:
          resized_frame = cv2.resize(frame, (416, 128), interpolation=cv2.INTER_AREA)
          cv2.imwrite("./dataset/video1/"  + str('%04d'%i) + ".jpg", resized_frame)
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

    all_folds = os.listdir('./dataset/video1/')
    print('Video1 sampled frame number is %d'%len(all_folds))
    if int(major_ver)  < 3 :
        fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
        print ("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
    else :
        fps = cap.get(cv2.CAP_PROP_FPS)
        print ("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))


    cap_2 = cv2.VideoCapture("./video/1.mp4")
    while True:
        j += 1
        ret_2, frame_2 = cap_2.read()
        if  j == frame_count_2+1:
            break
        #print("j:", j)
        if j%5 == 0:
          resized_frame_2 = cv2.resize(frame_2, (416, 128), interpolation=cv2.INTER_AREA)
          cv2.imwrite("./dataset/video2/"  + str('%04d'%j) + ".jpg", resized_frame_2)
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    all_folds_2 = os.listdir('./dataset/video2/')
    print('Video2 sampled frame number is %d'%len(all_folds_2))
    if int(major_ver)  < 3 :
        fps_2 = cap_2.get(cv2.cv.CV_CAP_PROP_FPS)
        print ("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps_2))
    else :
        fps_2 = cap_2.get(cv2.CAP_PROP_FPS)
        print ("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps_2))
    cap.release()
    cap_2.release()
    cv2.destroyAllWindows() 
    time_gap = 1/fps_2
    return time_gap*5

   


if __name__ == '__main__':
    #os.remove("loss_new.txt") #delete this command
    time_gap = split()
    for AoI in range(7,14): # Training from t+0 to t+20 # Grace (15,20)
      main(time_gap,AoI)


