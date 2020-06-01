import sys
sys.path.append('/home/aistudio/external-libraries')

import SinGAN.functions as functions
import SinGAN.models as models
import os
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import math
import matplotlib.pyplot as plt
from SinGAN.imresize import imresize

def train(opt,Gs,Zs,reals,NoiseAmp):
    real_1, real_2 = functions.read_image(opt) #1*3*250*250
    in_s = 0
    scale_num = 0
    real1 = imresize(real_1,opt.scale1,opt)
    real2 = imresize(real_2,opt.scale1,opt)
    reals1 = []
    reals2 = []
    reals1 = functions.creat_reals_pyramid(real1,reals1,opt)
    reals2 = functions.creat_reals_pyramid(real2,reals2,opt)
    #reals1 = np.array(reals1)
    #reals2 = np.array(reals2)
    #print("hahaha")
    #print(reals1.shape)
    #reals = [reals1, reals2]
    #reals = np.array(reals)
    #print("hahahahah")
    #print(reals.shape)
    nfc_prev = 0

    while scale_num<opt.stop_scale+1:
        opt.nfc = min(opt.nfc_init * pow(2, math.floor(scale_num / 4)), 128) #特征数
        opt.min_nfc = min(opt.min_nfc_init * pow(2, math.floor(scale_num / 4)), 128)

        opt.out,opt.out1, opt.out2 = functions.generate_dir2save(opt) #保存位置
        opt.outf1 = '%s/%d' % (opt.out1,scale_num) #文件名
        opt.outf2 = '%s/%d' % (opt.out2,scale_num) #文件名
        try:
            os.makedirs(opt.outf1)
            os.makedirs(opt.outf2)
        except OSError:
                pass

        #plt.imsave('%s/in.png' %  (opt.out_), functions.convert_image_np(real), vmin=0, vmax=1)
        #plt.imsave('%s/original.png' %  (opt.out_), functions.convert_image_np(real_), vmin=0, vmax=1)
        plt.imsave('%s/real_scale.png' %  (opt.outf1), functions.convert_image_np(reals1[scale_num]), vmin=0, vmax=1) #保存真实图片
        plt.imsave('%s/real_scale.png' %  (opt.outf2), functions.convert_image_np(reals2[scale_num]), vmin=0, vmax=1) #保存真实图片

        D_curr,G_curr = init_models(opt) #初始化生成器和鉴别器
        if (nfc_prev == opt.nfc): #除第一次之外都载入上个模型 
            G_curr.load_state_dict(torch.load('%s/%d/netG.pth' % (opt.out1,scale_num-1)))
            D_curr.load_state_dict(torch.load('%s/%d/netD.pth' % (opt.out1,scale_num-1)))

        z_curr,in_s,G_curr = train_single_scale(D_curr,G_curr,reals1,reals2,Gs,Zs,in_s,NoiseAmp,opt) #单次训练
        print('z_curr')
        print(z_curr.shape)

        G_curr = functions.reset_grads(G_curr,False) #requires_grad=False，冻结梯度
        G_curr.eval() #固定参数
        D_curr = functions.reset_grads(D_curr,False)
        D_curr.eval()
        

        Gs.append(G_curr)
        Zs.append(z_curr)
        NoiseAmp.append(opt.noise_amp) #0.1

        torch.save(Zs, '%s/Zs.pth' % (opt.out1)) 
        torch.save(Gs, '%s/Gs.pth' % (opt.out1)) #所有的生成器
        torch.save(reals1, '%s/reals.pth' % (opt.out1)) #所有的真实图片（不同尺度）
        torch.save(NoiseAmp, '%s/NoiseAmp.pth' % (opt.out1)) #所有噪声
        torch.save(Zs, '%s/Zs.pth' % (opt.out2)) 
        torch.save(Gs, '%s/Gs.pth' % (opt.out2)) #所有的生成器
        torch.save(reals2, '%s/reals.pth' % (opt.out2)) #所有的真实图片（不同尺度）
        torch.save(NoiseAmp, '%s/NoiseAmp.pth' % (opt.out2)) #所有噪声

        scale_num+=1
        nfc_prev = opt.nfc
        del D_curr,G_curr #删除变量
    return



def train_single_scale(netD,netG,reals1,reals2,Gs,Zs,in_s,NoiseAmp,opt,centers=None):

    real1 = reals1[len(Gs)] #第1张图片第n个scale的真实图片 1*3*x*x
    real2 = reals2[len(Gs)] #第2张图片第n个scale的真实图片
    real = torch.cat((real1,real2), dim=0) #2*3*x*x
    print("reals1:")
    for i in range(opt.stop_scale+1):
        print(reals1[i].shape)
    print("real1:")
    print(real1.shape)
    opt.nzx = real1.shape[2]#+(opt.ker_size-1)*(opt.num_layer) 高
    opt.nzy = real1.shape[3]#+(opt.ker_size-1)*(opt.num_layer) 宽
    opt.receptive_field = opt.ker_size + ((opt.ker_size-1)*(opt.num_layer-1))*opt.stride #ker_size=3 num_layer=5 stride=1
    pad_noise = int(((opt.ker_size - 1) * opt.num_layer) / 2)
    pad_image = int(((opt.ker_size - 1) * opt.num_layer) / 2)
    #if opt.mode == 'animation_train':
    #    opt.nzx = real.shape[2]+(opt.ker_size-1)*(opt.num_layer)
    #    opt.nzy = real.shape[3]+(opt.ker_size-1)*(opt.num_layer)
    #    pad_noise = 0
    m_noise = nn.ZeroPad2d(int(pad_noise))
    m_image = nn.ZeroPad2d(int(pad_image))

    alpha = opt.alpha #重建损失的系数

    fixed_noise = functions.generate_noise([opt.nc_z,opt.nzx,opt.nzy],device=opt.device) #与该scale图像相同大小的噪声
    z_opt = torch.full(fixed_noise.shape, 0, device=opt.device) #噪声初始化 
    z_opt = m_noise(z_opt) #10

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999))
    schedulerD = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerD,milestones=[1600],gamma=opt.gamma) #1600次时降低学习率
    schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerG,milestones=[1600],gamma=opt.gamma)

    errD2plot = []
    errG2plot = []
    D_real2plot = []
    D_fake2plot = []
    z_opt2plot = []

    for epoch in range(opt.niter): #2000
        if (Gs == []) & (opt.mode != 'SR_train'): #第一次训练
            z_opt = functions.generate_noise([1,opt.nzx,opt.nzy], device=opt.device) #高斯噪声
            z_opt = m_noise(z_opt.expand(opt.num_images,3,opt.nzx,opt.nzy)) #用于计算重建损失的第一层噪声
            noise_ = functions.generate_noise([1,opt.nzx,opt.nzy], device=opt.device)
            noise_ = m_noise(noise_.expand(opt.num_images,3,opt.nzx,opt.nzy)) #n张图片
        else:
            z_opt = functions.generate_noise([1,opt.nzx,opt.nzy], device=opt.device) #高斯噪声
            z_opt = m_noise(z_opt.expand(opt.num_images,3,opt.nzx,opt.nzy)) #用于计算重建损失的第一层噪声
            noise_ = functions.generate_noise([opt.nc_z,opt.nzx,opt.nzy], device=opt.device)
            noise_ = m_noise(noise_)

        ############################
        # (1) Update D network: maximize D(x) + D(G(z))
        ###########################
        for j in range(opt.Dsteps): #3
            # train with real
            netD.zero_grad()
            
            labels_onehot = np.array([[1.0,0.0],[0.0,1.0]]) #2*2 
#           labels_onehot[np.arange(num_img),label.numpy()]=1 #将真的label的值改为1
#             img=img.view(num_img,-1)
#             img=np.concatenate((img.numpy(),labels_onehot))
#             img=torch.from_numpy(img)
#            img=Variable(img).cuda()
            real_label = Variable(torch.from_numpy(labels_onehot).float()).cuda()#真实label为1
            fake_label = Variable(torch.zeros(opt.num_images,2)).cuda()#假的label为0
            #real = torch.tensor(real, dtype=torch.float)
           
            #real_label_ = real_label.unsqueeze(2).unsqueeze(3)
            #real_label_ = real_label_.repeat(1,1,opt.nzx,opt.nzy)
            #real_ = torch.cat((real,real_label_), dim=1) ##2*5*x*x
            real_out = netD(real).to(opt.device) 
            real_out = torch.sigmoid(real_out) #将结果映射到（0，1）中
            output = torch.zeros((opt.num_images,2)).cuda()
            output[0,0] = real_out[0,0].mean()
            output[0,1] = real_out[0,1].mean()
            output[1,0] = real_out[1,0].mean()
            output[1,1] = real_out[1,1].mean()
            
            #D_real_map = output.detach()
            #criterion=nn.MSELoss()
            #errD_real = criterion(output,real_label)
            #errD_real = -output.mean()#-a 
            errD_real1 = -output[0,0].mean()
            errD_real2 = -output[1,1].mean()
            errD_real = errD_real1 + errD_real2
            errD_real.backward(retain_graph=True)
            #D_x = -errD_real.item()

            # train with fake
            if (j==0) & (epoch == 0):
                if (Gs == []) & (opt.mode != 'SR_train'): #第0个scale
                    prev = torch.full([opt.num_images,opt.nc_z,opt.nzx,opt.nzy], 0, device=opt.device) #全0
                    #in_s = prev
                    in_s = torch.full([1,opt.nc_z,opt.nzx,opt.nzy], 0, device=opt.device) #全0
                    real_prev = m_image(prev) #real模式的上个scale传来的图片
                    random_prev = m_image(prev) #random模式的上个scale传来的图片
                    z_prev = torch.full([opt.num_images,opt.nc_z,opt.nzx,opt.nzy], 0, device=opt.device)
                    z_real_prev = m_noise(z_prev)
                    z_random_prev = m_noise(z_prev)
                    opt.noise_amp = 1
#                elif opt.mode == 'SR_train':
#                    z_prev = in_s
#                    criterion = nn.MSELoss()
#                    RMSE = torch.sqrt(criterion(real, z_prev))
#                    opt.noise_amp = opt.noise_amp_init * RMSE
#                    z_prev = m_image(z_prev)
#                    prev = z_prev
                else:
                    real_prev1 = draw_concat(Gs,Zs,reals1,NoiseAmp,in_s,'rand',m_noise,m_image,'real',1,opt)
                    #prev1 = m_image(prev1)
                    real_prev2 = draw_concat(Gs,Zs,reals2,NoiseAmp,in_s,'rand',m_noise,m_image,'real',2,opt)
                    #prev2 = m_image(prev2)
                    real_prev = torch.cat((real_prev1,real_prev2),dim=0)
                    #print('prev:')
                    #print(prev.shape)
                    #prev = draw_concat(Gs,Zs,reals,NoiseAmp,in_s,'rand',m_noise,m_image,opt) #每一层都是随机噪声
                    real_prev = m_image(real_prev)
                    z_real_prev1 = draw_concat(Gs,Zs,reals1,NoiseAmp,in_s,'rec',m_noise,m_image,'real',1,opt)
                    z_real_prev2 = draw_concat(Gs,Zs,reals2,NoiseAmp,in_s,'rec',m_noise,m_image,'real',2,opt)
                    z_real_prev = torch.cat((z_real_prev1,z_real_prev2),dim=0)
                    #z_prev = draw_concat(Gs,Zs,reals,NoiseAmp,in_s,'rec',m_noise,m_image,opt) #仅第一层为随机噪声
                    criterion = nn.MSELoss()
                    RMSE = torch.sqrt(criterion(real, z_real_prev)) #开根号
                    opt.noise_amp = opt.noise_amp_init*RMSE #每个scale第一轮改变噪声系数
                    z_real_prev = m_image(z_real_prev)
                    
                    random_prev1 = draw_concat(Gs,Zs,reals1,NoiseAmp,in_s,'rand',m_noise,m_image,'random',1,opt)
                    #prev1 = m_image(prev1)
                    random_prev2 = draw_concat(Gs,Zs,reals2,NoiseAmp,in_s,'rand',m_noise,m_image,'random',2,opt)
                    #prev2 = m_image(prev2)
                    random_prev = torch.cat((random_prev1,random_prev2),dim=0)
                    #print('prev:')
                    #print(prev.shape)
                    #prev = draw_concat(Gs,Zs,reals,NoiseAmp,in_s,'rand',m_noise,m_image,opt) #每一层都是随机噪声
                    random_prev = m_image(random_prev)
                    z_random_prev1 = draw_concat(Gs,Zs,reals1,NoiseAmp,in_s,'rec',m_noise,m_image,'random',1,opt)
                    z_random_prev2 = draw_concat(Gs,Zs,reals2,NoiseAmp,in_s,'rec',m_noise,m_image,'random',2,opt)
                    z_random_prev = torch.cat((z_random_prev1,z_random_prev2),dim=0)
                    #z_prev = draw_concat(Gs,Zs,reals,NoiseAmp,in_s,'rec',m_noise,m_image,opt) #仅第一层为随机噪声
                    criterion = nn.MSELoss()
                    RMSE = torch.sqrt(criterion(real, z_random_prev)) #开根号
                    opt.noise_amp = opt.noise_amp_init*RMSE #每个scale第一轮改变噪声系数
                    z_random_prev = m_image(z_random_prev)
            else:
                real_prev1 = draw_concat(Gs,Zs,reals1,NoiseAmp,in_s,'rand',m_noise,m_image,'real',1,opt)
                real_prev1 = m_image(real_prev1)
                #print('prev1')
                #print(prev1.shape)
                real_prev2 = draw_concat(Gs,Zs,reals2,NoiseAmp,in_s,'rand',m_noise,m_image,'real',2,opt)
                real_prev2 = m_image(real_prev2)
                real_prev = torch.cat((real_prev1,real_prev2),dim=0)
                
                random_prev1 = draw_concat(Gs,Zs,reals1,NoiseAmp,in_s,'rand',m_noise,m_image,'random',1,opt)
                random_prev1 = m_image(random_prev1)
                #print('prev1')
                #print(prev1.shape)
                random_prev2 = draw_concat(Gs,Zs,reals2,NoiseAmp,in_s,'rand',m_noise,m_image,'random',2,opt)
                random_prev2 = m_image(random_prev2)
                random_prev = torch.cat((random_prev1,random_prev2),dim=0)

#            if opt.mode == 'paint_train':
#                prev = functions.quant2centers(prev,centers)
#                plt.imsave('%s/prev.png' % (opt.outf), functions.convert_image_np(prev), vmin=0, vmax=1)

            #print('noise_')
            #print(noise_.shape)
            if (Gs == []) & (opt.mode != 'SR_train'):
                noise_real = noise_ #第一层纯噪声
                noise_random = noise_
            else:
                noise_real = opt.noise_amp*noise_+real_prev #最后一个scale的生成器输入
                noise_random = opt.noise_amp*noise_+random_prev #最后一个scale的生成器输入
                
            #print('noise')
            #print(noise.shape)
            #print('prev')
            #print(prev.shape)
                

            random_label = torch.randn((opt.num_images,2))
            random_label_ = random_label.unsqueeze(2).unsqueeze(3)
            random_label_ = random_label_.repeat(1,1,opt.nzx+2*int(pad_noise),opt.nzy+2*int(pad_noise)).cuda()
            l_noise = torch.cat((noise_random,random_label_),dim=1).cuda() #带标签的噪声 2*5*x*x
            #l_prev = torch.cat((random_prev,random_label_),dim=1).cuda() #带标签的上个scale传来的图像
            #print('l_prev')
            #print((l_noise.detach()+l_prev).shape)
            fake_d = netG(l_noise.detach(),random_prev) #最终的fake图片 2*3*x*x 
            fake_out = netD(fake_d).to(opt.device) #2*2*x*x
            fake_out = torch.sigmoid(fake_out)
            output = torch.zeros((opt.num_images,2)).cuda()
            output[0,0] = fake_out[0,0].mean()
            output[0,1] = fake_out[0,1].mean()
            output[1,0] = fake_out[1,0].mean()
            output[1,1] = fake_out[1,1].mean()
            #errD_fake = criterion(output, fake_label)
            #errD_fake = output.mean() #越低越好
            errD_fake1 = output[0,0].mean()
            errD_fake2 = output[1,1].mean()
            errD_fake = errD_fake1 + errD_fake2
            errD_fake.backward(retain_graph=True)
            #D_G_z = output.mean().item()
            fake1 = fake_d[0].unsqueeze(0)
            fake2 = fake_d[1].unsqueeze(0)

            gradient_penalty1 = functions.calc_gradient_penalty(netD, real1, fake1, opt.lambda_grad, opt.device)
            gradient_penalty2 = functions.calc_gradient_penalty(netD, real2, fake2, opt.lambda_grad, opt.device)
            #gradient_penalty = (gradient_penalty1 + gradient_penalty2) / 2
            gradient_penalty1.backward()
            gradient_penalty2.backward()

            #errD = errD_real + errD_fake + gradient_penalty #越小越好
            optimizerD.step()

        #errD2plot.append(errD.detach())

        ############################
        # (2) Update G network: maximize D(G(z))
        ###########################

        for j in range(opt.Gsteps): #3
            netG.zero_grad()
            labels_onehot = np.array([[1.0,0.0],[0.0,1.0]]) #2*2
            real_label = Variable(torch.from_numpy(labels_onehot).float()).cuda()#真实label为1
            real_label_ = real_label.unsqueeze(2).unsqueeze(3)
            real_label_ = real_label_.repeat(1,1,opt.nzx+2*int(pad_noise),opt.nzy+2*int(pad_noise)).cuda()
            l_noise = torch.cat((noise_real,real_label_),dim=1).cuda() #带标签的噪声 2*5*x*x
            #l_prev = torch.cat((real_prev,real_label_),dim=1).cuda() #带标签的上个scale传来的图像
            fake_g = netG(l_noise.detach(),real_prev) #最终的fake图片 2*3*x*x 
            fake1 = fake_g[0].unsqueeze(0)
            fake2 = fake_g[1].unsqueeze(0)
            fake_real_out = netD(fake_g)
            output = torch.zeros((opt.num_images,2)).cuda()
            fake_real_out = torch.sigmoid(fake_real_out)
            output[0,0] = fake_real_out[0,0].mean()
            output[0,1] = fake_real_out[0,1].mean()
            output[1,0] = fake_real_out[1,0].mean()
            output[1,1] = fake_real_out[1,1].mean()
            #criterion=nn.MSELoss()
            #D_fake_map = output.detach()
            #errG = -output.mean() #假图片得分越小越好
            #errG = criterion(output, real_label)
            errG1 = -output[0,0].mean()
            errG2 = -output[1,1].mean()
            errG = errG1 + errG2
            errG.backward(retain_graph=True)
            if alpha!=0:
                loss = nn.MSELoss()
                if opt.mode == 'paint_train':
                    z_prev = functions.quant2centers(z_prev, centers)
                    plt.imsave('%s/z_prev.png' % (opt.outf), functions.convert_image_np(z_prev), vmin=0, vmax=1)
                Z_opt = opt.noise_amp*z_opt+z_real_prev
                z_label = Variable(torch.from_numpy(labels_onehot).float()).cuda()#真实label为1
                z_label_ = z_label.unsqueeze(2).unsqueeze(3)
                z_label_ = z_label_.repeat(1,1,opt.nzx+2*int(pad_noise),opt.nzy+2*int(pad_noise)).cuda()
                Z_opt = torch.cat((Z_opt,z_label_),dim=1).cuda() #带标签的噪声 2*5*x*x
                rec_loss = alpha*loss(netG(Z_opt.detach(),z_real_prev),real)
                rec_loss.backward(retain_graph=True)
                rec_loss = rec_loss.detach()
            else:
                Z_opt = z_opt
                rec_loss = 0

            optimizerG.step()

        #errG2plot.append(errG.detach()+rec_loss)
        #D_real2plot.append(D_x)
        #D_fake2plot.append(D_G_z)
        #z_opt2plot.append(rec_loss)

        if epoch % 25 == 0 or epoch == (opt.niter-1):
            print('scale %d:[%d/%d]' % (len(Gs), epoch, opt.niter))

        if epoch % 500 == 0 or epoch == (opt.niter-1):
            plt.imsave('%s/fake_sample.png' %  (opt.outf1), functions.convert_image_np(fake1.detach()), vmin=0, vmax=1)
            plt.imsave('%s/G(z_opt).png'    % (opt.outf1),  functions.convert_image_np(netG(Z_opt[0].unsqueeze(0).detach(), z_real_prev[0].unsqueeze(0)).detach()), vmin=0, vmax=1)
            plt.imsave('%s/fake_sample.png' %  (opt.outf2), functions.convert_image_np(fake2.detach()), vmin=0, vmax=1)
            plt.imsave('%s/G(z_opt).png'    % (opt.outf2),  functions.convert_image_np(netG(Z_opt[1].unsqueeze(0).detach(), z_real_prev[1].unsqueeze(0)).detach()), vmin=0, vmax=1)
            #plt.imsave('%s/D_fake.png'   % (opt.outf), functions.convert_image_np(D_fake_map))
            #plt.imsave('%s/D_real.png'   % (opt.outf), functions.convert_image_np(D_real_map))
            #plt.imsave('%s/z_opt.png'    % (opt.outf), functions.convert_image_np(z_opt.detach()), vmin=0, vmax=1)
            #plt.imsave('%s/prev.png'     %  (opt.outf), functions.convert_image_np(prev), vmin=0, vmax=1)
            #plt.imsave('%s/noise.png'    %  (opt.outf), functions.convert_image_np(noise), vmin=0, vmax=1)
            #plt.imsave('%s/z_prev.png'   % (opt.outf), functions.convert_image_np(z_prev), vmin=0, vmax=1)


            torch.save(z_opt, '%s/z_opt.pth' % (opt.outf1))
            torch.save(z_opt, '%s/z_opt.pth' % (opt.outf2))

        schedulerD.step()
        schedulerG.step()

    functions.save_networks(netG,netD,z_opt,opt)
    return z_opt,in_s,netG    

def draw_concat(Gs,Zs,reals,NoiseAmp,in_s,mode,m_noise,m_image,kind,pic,opt):
    G_z = in_s #prev
    #print('in_s')
    #print(in_s.shape)
    #print('reals')
    #for i in range(opt.stop_scale+1):
    #    print(reals[i].shape)
    if len(Gs) > 0:
        if mode == 'rand':
            count = 0
            pad_noise = int(((opt.ker_size-1)*opt.num_layer)/2)
            if opt.mode == 'animation_train':
                pad_noise = 0
            for G,Z_opt,real_curr,real_next,noise_amp in zip(Gs,Zs,reals,reals[1:],NoiseAmp):
                #print('real_curr')
                #print(real_curr.shape)
                #print('real_next')
                #print(real_next.shape)
                if count == 0:
                    z = functions.generate_noise([1, Z_opt.shape[2] - 2 * pad_noise, Z_opt.shape[3] - 2 * pad_noise], device=opt.device)
                    z = z.expand(1, 3, z.shape[2], z.shape[3])
                else:
                    z = functions.generate_noise([opt.nc_z,Z_opt.shape[2] - 2 * pad_noise, Z_opt.shape[3] - 2 * pad_noise], device=opt.device)
                z = m_noise(z)
                G_z = G_z[:,:,0:real_curr.shape[2],0:real_curr.shape[3]]
                #print('G_z')
                #print(G_z.shape)
                G_z = m_image(G_z)
                z_in = noise_amp*z+G_z #1*3*x*x
                #print('z_in')
                #print(z_in.shape)
                if kind == 'real':
                    if pic==1:
                        labels_onehot = np.array([[1.0,0.0]]) #1*2
                    if pic==2:
                        labels_onehot = np.array([[0.0,1.0]]) #1*2
                    real_label = Variable(torch.from_numpy(labels_onehot).float()).cuda()#真实label为1
                    real_label_ = real_label.unsqueeze(2).unsqueeze(3)
                    real_label_ = real_label_.repeat(1,1,z_in.shape[2],z_in.shape[3])
                    z_in = torch.cat((z_in,real_label_),dim=1)
                    #G_z = torch.cat((G_z,real_label_),dim=1)
                if kind == 'random':
                    random_label = torch.randn((1,2)).cuda()
                    random_label_ = random_label.unsqueeze(2).unsqueeze(3)
                    random_label_ = random_label_.repeat(1,1,z_in.shape[2],z_in.shape[3])
                    z_in = torch.cat((z_in,random_label_),dim=1).cuda() #带标签的噪声 1*5*x*x
                    #G_z = torch.cat((G_z,random_label_),dim=1)
                G_z = G(z_in.detach(),G_z)
                G_z = imresize(G_z,1/opt.scale_factor,opt)
                G_z = G_z[:,:,0:real_next.shape[2],0:real_next.shape[3]]
                #print('G_z:')
                #print(G_z.shape)
                count += 1
        if mode == 'rec':
            count = 0
            for G,Z_opt,real_curr,real_next,noise_amp in zip(Gs,Zs,reals,reals[1:],NoiseAmp):
                G_z = G_z[:, :, 0:real_curr.shape[2], 0:real_curr.shape[3]]
                G_z = m_image(G_z)
                if pic==1:
                    Z_opt = Z_opt[0].unsqueeze(0)
                if pic==2:
                    Z_opt = Z_opt[1].unsqueeze(0)
                z_in = noise_amp*Z_opt+G_z
                print('z_in')
                print(z_in.shape)
                if kind == 'real':
                    if pic==1:
                        labels_onehot = np.array([[1.0,0.0]]) #1*2
                    if pic==2:
                        labels_onehot = np.array([[0.0,1.0]]) #1*2:
                    #print('labels_onehot')
                    #print(labels_onehot.shape)
                    real_label = Variable(torch.from_numpy(labels_onehot).float()).cuda()#真实label为1
                    real_label_ = real_label.unsqueeze(2).unsqueeze(3)
                    real_label_ = real_label_.repeat(1,1,z_in.shape[2],z_in.shape[3])
                    #print('real_label_')
                    #print(real_label_.shape)
                    z_in = torch.cat((z_in,real_label_),dim=1).cuda()
                    #G_z = torch.cat((G_z,real_label_),dim=1)
                if kind == 'random':
                    random_label = torch.randn((1,2)).cuda()
                    random_label_ = random_label.unsqueeze(2).unsqueeze(3)
                    random_label_ = random_label_.repeat(1,1,z_in.shape[2],z_in.shape[3])
                    z_in = torch.cat((z_in,random_label_),dim=1).cuda() #带标签的噪声 1*5*x*x
                    #G_z = torch.cat((G_z,random_label_),dim=1)
                G_z = G(z_in.detach(),G_z)
                G_z = imresize(G_z,1/opt.scale_factor,opt)
                G_z = G_z[:,:,0:real_next.shape[2],0:real_next.shape[3]]
                #if count != (len(Gs)-1):
                #    G_z = m_image(G_z)
                count += 1
    return G_z

def train_paint(opt,Gs,Zs,reals,NoiseAmp,centers,paint_inject_scale):
    in_s = torch.full(reals[0].shape, 0, device=opt.device)
    scale_num = 0
    nfc_prev = 0

    while scale_num<opt.stop_scale+1:
        if scale_num!=paint_inject_scale:
            scale_num += 1
            nfc_prev = opt.nfc
            continue
        else:
            opt.nfc = min(opt.nfc_init * pow(2, math.floor(scale_num / 4)), 128)
            opt.min_nfc = min(opt.min_nfc_init * pow(2, math.floor(scale_num / 4)), 128)

            opt.out_ = functions.generate_dir2save(opt)
            opt.outf = '%s/%d' % (opt.out_,scale_num)
            try:
                os.makedirs(opt.outf)
            except OSError:
                    pass

            #plt.imsave('%s/in.png' %  (opt.out_), functions.convert_image_np(real), vmin=0, vmax=1)
            #plt.imsave('%s/original.png' %  (opt.out_), functions.convert_image_np(real_), vmin=0, vmax=1)
            plt.imsave('%s/in_scale.png' %  (opt.outf), functions.convert_image_np(reals[scale_num]), vmin=0, vmax=1)

            D_curr,G_curr = init_models(opt)

            z_curr,in_s,G_curr = train_single_scale(D_curr,G_curr,reals[:scale_num+1],Gs[:scale_num],Zs[:scale_num],in_s,NoiseAmp[:scale_num],opt,centers=centers)

            G_curr = functions.reset_grads(G_curr,False)
            G_curr.eval()
            D_curr = functions.reset_grads(D_curr,False)
            D_curr.eval()

            Gs[scale_num] = G_curr
            Zs[scale_num] = z_curr
            NoiseAmp[scale_num] = opt.noise_amp

            torch.save(Zs, '%s/Zs.pth' % (opt.out_))
            torch.save(Gs, '%s/Gs.pth' % (opt.out_))
            torch.save(reals, '%s/reals.pth' % (opt.out_))
            torch.save(NoiseAmp, '%s/NoiseAmp.pth' % (opt.out_))

            scale_num+=1
            nfc_prev = opt.nfc
        del D_curr,G_curr
    return


def init_models(opt):

    #generator initialization:
    netG = models.GeneratorConcatSkip2CleanAdd(opt).to(opt.device)
    netG.apply(models.weights_init)
    if opt.netG != '':
        netG.load_state_dict(torch.load(opt.netG))
    print(netG)

    #discriminator initialization:
    netD = models.WDiscriminator(opt).to(opt.device)
    netD.apply(models.weights_init)
    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))
    print(netD)

    return netD, netG
