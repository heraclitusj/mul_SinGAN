import sys
sys.path.append('/home/aistudio/external-libraries')

import torch
from config import get_arguments
from SinGAN.manipulate import *
from SinGAN.training import *
import SinGAN.functions as functions



if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--input_dir', help='input image dir', default='Input/Images')
    #parser.add_argument('--input_name', help='input image name', required=True)
    parser.add_argument('--mode', help='task to be done', default='train')
    opt = parser.parse_args()
    opt = functions.post_config(opt)
    Gs = []
    Zs = []
    reals = []
    NoiseAmp = []
    dir2save, dir2save1, dir2save2 = functions.generate_dir2save(opt)

    if (os.path.exists(dir2save1) or os.path.exists(dir2save2)):
        print('trained model already exist')
    else:
        try:
            #os.makedirs(dir2save)
            os.makedirs(dir2save1)
            os.makedirs(dir2save2)
        except OSError:
            pass
        real1, real2 = functions.read_image(opt) #np2torch real真实图片集 1*3*250*250
        #real = torch.cat((real1,real2), 0) #2*3*250*250
        for i in range(opt.num_images): #2
            functions.adjust_scales2image(real1, opt) #两张图片大小得相同
            functions.adjust_scales2image(real2, opt) #两张图片大小得相同
            print(opt.stop_scale)
        train(opt, Gs, Zs, reals, NoiseAmp)
        SinGAN_generate(Gs,Zs,reals, NoiseAmp,opt)
