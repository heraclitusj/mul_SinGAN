# mul_SinGAN
one to many natural images expansion

首先对要扩展的图片进行训练，默认多图为两图，两图大小需要完全一致，将待训练图片置于Input/Images目录下。
python main_train.py --num_images 2 --input_name1 image1.png --input_name2 image2.png

训练好后可进行任意尺度的随机扩展，结果保存于自动生成的Output文件夹中。
python random_samples.py --input_name image1.png --input_name1 image1.png --input_name2 image2.png --mode random_samples_arbitrary_sizes --scale_h 2 --scale_v 1
其中scale_h，scale_v分别为水平，竖直方向上图片的扩展比。
