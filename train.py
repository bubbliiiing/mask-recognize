import cv2
import numpy as np
from keras import backend as K
from keras.applications.imagenet_utils import preprocess_input
from keras.callbacks import (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
                             TensorBoard)
from keras.optimizers import Adam
from keras.utils import get_file, np_utils
from PIL import Image

from net.mobilenet import MobileNet
from utils.utils import get_random_data

K.set_image_dim_ordering('tf')

BASE_WEIGHT_PATH = ('https://github.com/fchollet/deep-learning-models/releases/download/v0.6/')

def letterbox_image(image, size):
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image

def generate_arrays_from_file(lines,batch_size,train):
    # 获取总长度
    n = len(lines)
    i = 0
    while 1:
        X_train = []
        Y_train = []
        # 获取一个batch_size大小的数据
        for b in range(batch_size):
            if i==0:
                np.random.shuffle(lines)

            name = lines[i].split(';')[0]
            #------------------------------#
            #   从文件中读取图像
            #------------------------------#
            img = Image.open("./data/image/train/" + name)
            if train == True:
                img = np.array(get_random_data(img,[HEIGHT,WIDTH]), dtype = np.float64)
            else:
                img = np.array(letterbox_image(img,[WIDTH,HEIGHT]), dtype = np.float64)

            X_train.append(img)
            Y_train.append(lines[i].split(';')[1])
            #------------------------------#
            #   读完一个周期后重新开始
            #------------------------------#
            i = (i+1) % n

        #------------------------------#
        #   处理图像
        #------------------------------#
        X_train = preprocess_input(np.array(X_train))
        Y_train = np_utils.to_categorical(np.array(Y_train),num_classes= NUM_CLASSES)   
        yield (X_train, Y_train)


if __name__ == "__main__":
    HEIGHT = 224
    WIDTH = 224
    NUM_CLASSES = 2
    
    #------------------------------#
    #   训练好的模型保存的位置
    #------------------------------#
    log_dir = "./logs/"

    #------------------------------#
    #   建立MobileNet模型
    #------------------------------#
    model = MobileNet(input_shape=[HEIGHT,WIDTH,3],classes=NUM_CLASSES)
    
    #---------------------------------------------------------------------#
    #   这一步是获得主干特征提取网络的权重、使用的是迁移学习的思想
    #   如果下载过慢，可以复制连接到迅雷进行下载。
    #   之后将权值复制到目录下，根据路径进行载入。
    #   如：
    #   weights_path = "xxxxx.h5"
    #   model.load_weights(weights_path,by_name=True,skip_mismatch=True)
    #---------------------------------------------------------------------#
    model_name = 'mobilenet_1_0_224_tf_no_top.h5'
    weight_path = BASE_WEIGHT_PATH + model_name
    weights_path = get_file(model_name, weight_path, cache_subdir='models')
    model.load_weights(weights_path, by_name=True)

    #-------------------------------------------------------------------------------#
    #   训练参数的设置
    #   checkpoint用于设置权值保存的细节，period用于修改多少epoch保存一次
    #   reduce_lr用于设置学习率下降的方式
    #   early_stopping用于设定早停，val_loss多次不下降自动结束训练，表示模型基本收敛
    #-------------------------------------------------------------------------------#
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                monitor='val_loss', save_weights_only=True, save_best_only=False, period=2)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    #------------------------------#
    #   打开数据集对应的txt
    #------------------------------#
    with open("./data/train.txt","r") as f:
        lines = f.readlines()
    #------------------------------#
    #   打乱的数据更有利于训练
    #   90%用于训练，10%用于估计。
    #------------------------------#
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*0.1)
    num_train = len(lines) - num_val

    #-------------------------------------------------------------------------------#
    #   这里使用的是迁移学习的思想，主干部分提取出来的特征是通用的
    #   所以我们可以不训练主干部分先，因此训练部分分为两步，分别是冻结训练和解冻训练
    #   冻结训练是不训练主干的，解冻训练是训练主干的。
    #   由于训练的特征层变多，解冻后所需显存变大
    #-------------------------------------------------------------------------------#
    trainable_layer = 80
    for i in range(trainable_layer):
        model.layers[i].trainable = False
    print('freeze the first {} layers of total {} layers.'.format(trainable_layer, len(model.layers)))

    if True:
        lr = 1e-3
        batch_size = 8
        model.compile(loss = 'categorical_crossentropy',
                optimizer = Adam(lr=lr),
                metrics = ['accuracy'])

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(generate_arrays_from_file(lines[:num_train], batch_size, True),
                steps_per_epoch=max(1, num_train//batch_size),
                validation_data=generate_arrays_from_file(lines[num_train:], batch_size, False),
                validation_steps=max(1, num_val//batch_size),
                epochs=50,
                initial_epoch=0,
                callbacks=[checkpoint, reduce_lr, early_stopping])

    for i in range(len(model.layers)):
        model.layers[i].trainable = True

    if True:
        lr = 1e-4
        batch_size = 8
        model.compile(loss = 'categorical_crossentropy',
                optimizer = Adam(lr=lr),
                metrics = ['accuracy'])

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(generate_arrays_from_file(lines[:num_train], batch_size, True),
                steps_per_epoch=max(1, num_train//batch_size),
                validation_data=generate_arrays_from_file(lines[num_train:], batch_size, False),
                validation_steps=max(1, num_val//batch_size),
                epochs=100,
                initial_epoch=50,
                callbacks=[checkpoint, reduce_lr, early_stopping])
        model.save_weights(log_dir+'last_one.h5')

