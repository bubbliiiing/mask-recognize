from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.applications.imagenet_utils import preprocess_input
from keras.utils import np_utils,get_file
from keras.optimizers import Adam
from keras import backend as K
from utils.utils import get_random_data
from net.mobileNet import MobileNet
from PIL import Image
import numpy as np
import cv2
K.set_image_dim_ordering('tf')

BASE_WEIGHT_PATH = ('https://github.com/fchollet/deep-learning-models/'
                    'releases/download/v0.6/')

HEIGHT = 160
WIDTH = 160
NUM_CLASSES = 2

def letterbox_image(image, size):
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (0,0,0))
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
            # 从文件中读取图像
            img = Image.open(r".\data\image\train" + '/' + name)
            if train == True:
                img = np.array(get_random_data(img,[HEIGHT,WIDTH]),dtype = np.float64)
            else:
                img = np.array(letterbox_image(img,[HEIGHT,WIDTH]),dtype = np.float64)
            X_train.append(img)
            Y_train.append(lines[i].split(';')[1])
            # 读完一个周期后重新开始
            i = (i+1) % n
        # 处理图像
        X_train = preprocess_input(np.array(X_train).reshape(-1,HEIGHT,WIDTH,3))
        
        Y_train = np_utils.to_categorical(np.array(Y_train),num_classes= NUM_CLASSES)   
        yield (X_train, Y_train)


if __name__ == "__main__":
    # 模型保存的位置
    log_dir = "./logs/"

    # 打开数据集的txt
    with open(r".\data\train.txt","r") as f:
        lines = f.readlines()

    # 打乱行，这个txt主要用于帮助读取数据来训练
    # 打乱的数据更有利于训练
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)

    # 90%用于训练，10%用于估计。
    num_val = int(len(lines)*0.1)
    num_train = len(lines) - num_val

    # 建立MobileNet模型
    model = MobileNet(input_shape=[HEIGHT,WIDTH,3],classes=NUM_CLASSES)
    
    model_name = 'mobilenet_1_0_224_tf_no_top.h5'
    weight_path = BASE_WEIGHT_PATH + model_name
    weights_path =get_file(model_name,weight_path,cache_subdir='models')
    model.load_weights(weights_path,by_name=True)

    # 保存的方式，3世代保存一次
    checkpoint_period1 = ModelCheckpoint(
                                    log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                    monitor='acc', 
                                    save_weights_only=False, 
                                    save_best_only=True, 
                                    period=3
                                )
    # 学习率下降的方式，acc三次不下降就下降学习率继续训练
    reduce_lr = ReduceLROnPlateau(
                            monitor='acc', 
                            factor=0.5, 
                            patience=3, 
                            verbose=1
                        )
    # 是否需要早停，当val_loss一直不下降的时候意味着模型基本训练完毕，可以停止
    early_stopping = EarlyStopping(
                            monitor='val_loss', 
                            min_delta=0, 
                            patience=10, 
                            verbose=1
                        )
    # 交叉熵
    model.compile(loss = 'categorical_crossentropy',
            optimizer = Adam(lr=1e-3),
            metrics = ['accuracy'])

    # 一次的训练集大小
    batch_size = 8

    # 开始训练
    model.fit_generator(generate_arrays_from_file(lines[:num_train], batch_size, True),
            steps_per_epoch=max(1, num_train//batch_size),
            validation_data=generate_arrays_from_file(lines[num_train:], batch_size, False),
            validation_steps=max(1, num_val//batch_size),
            epochs=10,
            initial_epoch=0,
            callbacks=[checkpoint_period1, reduce_lr])

    model.save_weights(log_dir+'middle_one.h5')

    # 交叉熵
    model.compile(loss = 'categorical_crossentropy',
            optimizer = Adam(lr=1e-4),
            metrics = ['accuracy'])

    # 开始训练
    model.fit_generator(generate_arrays_from_file(lines[:num_train], batch_size, True),
            steps_per_epoch=max(1, num_train//batch_size),
            validation_data=generate_arrays_from_file(lines[num_train:], batch_size, False),
            validation_steps=max(1, num_val//batch_size),
            epochs=20,
            initial_epoch=10,
            callbacks=[checkpoint_period1, reduce_lr])

    model.save_weights(log_dir+'last_one.h5')