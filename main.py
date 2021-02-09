import random
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tensorflow.keras.callbacks import *
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model


def reduce_clasess(mask):
    labels = [('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0)),
              ('ego vehicle', 1, 255, 'void', 0, False, True, (0, 0, 0)),
              ('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
              ('out of roi', 3, 255, 'void', 0, False, True, (0, 0, 0)),
              ('static', 4, 255, 'void', 0, False, True, (0, 0, 0)),
              ('dynamic', 5, 255, 'void', 0, False, True, (111, 74, 0)),
              ('ground', 6, 255, 'void', 0, False, True, (81, 0, 81)),
              ('road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),
              ('sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232)),
              ('parking', 9, 255, 'flat', 1, False, True, (250, 170, 160)),
              ('rail track', 10, 255, 'flat', 1, False, True, (230, 150, 140)),
              ('building', 11, 2, 'construction', 2, False, False, (70, 70, 70)),
              ('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
              ('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
              ('guard rail', 14, 255, 'construction', 2, False, True, (180, 165, 180)),
              ('bridge', 15, 255, 'construction', 2, False, True, (150, 100, 100)),
              ('tunnel', 16, 255, 'construction', 2, False, True, (150, 120, 90)),
              ('pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
              ('polegroup', 18, 255, 'object', 3, False, True, (153, 153, 153)),
              ('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),
              ('traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)),
              ('vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)),
              ('terrain', 22, 9, 'nature', 4, False, False, (152, 251, 152)),
              ('sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)),
              ('person', 24, 11, 'human', 6, True, False, (220, 20, 60)),
              ('rider', 25, 12, 'human', 6, True, False, (255, 0, 0)),
              ('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
              ('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
              ('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
              ('caravan', 29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
              ('trailer', 30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
              ('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
              ('motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
              ('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32))]
    new_lables = defaultdict(list)
    [new_lables[label[3]].append(label[1]) for label in labels];
    mapping = dict(zip(range(len(new_lables.keys())), new_lables.values()))
    n_classes = len(mapping)
    mask = np.squeeze(mask)
    for i in mapping:
        indices = np.any([mask == j for j in mapping[i]], axis=0)
        mask[indices] = i
    return mask, n_classes


def expand_label(mask, n_classes=8):
    mask_expand = np.zeros((*mask.shape, n_classes), dtype="uint8")
    for i in range(n_classes):
        mask_expand[:, :, :, i] = mask == i
    return mask_expand


def data_Generator(data_path, batch_size, validation_split, subset):
    image_datagen = ImageDataGenerator(rescale=1 / 255., validation_split=validation_split)
    image_generator = image_datagen.flow_from_directory(f"{data_path}/images", class_mode=None, batch_size=batch_size,
                                                        subset=subset, seed=1)
    # print(f"image_generator.samples: {image_generator.samples}")
    data_Generator.samples = image_generator.samples
    mask_datagen = ImageDataGenerator(validation_split=validation_split)
    mask_generator = mask_datagen.flow_from_directory(f"{data_path}/masks", class_mode=None, batch_size=batch_size,
                                                      color_mode="grayscale", subset=subset, seed=1)
    train_generator = zip(image_generator, mask_generator)
    for (img, mask) in train_generator:
        mask, n_classes = reduce_clasess(mask)
        mask = expand_label(mask, n_classes)
        yield (img, mask)


def testGenerator(data_path, batch_size):
    test_datagen = ImageDataGenerator(rescale=1 / 255.)
    test_generator = test_datagen.flow_from_directory(f"{data_path}/images", class_mode=None, batch_size=batch_size)
    return test_generator


def conv_block(tensor, nfilters, size=3, padding='same', initializer="he_normal"):
    x = Conv2D(filters=nfilters, kernel_size=(size, size), padding=padding, kernel_initializer=initializer)(tensor)
    # x = BatchNormalization()(x)
    # x = Activation("relu")(x)
    x = Conv2D(filters=nfilters, kernel_size=(size, size), padding=padding, kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


def deconv_block(tensor, residual, nfilters, size=3, padding='same', strides=(2, 2)):
    y = Conv2DTranspose(nfilters, kernel_size=(size, size), strides=strides, padding=padding)(tensor)
    y = concatenate([y, residual], axis=3)
    y = conv_block(y, nfilters)
    return y


def Unet(img_height, img_width, n_classes=3, filters=64):
    # down
    input_layer = Input(shape=(img_height, img_width, 3), name='image_input')
    conv1 = conv_block(input_layer, nfilters=filters)
    conv1_out = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = conv_block(conv1_out, nfilters=filters * 2)
    conv2_out = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = conv_block(conv2_out, nfilters=filters * 4)
    conv3_out = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = conv_block(conv3_out, nfilters=filters * 8)
    conv4_out = MaxPooling2D(pool_size=(2, 2))(conv4)
    conv4_out = Dropout(0.5)(conv4_out)
    conv5 = conv_block(conv4_out, nfilters=filters * 16)
    conv5 = Dropout(0.5)(conv5)
    # up
    deconv6 = deconv_block(conv5, residual=conv4, nfilters=filters * 8)
    deconv6 = Dropout(0.5)(deconv6)
    deconv7 = deconv_block(deconv6, residual=conv3, nfilters=filters * 4)
    deconv7 = Dropout(0.5)(deconv7)
    deconv8 = deconv_block(deconv7, residual=conv2, nfilters=filters * 2)
    deconv9 = deconv_block(deconv8, residual=conv1, nfilters=filters)
    # output
    output_layer = Conv2D(filters=n_classes, kernel_size=(1, 1))(deconv9)
    output_layer = BatchNormalization()(output_layer)
    output_layer = Activation('softmax')(output_layer)
    model = Model(inputs=input_layer, outputs=output_layer, name='Unet')
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def IoU(y_testi, y_predi):
    ## mean Intersection over Union
    ## Mean IoU = TP/(FN + TP + FP)
    IoUs = []
    n_classes = int(np.max(y_testi)) + 1
    for c in range(n_classes):
        TP = np.sum((y_testi == c) & (y_predi == c))
        FP = np.sum((y_testi != c) & (y_predi == c))
        FN = np.sum((y_testi == c) & (y_predi != c))
        IoU = TP / float(TP + FP + FN)
        print("class {:02.0f}: #TP={:6.0f}, #FP={:6.0f}, #FN={:5.0f}, IoU={:4.3f}".format(c, TP, FP, FN, IoU))
        IoUs.append(IoU)
    mIoU = np.mean(IoUs)
    print("____________________________________________________")
    print("Mean IoU: {:4.3f}".format(mIoU))


def give_color_to_seg_img(seg, n_classes):
    if len(seg.shape) == 2:
        seg = np.dstack([seg] * 3)
    seg_img = np.zeros_like(seg).astype('float')
    colors = sns.color_palette("hls", n_classes)
    for c in range(n_classes):
        segc = (seg == c)
        seg_img += segc * (colors[c])
    return seg_img


def visualize_pred(X_test, y_testi, y_predi):
    n_classes = 10
    plt.figure(figsize=(10, 30))
    for i, v in enumerate(random.sample(range(X_test.shape[0]), k=3)):
        img = (X_test[v] + 1) * (255.0 / 2)
        seg = y_predi[v]
        segtest = y_testi[v]

        plt.subplot(3, 3, 1 + 3 * i)
        plt.imshow(img / 255.0)
        plt.title("original")

        plt.subplot(3, 3, 2 + 3 * i)
        plt.imshow(give_color_to_seg_img(seg, n_classes))
        plt.title("predicted class")

        plt.subplot(3, 3, 3 + 3 * i)
        plt.imshow(give_color_to_seg_img(segtest, n_classes))
        plt.title("true class")
        plt.show()  # plt.savefig("result/predictions")


if __name__ == '__main__':
    batch_size = 16
    validation_split = 0.2
    img_height = img_width = 256
    filters = 64
    n_classes = 8
    train_generator = data_Generator(data_path="data/train", batch_size=batch_size,
                                     validation_split=validation_split, subset='training')
    next(train_generator);
    steps_per_epoch = data_Generator.samples // batch_size
    valid_generator = data_Generator(data_path="data/train", batch_size=batch_size,
                                     validation_split=validation_split, subset='validation')
    next(valid_generator);
    validation_steps = data_Generator.samples // batch_size
    test_generator = data_Generator(data_path="data/test", batch_size=batch_size, validation_split=None,
                                    subset=None)
    next(test_generator);
    steps = data_Generator.samples // batch_size
    model = Unet(img_height, img_width, n_classes=n_classes, filters=64)
    plot_model(model)
    
    modelcheckpoint = ModelCheckpoint(mode='max', filepath='models-best.h5', monitor='val_accuracy',
                                      save_best_only='True', verbose=1)
    earlyStopping = EarlyStopping(mode='max', monitor='val_accuracy', patience=10, verbose=1)
    history = model.fit(train_generator, steps_per_epoch=149, epochs=100, validation_data=valid_generator,
                        validation_steps=30, callbacks=[modelcheckpoint, earlyStopping])


    model = load_model('models-best.h5')
    model.evaluate(test_generator, batch_size=batch_size, steps=steps)

    X_test, y_test = next(test_generator)
    y_pred = model.predict(X_test)
    y_predi = np.argmax(y_pred, axis=-1)
    y_testi = np.argmax(y_test, axis=-1)

    ## Visualize the model performance
    visualize_pred(X_test, y_testi, y_predi)

    ## Calculate intersection over union for each segmentation class
    IoU(y_testi, y_predi)
