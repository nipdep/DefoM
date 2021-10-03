#%%

import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
from scipy import io
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
#%%
data = np.load("data\data\image_triples1.npz")
print(len(data))
# %%
fimg_data = data['first_image']
simg_data = data['second_image']
mask_data = data['mask']

# %%
ff_np = np.expand_dims(fimg_data, 0)
sf_np = np.expand_dims(simg_data, 0)
c_np = np.vstack((ff_np, sf_np))
n_np = np.moveaxis(c_np, [0,1,2,3], [3,0,1,2])/65536
n_np.shape
# %%
train_X = tf.data.Dataset.from_tensor_slices(n_np)
mask_y = tf.data.Dataset.from_tensor_slices(np.expand_dims(mask_data, -1))
# %%
train_dt = tf.data.Dataset.zip((train_X, mask_y))
train_dt.element_spec
# %%

def normalize(img, mask):
    # one, zero = tf.ones_like(mask), tf.zeros_like(mask)
    img = img/65536
    # mask = mask/tf.math.reduce_max(mask)
    # mask = tf.where(mask > 0.0, x=one, y=zero)
    return tf.cast(img, dtype=tf.float32), tf.cast(mask, dtype=tf.float32)

def brightness(img1, mask):
    img1 = tf.image.adjust_brightness(img1, 0.1)
#     img2 = tf.image.adjust_brightness(img2, 0.1)
    return img1, mask

def gamma(img1, mask):
    img1 = tf.image.adjust_gamma(img1, 0.1)
#     img2 = tf.image.adjust_gamma(img2, 0.1) 
    return img1, mask

def hue(img1, mask):
    img1 = tf.image.adjust_hue(img1, -0.1)
#     img2 = tf.image.adjust_hue(img2, -0.1)
    return img1, mask

# def crop(img, mask):
#      img = tf.image.central_crop(img, 0.7)
#      img = tf.image.resize(img, (128,128))
#      mask = tf.image.central_crop(mask, 0.7)
#      mask = tf.image.resize(mask, (128,128))
#      mask = tf.cast(mask, tf.uint8)
#      return img, mask

def flip_hori(img1, mask):
    img1 = tf.image.flip_left_right(img1)
#     img2 = tf.image.flip_left_right(img2)
    mask = tf.image.flip_left_right(mask)
    return img1, mask

def flip_vert(img1, mask):
    img1 = tf.image.flip_up_down(img1)
#     img2 = tf.image.flip_up_down(img2)
    mask = tf.image.flip_up_down(mask)
    return img1, mask

def rotate(img1, mask):
    img1 = tf.image.rot90(img1)
#     img2 = tf.image.rot90(img2)
    mask = tf.image.rot90(mask)
    return img1, mask
# %%
# perform augmentation on train data only
# train_dt = train_dt.map(normalize)
a = train_dt.map(brightness)
b = train_dt.map(gamma)
e = train_dt.map(flip_hori)
f = train_dt.map(flip_vert)
g = train_dt.map(rotate)

train_dt = train_dt.concatenate(a)
train_dt = train_dt.concatenate(b)
train_dt = train_dt.concatenate(e)
train_dt = train_dt.concatenate(f)
train_dt = train_dt.concatenate(g) 

# %%
## test image augmentation
sample = train_dt.take(1)
sample_imgs = list(sample.as_numpy_iterator())
input_imgs = sample_imgs[0][0]
mask_img = sample_imgs[0][1]
print("input image shape : ", input_imgs.shape, "mask image shape : ", mask_img.shape)
plt.subplot(1, 3, 1)
plt.axis('off')
plt.imshow(input_imgs[:, :, 0])

plt.subplot(1, 3, 2)
plt.axis('off')
plt.imshow(input_imgs[:, :, 1])

plt.subplot(1, 3, 3)
plt.axis('off')
plt.imshow(mask_img)
fir_img = input_imgs[:, :,0]
# %%
BATCH = 32
AT = tf.data.AUTOTUNE
BUFFER = 1000
STEPS_PER_EPOCH = 800//BATCH
VALIDATION_STEPS = 200//BATCH
train_dt = train_dt.cache().shuffle(BUFFER).batch(BATCH).repeat()
train_dt = train_dt.prefetch(buffer_size=AT)
# %%
from tensorflow.keras.initializers import RandomNormal, HeUniform
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, LeakyReLU, Activation, Dropout, BatchNormalization, LeakyReLU, GlobalMaxPool2D, Concatenate, ReLU, AveragePooling2D
from tensorflow.keras import losses
# %%
def define_encoder_block(layer_in, n_filters, batchnorm=True):
    init = HeUniform()
    g = Conv2D(n_filters, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(layer_in)
    if batchnorm:
        g =  BatchNormalization()(g, training=True)
    g = LeakyReLU(alpha=0.2)(g)
    return g

def define_decoder_block(layer_in, skip_in, n_filters, dropout=True):
    init = RandomNormal(stddev=0.02)
    g = Conv2DTranspose(n_filters, (4, 4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
    g = BatchNormalization()(g, training=True)
    if dropout:
        g = Dropout(0.4)(g, training=True)
    g = Concatenate()([g, skip_in])
    g = ReLU()(g)
    return g

def define_generator(latent_size, image_shape=(128, 128, 2)):
    init = RandomNormal(stddev=0.02)
    input_image = Input(shape=image_shape)
#     style_image = Input(shape=image_shape)
    # stack content and style images
#     stacked_layer = Concatenate()([content_image, style_image])
    #encoder model
    e1 = define_encoder_block(input_image, 32, batchnorm=False)
    e2 = define_encoder_block(e1, 64)
    e3 = define_encoder_block(e2, 128)
    e4 = define_encoder_block(e3, 256)
    e5 = define_encoder_block(e4, 256)
    e6 = define_encoder_block(e5, 256)
    #e7 = define_encoder_block(e6, 512)
    # bottleneck layer
    b = Conv2D(latent_size, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(e6)
    b = ReLU()(b)
    #decoder model
    #d1 = define_decoder_block(b, e7, 512)
    d2 = define_decoder_block(b, e6, 256)
    d3 = define_decoder_block(d2, e5, 256)
    d4 = define_decoder_block(d3, e4, 256, dropout=False)
    d5 = define_decoder_block(d4, e3, 128, dropout=False)
    d6 = define_decoder_block(d5, e2, 64, dropout=False)
    d7 = define_decoder_block(d6, e1, 32, dropout=False)
    #output layer
    g = Conv2DTranspose(1, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d7)
    out_image = Activation('sigmoid')(g)
    model = Model(inputs=input_image, outputs=out_image, name='generator')
    return model
# %%
model = define_generator(32, (256, 256, 2))
# %%
samples = np.ones((1, 256, 256, 2))
res = model(samples)
# %%
model.compile(loss=keras.losses.BinaryCrossentropy(from_logits=True),
             optimizer=keras.optimizers.Adam(5e-5),
             metrics=['accuracy']) 

hist = model.fit(train_dt,
            steps_per_epoch=STEPS_PER_EPOCH,
            epochs=20)

# %%
print(n_np.shape)
# sample1 = 
# %%
plt.imshow(n_np[0, :, :, 0]/65536)

# %%

mask_1 = model(h)
plt.imshow(mask_1[0, ...])
# %%
model.save_weights('./data/models/defo_mask1.h5')
# %%
