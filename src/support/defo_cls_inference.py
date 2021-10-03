#%%
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import RandomNormal, HeUniform
from tensorflow.keras.models import Model


# %%

class ClassiModel(object):
    __instance = None

    def __init__(self, model_path, model, version, n_labels, label_dict, image_shape):
        if ClassiModel.__instance != None:
            raise Exception("This class is a singelton.")
        else:
            self.model_path = model_path
            self.model_version = version
            self.model = model
            self.n_labels = n_labels
            self.label_dict = label_dict
            self.image_shape = image_shape
            ClassiModel.__instance = self
        

    @staticmethod
    def get_v2_model():
        height, width, channels = 224, 224, 3  # the image size
        n_labels = 8 # the number of classes
        class_dict = {0 : "agriculture", 1 : "clear", 2 : "cloudy", 3 : "cultivation", 4 : "habitation", 5 : "primary", 6 : "road", 7 : "water"}

        feat_model = tf.keras.applications.densenet.DenseNet121(weights='imagenet', include_top=False, input_shape=(height, width, channels))
  
        x = feat_model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(1024, activation='relu', name='last_linear')(x)
        x = layers.Dense(n_labels, activation='sigmoid', name='output')(x)
        model = Model(inputs=feat_model.input, outputs=x, name='multi_class_multi_label_classifier')
        
        for layer in model.layers:
            layer.trainable = False
          
        return model, (height, width, channels), n_labels, class_dict

    @staticmethod
    def get_v3_model():
        height, width, channels = 128, 128, 3
        n_labels = 8 # the number of classes
        class_dict = {0 : "agriculture", 1 : "clear", 2 : "cloudy", 3 : "cultivation", 4 : "habitation", 5 : "primary", 6 : "road", 7 : "water"}
        feat_model = tf.keras.applications.MobileNetV3Small(weights='imagenet', include_top=False, input_shape=(height, width, channels))
        

        x = feat_model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(n_labels, activation='sigmoid', name='output')(x)
        model = Model(inputs=feat_model.input, outputs=x, name='multi_class_multi_label_classifier')

        for layer in model.layers:
            layer.trainable = False
            
        return model, (height, width, channels), n_labels, class_dict


    @staticmethod
    def build_model(weight_path, version=2):
        if version == 2:
            model, image_shape, n_labels, class_dict = ClassiModel.get_v3_model()
        else:
            raise Exception("Invalid version")
        try:
            model.load_weights(weight_path)
            return model, image_shape, n_labels, class_dict
        except Exception as e:
            return e
        
    @staticmethod
    def getInstance():
        model_path = "../../data/models/defo_ks2.h5"
        version = 2
        if ClassiModel.__instance == None:
            model, image_shape, n_labels, class_dict = ClassiModel.build_model(model_path, version)
            ClassiModel(model_path, model, version, n_labels, class_dict, image_shape)
        return ClassiModel.__instance

    def load_image(self):
        #TODO : load image from db given image_ID > return np.array() object
        pass
    
    def load_images(self, path):
        #TODO : load images from db given image_ID array > return np.array() object
        pass

    def preprocess_resize(self, images):

        def func(x):
            return cv2.resize(x, (self.image_shape[0], self.image_shape[1]), interpolation = cv2.INTER_NEAREST)

        if len(images.shape) == 3:
            images = images[np.newaxis, ...]
        else:
            pre_list = []
            for i in range(images.shape[0]):
                pre_list.append(func(images[i, ...]))
            resized_images = np.array(pre_list)
        return resized_images

    def inference(self, images):
        threshold = 0.98
        images = self.preprocess_resize(images)
        images = images/255
        results = (self.model.predict(images)>threshold).astype('int')
        
        class_result = []
        for i in range(results.shape[0]):
            result = results[i, ...]
            idx = list(np.where(result == 1)[0])
            classes = [self.label_dict[j] for j in idx]
            class_result.append(classes)


        return class_result

# %%

class MaskModel(object):
    __instance = None

    def __init__(self, model_path, model, version, output_shape, image_shape):
        if MaskModel.__instance != None:
            raise Exception("This class is a singelton.")
        else:
            self.model_path = model_path
            self.model_version = version
            self.model = model
            self.output_shape = output_shape
            self.image_shape = image_shape
            MaskModel.__instance = self
        

    @staticmethod
    def get_v1_model():
        latent_size = 32
        image_shape = (256, 256, 2)
        output_shape = (256, 256, 1)

        def define_encoder_block(layer_in, n_filters, batchnorm=True):
            init = HeUniform()
            g = layers.Conv2D(n_filters, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(layer_in)
            if batchnorm:
                g =  layers.BatchNormalization()(g, training=True)
            g = layers.LeakyReLU(alpha=0.2)(g)
            return g

        def define_decoder_block(layer_in, skip_in, n_filters, dropout=True):
            init = RandomNormal(stddev=0.02)
            g = layers.Conv2DTranspose(n_filters, (4, 4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
            g = layers.BatchNormalization()(g, training=True)
            if dropout:
                g = layers.Dropout(0.4)(g, training=True)
            g = layers.Concatenate()([g, skip_in])
            g = layers.ReLU()(g)
            return g

        init = RandomNormal(stddev=0.02)
        input_image = layers.Input(shape=image_shape)
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
        b = layers.Conv2D(latent_size, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(e6)
        b = layers.ReLU()(b)
        #decoder model
        #d1 = define_decoder_block(b, e7, 512)
        d2 = define_decoder_block(b, e6, 256)
        d3 = define_decoder_block(d2, e5, 256)
        d4 = define_decoder_block(d3, e4, 256, dropout=False)
        d5 = define_decoder_block(d4, e3, 128, dropout=False)
        d6 = define_decoder_block(d5, e2, 64, dropout=False)
        d7 = define_decoder_block(d6, e1, 32, dropout=False)
        #output layer
        g = layers.Conv2DTranspose(1, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d7)
        out_image = layers.Activation('sigmoid')(g)
        model = Model(inputs=input_image, outputs=out_image, name='generator')

        for layer in model.layers:
            layer.trainable = False

        return model, output_shape, image_shape

    @staticmethod
    def build_model(weight_path, version=1):
        if version == 1:
            model, output_shape, image_shape = MaskModel.get_v1_model()
        else:
            raise Exception("Invalid version")
        try:
            model.load_weights(weight_path)
            return model, output_shape, image_shape
        except Exception as e:
            return e
        
    @staticmethod
    def getInstance():
        model_path = "../../data/models/defo_mask1.h5"
        version = 1
        if MaskModel.__instance == None:
            model, output_shape, image_shape = MaskModel.build_model(model_path, version)
            MaskModel(model_path, model, version, output_shape, image_shape)
        return MaskModel.__instance

    def load_image(self, path):
        #TODO : load image from db given image_ID > return np.array() object
        pass
    
    def load_images(self, path):
        #TODO : load images from db given image_ID array > return np.array() object
        pass

    def preprocess_resize(self, images):

        def func(x):
            return cv2.resize(x, (self.image_shape[0], self.image_shape[1]), interpolation = cv2.INTER_NEAREST)

        if len(images.shape) == 3:
            images = images[np.newaxis, ...]
        
        pre_list = []
        for i in range(images.shape[0]):
            pre_list.append(func(images[i, ...]))
        resized_images = np.array(pre_list)
        return resized_images

    def inference(self, images):
        threshold = 0.24
        images = self.preprocess_resize(images)
        images = images/255
        results = (self.model.predict(images)>threshold).astype('int')

        return results

# %%
if __name__ == '__main__':
    # s = ClassiModel.getInstance()
    images = np.load("../../data/data/sample_RGB_dataset.npz")['images']

    m = MaskModel.getInstance()
# %%
