import vgg
import numpy as np
import reader
import pdb
from task1_resnet101active import resnet101_model_new

BATCH_SIZE = 32
IMAGE_SIZE = 224
TRAIN_IMAGES_PATH = '/media/michelle/1.8T/2017_ISBI_suply/224data/toy'
VGG_PATH = 'imagenet-vgg-verydeep-19.mat'

def get_repscore(features_vgg,features_model,VGG_PATH=VGG_PATH):
    '''The get_repscore is a function that calculate the loss between proposed model and pre-trained other model
    # Arguments
        features: batch_size*14*14*2048
        VGG_PATH:
    author:michelle
    '''
    CONTENT_LAYERS = "relu4_2"
    content_layers = CONTENT_LAYERS.split(',')
    [batch_size,height,width,channels] = features.get_shape().as_list()
    vgg_loss += tf.nn.l2_loss(features - content_images) / tf.to_float(size)

def val(model_path):
    data_path = '../../../../2017_ISBI_suply/224data'
    config = tf.ConfigProto(device_count={'gpu':0})
    config.gpu_options.per_process_gpu_memory_fraction = 0.95
    session = tf.Session(config=config)
    K.set_session(session)

    model_new = resnet101_model_new(224, 224, 3, 2)
    model_new.load_weights(model_path)
    select_data_dir = data_path + '/test'
    batch_size = 30
    
    pred_val_lists = []
    label_lists = []
    data_label_list = get_data_label_predict(model_new,select_data_dir,batch_size)
    pdb.set_trace()

def get_vgg_features(feature_arr,VGG_PATH=VGG_PATH):
    # images = reader.image(BATCH_SIZE, 224, IMAGE_SIZE, TRAIN_IMAGES_PATH)
    model_new = resnet101_model_new(224, 224, 3, 2)
    model_new.load_weights(model_path)
    net, _ = vgg.net(VGG_PATH, images)
    # for f in net:
    #     print (f)
    vgg_features= net['relu5_4']
    # pdb.set_trace()
    return vgg_features

def get_res_features(TRAIN_IMAGES_PATH=TRAIN_IMAGES_PATH,VGG_PATH=VGG_PATH):
    images = reader.image(BATCH_SIZE, 224, IMAGE_SIZE, TRAIN_IMAGES_PATH)
    net, _ = vgg.net(VGG_PATH, images)
    # for f in net:
    #     print (f)
    vgg_features= net['relu5_4']
    # pdb.set_trace()
    return vgg_features
    
# for layer in content_layers:
#     generated_images, content_images = tf.split(value=net[layer], num_or_size_splits=2, axis=0)
#     size = tf.size(generated_images)
#     shape = tf.shape(generated_images)
#     width = shape[1]
#     height = shape[2]
#     num_filters = shape[3]
#     content_loss += tf.nn.l2_loss(generated_images - content_images) / tf.to_float(size)
# content_loss = content_loss