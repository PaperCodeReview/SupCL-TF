import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import Model

model_dict = {
    'vgg16'         : tf.keras.applications.VGG16,
    'vgg19'         : tf.keras.applications.VGG19,
    'resnet50'      : tf.keras.applications.ResNet50,
    'resnet50v2'    : tf.keras.applications.ResNet50V2,
    'resnet101'     : tf.keras.applications.ResNet101,
    'resnet101v2'   : tf.keras.applications.ResNet101V2,
    'resnet152'     : tf.keras.applications.ResNet152,
    'resnet152v2'   : tf.keras.applications.ResNet152V2,
    'xception'      : tf.keras.applications.Xception, # 299
    'densenet121'   : tf.keras.applications.DenseNet121, # 224
    'densenet169'   : tf.keras.applications.DenseNet169, # 224
    'densenet201'   : tf.keras.applications.DenseNet201, # 224
}

def create_model(args, logger):
    backbone = model_dict[args.backbone](
        include_top=False,
        pooling='avg',
        weights=None,
        input_shape=(args.img_size, args.img_size, 3))

    if args.loss == 'crossentropy':
        x = Dense(args.classes)(backbone.output)
        x = Activation('softmax', name='main_output')(x)
    elif args.loss == 'supcon':
        x = Dense(2048, name='proj_hidden')(backbone.output)
        x = Dense(128, name='proj_output')(x)
        x = Lambda(lambda x: tf.math.l2_normalize(x, axis=-1), name='main_output')(x)
    model = Model(backbone.input, x, name=args.backbone)

    if args.snapshot:
        model.load_weights(args.snapshot)
        logger.info('Load weights at {}'.format(args.snapshot))
    return model