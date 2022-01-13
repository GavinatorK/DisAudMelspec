from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import json
from tensorflow.python.platform import tf_logging
import logging as _logging
import sys as _sys
from tensorflow.keras.callbacks import ReduceLROnPlateau
import os
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers


INPUT_TENSOR_NAME = "input_2_input"  # Watch out, it needs to match the name of the first layer + "_input"
HEIGHT = 224
WIDTH = 224
DEPTH = 3
IM_SIZE = (224, 224)
NUM_CLASSES = 5
BATCH_SIZE = 10
CLASSES = ["Animals", "domestic_sounds", "Human_non-speech_sounds", "Natural_soundscapes_&_water_sounds",
           "urban_noises"]

#change 1


def keras_model_fn(train_batches, val_batches, epochs):
    net = tf.keras.applications.efficientnet.EfficientNetB7(
        include_top=False,
        weights="imagenet",
        input_tensor=None,
        input_shape=(224, 224, 3),

    )
    
    #change 1
    # Freeze the pretrained weights
    net.trainable = True

    x = net.output
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    output_layer = Dense(5, activation='softmax', name='softmax')(x)
    model = Model(inputs=net.input, outputs=output_layer)
    
    # change 2
    for layer in model.layers[-10:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True
    
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    # Estimate class weights for unbalanced dataset
    # class_weights = class_weight.compute_class_weight(
    #                'balanced',
    #                 np.unique(train_batches.classes),
    #                 train_batches.classes)

    ReduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                 patience=5, min_lr=3e-4)

    #change 3
    model.fit(train_batches,
                        validation_data=valid_batches,
                        epochs=epochs,
                        steps_per_epoch=96,
                        callbacks=[ReduceLR])
    return model



def _parse_args():
    import argparse

    parser = argparse.ArgumentParser()

    # Data, model, and output directories
    # model_dir is always passed in from SageMaker. By default this is a S3 path under the default bucket.
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--sm-model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ.get('SM_HOSTS')))
    parser.add_argument('--current-host', type=str, default=os.environ.get('SM_CURRENT_HOST'))

    return parser.parse_known_args()


if __name__ == "__main__":
    args, unknown = _parse_args()

    data_dir = args.train
    train_datagen = ImageDataGenerator(
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.1,
        fill_mode='nearest')
    train_batches = train_datagen.flow_from_directory(data_dir + '/train',
                                                      classes=CLASSES,
                                                      target_size=IM_SIZE,
                                                      class_mode='categorical', shuffle=True,
                                                      batch_size=BATCH_SIZE)

    valid_datagen = ImageDataGenerator()
    valid_batches = valid_datagen.flow_from_directory(data_dir + '/val',
                                                      classes=CLASSES,
                                                      target_size=IM_SIZE,
                                                      class_mode='categorical', shuffle=False,
                                                      batch_size=BATCH_SIZE)

    test_datagen = ImageDataGenerator()
    test_batches = test_datagen.flow_from_directory(data_dir + '/test',
                                                    classes=CLASSES,
                                                    target_size=IM_SIZE,
                                                    class_mode='categorical', shuffle=False,
                                                    batch_size=16)

    # Create the Estimator
    sound_classifier = keras_model_fn(train_batches, valid_batches, 30)

    if args.current_host == args.hosts[0]:
        # save model to an S3 directory with version number '00000001'
        sound_classifier.save(os.path.join(args.sm_model_dir, '000000001'), 'sound_model.h5')