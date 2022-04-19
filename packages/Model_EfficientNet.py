import tensorflow as tf
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2B0
tf.keras.mixed_precision.set_global_policy('mixed_float16')

def get_model(input_shape, output_shape):
    base_model = EfficientNetV2B0(input_shape=input_shape, include_top=False)
    inputs = tf.keras.Input(shape=input_shape)
    # Transfer Learning
    x = base_model(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    # Dense
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(output_shape, activation="softmax", dtype='float32')(x)

    model = tf.keras.Model(inputs, outputs)
    return model

if __name__ == '__main__':
    model = get_model(input_shape=(256, 256, 3), output_shape=100)
    print(model.summary())