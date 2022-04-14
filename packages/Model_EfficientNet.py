import tensorflow as tf
tf.keras.mixed_precision.set_global_policy('mixed_float16')

def get_model(input_shape, output_shape):
    base_model = tf.keras.applications.efficientnet.EfficientNetB0(input_shape=input_shape,
                                                                   include_top=False)
    inputs = tf.keras.Input(shape=input_shape)
    # Transfer Learning
    x = base_model(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    # Dense
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(output_shape, activation="softmax", dtype='float32')(x)

    model = tf.keras.Model(inputs, outputs)
    return model