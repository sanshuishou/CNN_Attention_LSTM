import tensorflow as tf

#这里是主要使用的模型结构 CNN_Attention_LSTM
def attention_3d_block(inputs,TIME_STEPS,SINGLE_ATTENTION_VECTOR):
    # inputs.shape = (batch_size, time_steps, input_dim)
    # inputs = tf.expand_dims(inputs,1)
    input_dim = int(inputs.shape[2])
    a = tf.keras.layers.Permute((2, 1))(inputs)
    a = tf.keras.layers.Reshape((input_dim, TIME_STEPS))(a)  # this line is not useful. It's just to know which dimension is what.
    a = tf.keras.layers.Dense(TIME_STEPS, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = tf.keras.layers.Lambda(lambda x: tf.keras.backend.mean(x, axis=1), name='dim_reduction')(a)
        a = tf.keras.layers.RepeatVector(input_dim)(a)
    a_probs = tf.keras.layers.Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = tf.keras.layers.Multiply()([inputs, a_probs])
    return output_attention_mul

def conv_lstm(TIME_STEPS, INPUT_DIM,lstm_units = 32):
    tf.keras.backend.clear_session()  # 清除之前的模型，省得压满内存
    inputs = tf.keras.Input(shape=(TIME_STEPS, INPUT_DIM,))
    x = tf.keras.layers.Conv1D(65,3,padding='same')(inputs)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv1D(128,3,padding='same')(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.RepeatVector(TIME_STEPS)(x)
    x = tf.keras.layers.LSTM(lstm_units,return_sequences=True)(x)
    x = attention_3d_block(x,TIME_STEPS,1)
    x = tf.keras.layers.LSTM(lstm_units)(x)
    x = tf.keras.layers.Dense(1024)(x)
    output = tf.keras.layers.Dense(1,kernel_regularizer=tf.keras.regularizers.L1L2())(x)
    model = tf.keras.Model(inputs=[inputs], outputs=output)
    return model