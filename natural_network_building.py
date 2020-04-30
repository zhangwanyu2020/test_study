import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 下载数据
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_lables), (test_images, test_lables) = fashion_mnist.load_data()


# train_images.shape
# train_lables[:5]
# 展示训练集前5张图
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(10, 10))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


plotImages(train_images[:5])
# 构建2层全连接神经网络
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')])
# model.summary()

# 配置模型的参数 优化器&损失函数&评估指标
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# 训练模型
model.fit(train_images, train_lables, epochs=10, validation_data=(test_images, test_lables))
# 保存权重
model.save_weights('/Users/zhangwanyu/Desktop/test_data')
model.load_weights('/Users/zhangwanyu/Desktop/test_data')
loss, acc = model.evaluate(test_images, test_lables, verbose=2)
# 保存模型
model.save('my_model.h5')
new_model = tf.keras.models.load_model('my_model.h5')
loss, acc = new_model.evaluate(test_images, test_lables, verbose=2)
# 创建一个回调，每个epoch保存模型的权重
checkpoint_path = 'fashion_mnist_1/cp-{epoch:04d}.ckpt'
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    save_freq='epoch')
# 使用checkpoint_path格式保存权重
model.save_weights(checkpoint_path.format(epoch=0))
# 使用新的回调训练模型
model.fit(train_images,
          train_lables,
          epochs=5,
          callbacks=[cp_callback],
          validation_data=(test_images, test_lables))
# CNN
train_images = train_images[..., tf.newaxis]
test_images = test_images[..., tf.newaxis]

model1 = tf.keras.Sequential()
model1.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model1.add(tf.keras.layers.MaxPooling2D((2, 2)))
model1.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model1.add(tf.keras.layers.MaxPooling2D((2, 2)))
model1.add(tf.keras.layers.Flatten())
model1.add(tf.keras.layers.Dense(256, activation='relu'))
model1.add(tf.keras.layers.Dense(10, activation='softmax'))
# model.summary()
# RNN
model2 = tf.keras.Sequential()
model2.add(tf.keras.layers.LSTM(128, input_shape=(None, 28)))
model2.add(tf.keras.layers.Dense(10, activation='softmax'))
# model2.summary()

from tensorflow.keras import layers


# 写法一
class Linear(layers.Layer):
    def __init__(self, units=32, input_dim=28):
        super(Linear, self).__init__()
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(initial_value=w_init(shape=(input_dim, units), dtype='float32'), trainable=True)
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(initial_value=b_init(shape=(units,), dtype='float32'), trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b


inputs = tf.ones((2, 2))
linear_layer = Linear(4, 2)
# print(linear_layer.w)
outputs = linear_layer(inputs)


# print(outputs)

# 写法二
class Linear(layers.Layer):
    def __init__(self, units=32):
        super(Linear, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                                 initializer='random_normal',
                                 trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b


inputs = tf.ones((2, 2))
linear_layer = Linear(4)
outputs = linear_layer(inputs)
print(linear_layer.w)


class MLPBlock(layers.Layer):
    def __init__(self):
        super(MLPBlock, self).__init__
        self.linear_1 = Linear(32)
        self.linear_2 = Linear(32)
        self, linear_3 = Linear(1)

    def call(self, inputs):
        x = self.linear_1(inputs)
        x = tf.nn.relu(x)
        x = self.linear_2(x)
        x = tf.nn.relu(x)
        return self.linear_3(x)


optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

for x_batch_train, y_batch_train in train_dataset:
    with tf.GradientTape() as tape:
        logits = layer(x_batch_train)
        loss_value = loss_fn(y_batch_train, logits)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradient(zip(grads, model.trainable_weights))