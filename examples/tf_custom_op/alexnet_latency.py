import tensorflow as tf
import numpy as np
import tf_fim_ops
import timeit
import argparse
import os
from tabulate import tabulate

NUM_CLASSES = 4096
image_width = 224
image_height = 224
channels = 3

SEED = 1235
initializer = tf.keras.initializers.RandomNormal(seed=SEED)

eval_time = []
parser = argparse.ArgumentParser(description='Process Alexnet arguments')
parser.add_argument('-b','--batch_size', default=1, help="Input batch size", type=int)
parser.add_argument('-i','--iterations', default=10, help="Number of iterations for profiling", type=int)
parser.add_argument('-p','--profile', action="store_true", help="Enabled/Disable profiling")
parser.add_argument('-f','--functional_verify', action="store_true", help="Enabled/Disable Functional verification")
parser.add_argument('-d','--dtype', default='fp16' , help="fp16 or fp32 execution")
parser.add_argument('-m','--module', default='keras' , help="keras or fim_custom execution")

args = parser.parse_args()

def DummyExecute():
    # Create some tensors
    a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    c = tf.matmul(a, b)

class FimDenseLayer(tf.keras.layers.Layer):
    def __init__(self, weight, bias, has_bias, dtype=tf.float16):
        super(FimDenseLayer, self).__init__()
        #print(type(weight))
        self.kernel = weight
        self.bias = bias
        self.has_bias =  tf.constant([0])
        if has_bias:
          self.has_bias =  tf.constant([1])
        self.reorder = tf.constant([1])

    def build(self, input_shape):
        pass

    def call(self, input):
      return tf_fim_ops.fim_dense(input, self.kernel, self.bias, self.has_bias, self.reorder)


class AlexNet(tf.keras.Model):
    def __init__(self,data_type,has_bias=False):
        # layer 1
        super(AlexNet, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=96,
                               kernel_size=(11, 11),
                               strides=4,
                               padding="valid",
                               activation=tf.keras.activations.relu,
                               input_shape=(image_height, image_width, channels), dtype=data_type)

        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                  strides=2,
                                  padding="valid", dtype=data_type)

        self.bn1 = tf.keras.layers.BatchNormalization(dtype=data_type)
        # layer 2
        self.conv2 = tf.keras.layers.Conv2D(filters=256,
                               kernel_size=(5, 5),
                               strides=1,
                               padding="same",
                               activation=tf.keras.activations.relu, dtype=data_type)
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                  strides=2,
                                  padding="same",dtype=data_type)
        self.bn2 = tf.keras.layers.BatchNormalization(dtype=data_type)
        # layer 3
        self.conv3 = tf.keras.layers.Conv2D(filters=384,
                               kernel_size=(3, 3),
                               strides=1,
                               padding="same",
                               activation=tf.keras.activations.relu,dtype=data_type)
        # layer 4
        self.conv4 = tf.keras.layers.Conv2D(filters=384,
                               kernel_size=(3, 3),
                               strides=1,
                               padding="same",
                               activation=tf.keras.activations.relu,dtype=data_type)
        # layer 5
        self.conv5 = tf.keras.layers.Conv2D(filters=256,
                               kernel_size=(3, 3),
                               strides=1,
                               padding="same",
                               activation=tf.keras.activations.relu,dtype=data_type)
        self.pool3 = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                  strides=2,
                                  padding="same",dtype=data_type)
        self.bn3 = tf.keras.layers.BatchNormalization(dtype=data_type)
        # layer 6
        self.flatten1 = tf.keras.layers.Flatten(dtype=data_type)
        self.dense1 = tf.keras.layers.Dense(units=4096, kernel_initializer=initializer, use_bias=has_bias, dtype=data_type)
        self.dropout1 = tf.keras.layers.Dropout(rate=0.2,dtype=data_type)
        # layer 7
        self.dense2 = tf.keras.layers.Dense(units=4096, kernel_initializer=initializer, use_bias=has_bias, dtype=data_type)
        self.dropout2 = tf.keras.layers.Dropout(rate=0.2,dtype=data_type)
        # layer 8
        self.dense3 = tf.keras.layers.Dense(units=NUM_CLASSES, use_bias=has_bias, dtype=data_type)


    def fill_weights(self):

       weights1 = self.dense1.get_weights()
       has_bias1 = len(weights1) > 1
       if has_bias1 == True:
           self.fim_dense1 = FimDenseLayer(tf.convert_to_tensor(weights1[0]),tf.convert_to_tensor(weights1[1]),has_bias1, dtype=tf.float16)
       else:
           self.fim_dense1 = FimDenseLayer(tf.convert_to_tensor(weights1[0]),tf.constant([0.0],dtype=tf.float16) ,has_bias1, dtype=tf.float16)

       weights2 = self.dense2.get_weights()
       has_bias2 = len(weights2) > 1
       if has_bias2 == True:
           self.fim_dense2 = FimDenseLayer(tf.convert_to_tensor(weights2[0]),tf.convert_to_tensor(weights2[1]),has_bias2, dtype=tf.float16)
       else:
           self.fim_dense2 = FimDenseLayer(tf.convert_to_tensor(weights2[0]),tf.constant([0.0],dtype=tf.float16),has_bias2, dtype=tf.float16)

       weights3 = self.dense3.get_weights()
       has_bias3 = len(weights3) > 1
       if has_bias3 == True:
           self.fim_dense3 = FimDenseLayer(tf.convert_to_tensor(weights3[0]),tf.convert_to_tensor(weights3[1]),has_bias3, dtype=tf.float16)
       else:
           self.fim_dense3 = FimDenseLayer(tf.convert_to_tensor(weights3[0]),tf.constant([0.0],dtype=tf.float16),has_bias3, dtype=tf.float16)


    def timeit_add(self,layer,index,name,input,output):
        if args.profile == True :
            #print(" {} {} input shape {}".format(name, index, input.shape))
            eval_time.append([name + str(index),
                (timeit.timeit(lambda: layer(input), number = args.iterations)), input.shape, output.shape])


    def profile_model(self, inputs , verify):
        x = self.conv1(inputs)
        self.timeit_add(self.conv1,1,"Conv ",inputs,x)
        x1 = self.pool1(x)
        self.timeit_add(self.pool1,1,"Pool ",x,x1)
        x = self.bn1(x1)
        self.timeit_add(self.bn1,1,"Bn ",x1,x)
        x1 = self.conv2(x)
        self.timeit_add(self.conv2,2,"Conv ",x,x1)
        x = self.pool2(x1)
        self.timeit_add(self.pool2,2,"Pool ",x1,x)
        x1 = self.bn2(x)
        self.timeit_add(self.bn2,2,"Bn ",x,x1)
        x = self.conv3(x1)
        self.timeit_add(self.conv3,3,"Conv ",x1,x)
        x1 = self.conv4(x)
        self.timeit_add(self.conv4,4,"Conv ",x,x1)
        x = self.conv5(x1)
        self.timeit_add(self.conv5,5,"Conv ",x1,x)
        x1 = self.pool3(x)
        self.timeit_add(self.pool3,3,"Pool ",x,x1)

        #Additional pool for 4096 matching
        x = self.pool3(x1)
        self.timeit_add(self.pool3,4,"Pool ",x1,x)
        x1 = self.flatten1(x)

        if verify:
            x1_gpu = np.copy(x1)
            x1_fim = np.copy(x1)

            x2 = self.dense1(x1_gpu)
            x3 = self.dense2(x2)
            x4_gpu = self.dense3(x3)

            x2 = self.fim_dense1(x1_fim)
            x3 = self.fim_dense2(x2)
            x4_fim = self.fim_dense3(x3)

            result = np.testing.assert_array_almost_equal(x4_gpu, x4_fim, decimal=5)
            print("Functional Verification : {}".format(result))
            return x4_gpu
        else:
            if args.module == 'keras':
              x2 = self.dense1(x1)
              self.timeit_add(self.dense1,1,"Dense ",x1,x2)
              x3 = self.dense2(x2)
              self.timeit_add(self.dense2,2,"Dense ",x2,x3)
              x4 = self.dense3(x3)
              self.timeit_add(self.dense3,3,"Dense ",x3,x4)
              return x4
            else:
              x2 = self.fim_dense1(x1)
              self.timeit_add(self.fim_dense1,1,"Dense ",x1,x2)
              x3 = self.fim_dense2(x2)
              self.timeit_add(self.fim_dense2,2,"Dense ",x2,x3)
              x4 = self.fim_dense3(x3)
              self.timeit_add(self.fim_dense3,3,"Dense ",x3,x4)
              return x4

    def call(self, inputs , verify):
        x = self.conv1(inputs)
        x1 = self.pool1(x)
        x = self.bn1(x1)
        x1 = self.conv2(x)
        x = self.pool2(x1)
        x1 = self.bn2(x)
        x = self.conv3(x1)
        x1 = self.conv4(x)
        x = self.conv5(x1)
        x1 = self.pool3(x)

        #Additional pool for 4096 matching
        x = self.pool3(x1)
        x1 = self.flatten1(x)

        if verify:
            x1_gpu = np.copy(x1)
            x1_fim = np.copy(x1)

            x2 = self.dense1(x1_gpu)
            x3 = self.dense2(x2)
            x4_gpu = self.dense3(x3)

            x2 = self.fim_dense1(x1_fim)
            x3 = self.fim_dense2(x2)
            x4_fim = self.fim_dense3(x3)

            result = np.testing.assert_array_almost_equal(x4_gpu, x4_fim, decimal=5)
            print("Functional Verification : {}".format(result))
            return x4_gpu
        else:
            if args.module == 'keras':
              x2 = self.dense1(x1)
              x3 = self.dense2(x2)
              x4 = self.dense3(x3)
              return x4
            else:
              x2 = self.fim_dense1(x1)
              x3 = self.fim_dense2(x2)
              x4 = self.fim_dense3(x3)
              return x4

def copy_weights(model,input):
    #Warmup and initialize weights
    #model.build( input_shape=(1,image_height, image_width, channels))
    orig_module = args.module
    args.module = 'keras'
    model(input, False)
    model.fill_weights()
    args.module = orig_module


def alexnet_model_run(dtype):
    tf_fim_ops.fim_init()

    input   = tf.random.uniform(shape=(args.batch_size, image_height, image_width, channels),dtype=dtype)
    model = AlexNet(dtype)
    copy_weights(model,input)

    if args.profile:
        # model and gpu initialization and LUT load
        DummyExecute()
        predictions = model(input, False)

    if args.functional_verify:
       predictions = model(input, True)

    eval_time.clear()
    predictions = model.profile_model(input, False)

    #Model Summary , Todo:
    # model.summary()

    if args.profile:
        # for disabling internal profiling calls.
        eval_time.append(["End to End ", (timeit.timeit(lambda : model(input, False),
                            number = args.iterations)), input.shape, predictions.shape])

        for i in range(len(eval_time)):
            eval_time[i][1] = (eval_time[i][1] * 1000 ) / args.iterations

        print(tabulate(eval_time, headers=["Index", "Layer", "Time(ms)", "Input", "Output"], showindex="always", tablefmt='github'))

    tf_fim_ops.fim_deinit()


if __name__ == '__main__':
    print('User arguments {}'.format(args))
    dtype = tf.float16
    if args.dtype == 'fp32':
        dtype = tf.float32
    else:
        tf.keras.backend.set_floatx('float16')
    alexnet_model_run(dtype)
