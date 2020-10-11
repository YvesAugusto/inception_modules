from __future__ import print_function, division
import tensorflow as tf
import numpy as np
from conv_layer import ConvLayer
from batch_norm_layer import BatchNormLayer
from avg_pool import AvgPool

class InceptionNaiveLayer:
    def __init__(self, branches, main_branch):
        self.f = tf.nn.relu
        branch_0 = branches[0]
        self.lateral_convs = {}
        self.lateral_conv0_0 = ConvLayer(branch_0["filters_size"][0],
                                         branch_0["inputs_size"][0],
                                         branch_0["inputs_size"][1], stride=1, padding='SAME')

        self.lateral_conv0_0_btn = BatchNormLayer(branch_0["inputs_size"][1])

        self.lateral_conv0_1 = ConvLayer(branch_0["filters_size"][1],
                                         branch_0["inputs_size"][1],
                                         branch_0["inputs_size"][2], padding='SAME')

        self.lateral_conv0_1_btn = BatchNormLayer(branch_0["inputs_size"][2])

        branch_1 = branches[1]
        self.lateral_conv1_0 = ConvLayer(branch_1["filters_size"][0],
                                         branch_1["inputs_size"][0],
                                         branch_1["inputs_size"][1], stride=1, padding='SAME')

        self.lateral_conv1_0_btn = BatchNormLayer(branch_1["inputs_size"][1])

        self.lateral_conv1_1 = ConvLayer(branch_1["filters_size"][1],
                                         branch_1["inputs_size"][1],
                                         branch_1["inputs_size"][2], padding='SAME')

        self.lateral_conv1_1_btn = BatchNormLayer(branch_1["inputs_size"][2])

        branch_2 = branches[2]
        self.lateral_conv2_0 = ConvLayer(branch_2["filters_size"][0],
                                         branch_2["inputs_size"][0],
                                         branch_2["inputs_size"][1], padding='SAME')
        self.lateral_conv2_0_btn = BatchNormLayer(branch_2["inputs_size"][1])


        self.lateral_main = ConvLayer(main_branch["filters_size"][0],
                                      main_branch["inputs_size"][0],
                                      main_branch["inputs_size"][1], padding='SAME')

        self.lateral_main_btn = BatchNormLayer(main_branch["inputs_size"][1])

    def forward(self, X):

        X = X.astype(np.float32)

        FX_0 = self.lateral_conv0_0.forward(X)
        FX_0 = self.lateral_conv0_0_btn.forward(FX_0)
        FX_0 = self.lateral_conv0_1.forward(FX_0)
        FX_0 = self.lateral_conv0_1_btn.forward(FX_0)
        FX_0 = self.f(FX_0)

        print(f'first branch output.shape:{FX_0.shape}')

        FX_1 = self.lateral_conv1_0.forward(X)
        FX_1 = self.lateral_conv1_0_btn.forward(FX_1)
        FX_1 = self.lateral_conv1_1.forward(FX_1)
        FX_1 = self.lateral_conv1_1_btn.forward(FX_1)
        FX_1 = self.f(FX_1)

        print(f'second branch output.shape: {FX_1.shape}')

        FX_2 = tf.nn.max_pool2d(X, ksize=2, strides=1, padding='VALID')
        FX_2 = self.lateral_conv2_0.forward(FX_2)
        FX_2 = self.lateral_conv2_0_btn.forward(FX_2)
        FX_2 = self.f(FX_2)

        print(f'third branch output.shape: {FX_2.shape}')

        X = self.lateral_main.forward(X)
        X = self.lateral_main_btn.forward(X)
        X = self.f(X)

        print(f'main branch output.shape: {X.shape}')

        return tf.concat((X, FX_0, FX_1, FX_2), axis=3)

        # TODO
        pass

    def predict(self, X):
        # TODO
        pass

    def set_session(self):
        # TODO
        pass


branches=[{"filters_size":[1,3], "inputs_size":[3, 64, 64], "strides":[]},
          {"filters_size":[1,5], "inputs_size":[3, 64, 64], "strides":[]},
          {"filters_size":[1], "inputs_size":[3, 64], "strides":[]}]
main_brach={"filters_size":[1], "inputs_size":[3, 64], "stride":None}

inception_layer=InceptionNaiveLayer(branches, main_brach)
output=inception_layer.forward(np.random.random((1,224,224,3)))
print(output)
