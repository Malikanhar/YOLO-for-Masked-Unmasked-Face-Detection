import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, LeakyReLU, Layer, ZeroPadding2D, Add
from tensorflow.keras.regularizers import l2

class DarknetBatchNormalization(BatchNormalization):
    def call(self, x, training=False):
        if training is None:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)

class DarknetConv(Layer):
    def __init__(self, filters, size, strides=1, batch_norm=True):
        super(DarknetConv, self).__init__()
        if strides == 1:
            self.padding = 'same'
        else:
            self.padding = 'valid'
        
        self.batch_norm = batch_norm

        self.bn = DarknetBatchNormalization()
        self.leakyRelu = LeakyReLU(alpha=0.1)
        
        self.conv = Conv2D(filters=filters, kernel_size=size, 
                    strides=strides, padding=self.padding, 
                    use_bias=not batch_norm, kernel_regularizer=l2(0.0005))

    def call(self, x):
        if self.padding == 'valid':
            x = ZeroPadding2D(((1, 0), (1, 0)))(x) # top left half-padding
        x = self.conv(x)
        if self.batch_norm:
            x = self.bn(x)
            x = self.leakyRelu(x)
        return x

class DarknetResidual(Layer):
    def __init__(self, filters):
        super(DarknetResidual, self).__init__()
        # didalam residual, jumlah filter setengahnya
        self.dnconv1 = DarknetConv(filters // 2, 1)
        # yang ini tetep utuh jumlah filternya
        self.dnconv2 = DarknetConv(filters, 3)
    
    def call(self, x):
        # simpen yg sebelumnya
        prev = x
        x = self.dnconv1(x)
        x = self.dnconv2(x)
        # gabungin
        x = Add()([prev, x])
        return x


class DarknetBlock(Layer):
    def __init__(self, filters, blocks):
        super(DarknetBlock, self).__init__()
        # konvolusi sebelum masuk ke residual, stridenya 2 dan kernel 3
        self.dnconv = DarknetConv(filters, 3, strides=2)
        self.dnconvs = []
        for _ in range(blocks):
            self.dnconvs.append(DarknetResidual(filters))

    def call(self, x):
        x = self.dnconv(x)
        for i in range(len(self.dnconvs)):
            x = self.dnconvs[i](x)
        return x



class Darknet53(Model):
    def __init__(self, name='yolo_darknet'):
        super(Darknet53, self).__init__(name=name)
        # output (256, 256)
        self.dnconv1 = DarknetConv(32, 3)
        # darknet block sebanyak 1 berukuran 64. output (128, 128)
        self.dnblock1 = DarknetBlock(64, 1)
        # Darknet block sebanyak 2 berukuran 128. output ()
        self.dnblock2 = DarknetBlock(128, 2)
        # Darknet block sebanyak 8 berukuran 256
        self.dnblock3 = DarknetBlock(256, 8)
        # Darknet block sebanyak 8 berukuran 512
        self.dnblock4 = DarknetBlock(512, 8)
        # Darknet block sebanyak 4 berukuran 1024
        self.dnblock5 = DarknetBlock(1024, 4)
    
    def call(self, x):
        x = self.dnconv1(x)
        x = self.dnblock1(x)
        x = self.dnblock2(x)
        x = x_36 = self.dnblock3(x)
        x = x_61 = self.dnblock4(x)
        x = self.dnblock5(x)
        # di darknet53, outputnya diambil di layer 36, 61, dan diujung
        return x_36, x_61, x
