#split_data.py是一个可以将数据集自动按9:1的训练集和验证集进行划分的脚本，如果训练自己的数据集要将train文件里和predict的num_classes的个数改成要分类的个数
import torch.nn as nn
import torch            #导入两个pytorch包


class AlexNet(nn.Module):       #创建一个类，名为AlexNet，继承nn.Module这个父类
    def __init__(self, num_classes=1000, init_weights=False):     #定义初始化函数，定义模型中使用的网络层结构
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(     #使用nn.Sequential模块，可以将一系列的层结构打包，组合成一个新的结构，取名为features，features代表
                                           #专门用来提取图像特征的结构，对于网络层次比较多的结构，可以使用nn.Sequential函数来精简代码
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),  # input[3, 224, 224]  output[48, 55, 55]
            #padding有两种类型的表示，int类型，tuple类型，tuple（1,2）表示上下方补一行0，左右列补两列0
            nn.ReLU(inplace=True),    #ReLU激活函数，inplace参数可以理解为pytorch通过一种方式增加计算量，但是扩大内存使用容量
                                      #通过inplace这个方法可以向内存中载入更大的模型
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[48, 27, 27]
            nn.Conv2d(48, 128, kernel_size=5, padding=2),           # output[128, 27, 27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 13, 13]
            nn.Conv2d(128, 192, kernel_size=3, padding=1),          # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),          # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),          # output[128, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 6, 6]
        )
        #展平操作在前向传播函数中，26-35为全连接层，上边是卷积层和池化层
        self.classifier = nn.Sequential(   #classifier结构包含了最后的三层全连接层，也使用nn.Sequential模块将全连接层打包成一个新的模块
            nn.Dropout(p=0.5),    #AlexNet使用Dropout方式在网络正向传播过程中随机失活一部分神经元，目的是为了防止过拟合，这句操作就是使用
                                  #Dropout方式,p是随机失活的比例，默认等于0.5
            nn.Linear(128 * 6 * 6, 2048),   #pytorch一般将channel放在首位，所以为128 * 6 * 6，第一个全连接层的节点个数是2048个
            nn.ReLU(inplace=True),      #经过一个ReLU激活函数
            nn.Dropout(p=0.5),         #（暂时）对于全连接层来说。linear之后一定会加上一个激活函数，如ReLU
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),   #num_classes即为数据集的类别个数
        )
        if init_weights:    #初始化权重，若在搭建网络中，初始化权重为True，则会进入初始化权重这个函数_initialize_weights
            self._initialize_weights()

    def forward(self, x):      #定义正向传播的过程，x为输入的数据Pytorch Tensor的通道顺序：[batch,channel.height.width]
        x = self.features(x)      #features即为一些列的卷积和池化操作
        x = torch.flatten(x, start_dim=1)    #对输出进行展平操作
        x = self.classifier(x)       #全连接层
        return x

    def _initialize_weights(self):    #初始化权重函数
        for m in self.modules():    #遍历self.modules这个模块，继承nn.Module，self.modules迭代定义的每个层结构，遍历每一个层结构，判断是哪一个类别
            if isinstance(m, nn.Conv2d):    #如果层结构是卷积层的话，就用凯明初始化变量方法对权重weight进行初始化，如果偏置不为空，就用0进行初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):  #如果层结构是全连接层的话，就用正态分布对权重weight进行初始化，将其偏置初始化为0
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
