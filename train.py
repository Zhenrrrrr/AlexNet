#导入包  AlexNet
import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt     #pyplot用来绘制图像
import numpy as np
import torch.optim as optim
from model import AlexNet
import os
import json
import time
import sys

from tqdm import tqdm

#使用torch.device函数指定在训练过程中的设备，如果现在有可使用的GPU，那就使用硬件设备上的第一块GPU，如果没有就使用CPU设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))

#定义数据预处理函数
data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),   #key=train，就返回一系列对训练集进行预处理的操作，堆积裁剪
                                                                      #裁剪成统一大小，224×224
                                     transforms.RandomHorizontalFlip(),     #水平方向随机翻转
                                     transforms.ToTensor(),         #转化成tensor
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),   #标准化处理
    "val": transforms.Compose([transforms.Resize((224, 224)),  # cannot 224, must (224, 224)    #验证集，resize为224×224
                                   transforms.ToTensor(),      #转化成tensor
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}     #标准化处理



data_root = os.path.abspath(os.path.join(os.getcwd(), ".."))  # get data root path获取数据集所在的根目录，os.getcwd()函数的作用：获取
                                                                  #当前文件的目录，os.path.join作用：将后边的两个路径连接在一起
                                                                  #..返回上一层目录，../..返回上上层目录
image_path = data_root+"/data_set/flower_data/"    # flower data set path找到图片的路径
train_dataset = datasets.ImageFolder(root=image_path + "train",    #通过datasets.ImageFolder函数去加载数据集，root=image_path+"train"
                                                                    #root是训练集的路径
                                    transform=data_transform["train"]) #transform采用数据预处理，上边定义的data_transform，传入的key等于
                                                                 #"train"，则进行一系列的对训练集进行预处理的操作
train_num = len(train_dataset)     #通过len函数打印训练集的图片个数

# {'daisy':0, 'dande lion':1, 'roses':2, 'sunflower':3, 'tulips':4}
flower_list = train_dataset.class_to_idx   #获取分类的名称所对应的索引
cla_dict = dict((val, key) for key, val in flower_list.items())    #遍历刚才所得到的字典flower_list，将它的key和value位置互换
                                                                   #键值变成0，value变成daisy
# write dict into json file
json_str = json.dumps(cla_dict, indent=4)     #将cla_dict转换成json格式
with open('class_indices.json', 'w') as json_file:    #将其保存到json文件中
    json_file.write(json_str)

batch_size = 32        #定义batch大小
#加载训练集
train_loader = torch.utils.data.DataLoader(train_dataset,    #通过DataLoader函数将载入的train_dataset加载进来
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=0)     #num_workers在windows系统中只能设置为0

#加载验证集（测试集）
validate_dataset = datasets.ImageFolder(root=image_path + "val",    #通过datasets.ImageFolder函数去加载测试集，root=image_path+"train"
                                                                    #root是训练集的路径
                                            transform=data_transform["val"])
val_num = len(validate_dataset)       #统计测试集文件个数
validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=True,
                                                  num_workers=0)    #DataLoader载入测试集

print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

# #随机查看图片
# test_data_iter = iter(validate_loader)
# # test_image, test_label = test_data_iter.next()
# test_image, test_label = next(test_data_iter)
#
# def imshow(img):
#     img = img / 2 + 0.5  # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()
#
# print(' '.join('%5s' % cla_dict[test_label[j].item()] for j in range(4)))
# imshow(utils.make_grid(test_image))

#模型实例化，AlexNet的第一个参数，即分类的类别个数为5，初始化权重为True
net = AlexNet(num_classes=5, init_weights=True)

net.to(device)     #将网络指定到规定的设备上


loss_function = nn.CrossEntropyLoss()    #定义损失函数，针对多分类问题
# pata = list(net.parameters())    #调试用，查看模型参数
optimizer = optim.Adam(net.parameters(), lr=0.0002)   #定义Adam优化器，优化对象是网络中所有可以训练的参数，学习率是0.0002


save_path = './AlexNet.pth'    #给定保存权重的路径
best_acc = 0.0      #定义了一个最佳准确率

#开始进行训练
for epoch in range(10):    #将训练集迭代10次
    # train
    net.train()   #在训练过程中使用net.train就会启用dropout
    running_loss = 0.0       #定义running_loss变量，用来累加训练过程中的损失
    t1 = time.perf_counter()     #为了统计训练一个epoch所需的时间，训练结束时间-训练开始时间
    for step, data in enumerate(train_loader,start=0):     #通过该训练，遍历训练集样本
        images, labels = data      #将数据分为图像和对应的标签
        optimizer.zero_grad()      #清空之前的梯度信息
        outputs = net(images.to(device))     #将训练图像指定到设备中，并将图像放置在网络中进行正向传播得到输出
        loss = loss_function(outputs, labels.to(device))     #通过定义的损失函数loss_function来计算预测值和真实值的损失
                                                             #将labels指定到设备当中
        loss.backward()          #将计算得到的损失反向传播到每个节点上
        optimizer.step()         #通过optimizer更新每个节点的参数

        # print statistics
        running_loss += loss.item()     #将loss的值累加到running_loss中
        #print train process打印训练过程中的训练进度
        rate = (step + 1) /len(train_loader)
        a = "*"*int(rate * 50)
        b = "."*int((1-rate) * 50)
        print("\rtrain loss:{:^3.0f}%[{}->{}]{:.3f}".format(int(rate*100),a,b,loss),end="")
    print()
    print(time.perf_counter()-t1)

    #validate在训练完一轮之后进行验证（或者可以理解为测试）
    net.eval()      #在测试过程中使用net.eval就不会启用dropout
    acc = 0.0  # accumulate accurate number / epoch
    with torch.no_grad():      #通过with torch.no_grad这个函数来禁止pytorch对参数进行跟踪，在验证过程中不计算损失梯度
        for data_test in validate_loader:   #上面将测试集载入到了validate_loader中
            test_images, test_labels = data_test    #将测试集划分为图片和对应的标签
            outputs = net(test_images.to(device))   #将图片指定到设备上，传到网络中，进行正向传播，得到输出
            predict_y = torch.max(outputs, dim=1)[1] #将输出的最大值作为预测，寻找输出网络预测最可能归为哪一类
            acc += torch.eq(predict_y, test_labels.to(device)).sum().item()
            #将预测值与真实的标签进行对比，若二者一致，则为1，二者不一致则为0，将其进行求和，实际就是预测正确的个数，放在acc中，用acc
            #变量和累计验证集中预测正确的样本个数
        accurate_test = acc/val_num    #验证正确个数/总验证个数，得到测试集（验证集）的准确率
        if accurate_test > best_acc:   #最开始将最优准确率设置为了0
            best_acc = accurate_test   #将当前的准确率赋给best_acc
            torch.save(net.state_dict(), save_path)   #保存当前的权重，打印相应的信息
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / step, acc/val_num))

print('Finished Training')



