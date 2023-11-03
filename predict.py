import torch
from model import AlexNet
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import json
import os

data_transform = transforms.Compose(    #定义一个图像预处理函数data_transform，用来对载入的图片进行预处理
     [transforms.Resize((224, 224)),    #resize操作，将其缩放到224×224大小
     transforms.ToTensor(),             #将其转化为tensor
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])    #标准化处理

# load image
img = Image.open("../tulip.jpg")   #使用python的PIL库直接载入图像
plt.imshow(img)      #对其进行展示
#[N, C, H, W]
img = data_transform(img)   #通过上边定义的data_transform函数对所要预测的图像进行预处理
# expand batch dimension
img = torch.unsqueeze(img, dim=0)   #因为载入的图片只有三个维度，要给它增加一个维度，【batch，channel，，，】

# read class_indict
try:
     json_file = open('./class_indices.json','r')   #读取刚才保存的json文件，即索引对应的类别名臣
     class_indict = json.load(json_file)       #对其进行解码，解成所需要的字典的形式
except Exception as e:
     print(e)
     exit(-1)



#create model初始化网络
model = AlexNet(num_classes=5)

# load model weights载入网络模型，并将刚才保存的权重文件载入到该模型的实例化model中
weights_path = "./AlexNet.pth"
model.load_state_dict(torch.load(weights_path))
model.eval()    #进行eval模式，关闭掉dropout

with torch.no_grad():      #通过with torch.no_grad方式让pytorch不去跟踪变量的损失梯度
        # predict class
        output = torch.squeeze(model(img))     #将数据通过model进行正向传播得到输出，并对输出进行压缩，对batch进行压缩，只剩3个维度，
        predict = torch.softmax(output, dim=0)   #经过softmax处理后变成正态分布
        predict_cla = torch.argmax(predict).numpy()   #torch.argmax获取概率最大时对应的索引值
print(class_indict[str(predict_cla)],predict[predict_cla].item())    #打印类别名称及对应的概率
plt.show()

