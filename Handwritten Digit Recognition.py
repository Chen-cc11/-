import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

##数据加载和预处理
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,),(0.3081,))
                                ])
#引入transforms模块，用于数据预处理。
#ToTensor()：转换图像数据类型为张量图像和缩放数据范围
#Normalize()对图像进行标准化处理，设置全局平均亮度和标准偏差，使得数据分布更加接近标准正态分布，更有助于网络学习和收敛

train_dataset = datasets.MNIST('./data',train=True,download=True,transform=transform)
# 下载并加载MNIST训练数据集
# './data'：指定数据集下载（如果需要）和存储的路径。
# train=True：指示下载/加载的是训练数据集。
# download=True：如果数据集未在指定路径下，则下载数据集。
# transform=transform：应用上面定义的预处理操作（ToTensor和Normalize）。

test_dataset = datasets.MNIST('./data',train=False,transform=transform)
# 加载MNIST测试数据集，与训练集相似，但设置train=False来指定加载的是测试集。

train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=64,shuffle=True)
# 使用DataLoader来封装训练数据集，便于批量加载和打乱数据顺序。
# train_dataset：指定要加载的数据集。
# batch_size=64：指定每个批次加载多少样本。
# shuffle=True：在每个训练周期开始时，打乱数据的顺序。

test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=1000,shuffle=False)
# 为测试数据集创建一个DataLoader，用于在评估模型时批量加载数据。
# test_dataset：指定要加载的测试数据集。
# batch_size=1000：在测试时可以使用更大的批次，因为不需要进行反向传播，计算需求较小。
# shuffle=False：在测试模型时，通常不需要打乱数据。

class Net(nn.Module):#定义一个神经网络类Net，它继承自nn.Module，nn.Module是所有神经网络模块的基类。
    def __init__(self):# 构造函数定义了模型的层次结构。
        super(Net,self).__init__()# 调用父类nn.Module的构造函数。
        self.fc1 = nn.Linear(28*28,512)
        # 定义第一个全连接层（fc1），输入特征的维度是28*28（即MNIST图像的像素数），输出特征的维度是512。
        # 这意味着这一层会将展平的图像（一维数组）转换为512维的隐藏层向量。
        self.fc2 = nn.Linear(512, 10)
        # 定义第二个全连接层（fc2），它接收第一个全连接层的512维输出作为输入，输出维度是10。
        # 输出层的10个节点对应于10个数字类别（0到9），每个节点输出该类别的原始分数（未归一化的概率）

    def forward(self,x): # 定义模型的前向传播路径。x是输入数据。
        x = x.view(-1,28*28)
        # 首先，将输入x的形状变换为一维数组，以匹配第一个全连接层的输入要求。
        # x.view(-1, 28*28)的意思是自动计算批次维度，确保每个样本都被展平为28*28的向量
        x = F.relu(self.fc1(x))
        # 通过第一个全连接层后，使用ReLU激活函数进行非线性变换。
        # ReLU函数将所有负值设为0，保留正值，这样可以增加模型的非线性，帮助捕捉复杂的关系。
        x = self.fc2(x)
        # 经过第二个全连接层，得到每个类别的原始分数。
        return F.log_softmax(x, dim=1)
        # 使用log_softmax函数对输出层的原始分数进行归一化，转换为对数概率。
        # 这样做的好处是在计算交叉熵损失时数值更稳定。
        # dim=1指定在哪个维度上进行softmax运算，这里是在每个样本的10个输出类别上进行。

#初始化模型和优化器
model = Net()
#创建了Net类的一个实例，即初始化了一个具体的神经网络模型。这个Net类之前被定义为包含至少两个全连接层的简单网络，
# 用于处理例如图像分类的任务。创建模型实例是准备训练和使用该模型的第一步。
optimizer = optim.SGD(model.parameters(),lr=0.01,momentum=0.5)
#创建了一个优化器对象，它将用于训练过程中更新模型的权重。选择SGD（随机梯度下降）作为优化算法是因为它简单
# 且广泛应用于各种深度学习任务中。通过调整lr（学习率）和momentum（动量）这两个超参数，
# 可以控制训练过程的速度和稳定性。学习率决定了每次参数更新的幅度大小，
# 而动量则帮助优化器在正确的方向上更快地移动，同时减少震荡

#这两步定义了模型的结构和如何更新模型的策略。

#定义训练模型
def train(model,device,train_loader,optimizer,epoch):
# 定义训练函数，接收模型、设备、数据加载器、优化器和当前的训练轮次作为参数。
    model.train()
# 将模型设置为训练模式。在训练模式下，某些层如Dropout层和BatchNorm层的行为会与评估模式不同。
    for batch_idx,(data,target) in enumerate(train_loader):
# 遍历数据加载器中的每个批次。enumerate(train_loader)返回每个批次的索引（batch_idx）和数据。
        data, target = data.to(device), target.to(device)
# 将数据和目标值发送到计算设备（如CPU或GPU）。这是为了支持在GPU上进行加速计算。
        optimizer.zero_grad()
# 在每次的训练迭代开始之前清零所有被优化的变量的梯度。这是因为默认情况下梯度是累加的。
        output = model(data)
# 通过模型传递数据获取输出结果。
        loss = F.nll_loss(output,target)
# 计算损失（loss），这里使用的是负对数似然损失函数，它常用于多分类问题的训练。
        loss.backward()
# 执行反向传播，计算模型参数的梯度。
        optimizer.step()
# 通过梯度下降法更新模型的参数。
        if batch_idx % 10 == 0:
            print(f'Train Epoch:{epoch}[{batch_idx * len(data)}/{len(train_loader.dataset)}'
                  f'({100.* batch_idx / len(train_loader):.0f}%)]\tLoss:{loss.item():.6f}')


def test(model,device,test_loader):
    # 定义测试函数，接收模型、设备和测试数据加载器作为参数。
    model.eval()
    # 将模型设置为评估模式。这样可以关闭Dropout和BatchNorm等对模型有特定影响的层。
    test_loss = 0
    correct = 0

    with torch.no_grad():
    #torch.no_grad()表示接下来的操作不需要计算梯度，也不会进行反向传播，用于减少计算和内存消耗。
    #with语句在Python中是一种处理资源和上下文管理的优雅方式，而在PyTorch中，torch.no_grad()
    #是一个特别有用的上下文管理器，用于优化模型的评估和推理过程。
        for data,target in test_loader:# 遍历测试数据加载器中的所有批次。
            data,target = data.to(device),target.to(device)
            # 将数据和目标值发送到计算设备（如CPU或GPU）。
            output = model(data)
            # 通过模型获取对这批数据的输出预测。
            test_loss += F.nll_loss(output,target,reduction='sum').item()
            # 累加这批数据的损失。F.nll_loss是负对数似然损失，用于多分类问题。
            # reduction='sum'意味着对所有样本的损失求和。
            pred = output.argmax(dim=1,keepdim=True)
            # 获取预测的类别（概率最高的类别）。argmax(dim=1)在每行中找到值最大的元素，即预测的类别。
            correct += pred.eq(target.view_as(pred)).sum().item()
            # 计算正确预测的数量。pred.eq(target.view_as(pred))比较预测和真实标签是否相等，
            # 返回一个布尔值数组，sum().item()将其转换为正确预测的总数。
    test_loss /= len(test_loader.dataset)
    # 计算平均损失。将累加的损失除以测试数据集的总大小。
    print(f'\nTest set:Average loss:{test_loss:.4f},Accuracy:{correct}/{len(test_loader.dataset)}'
          f'({100.*correct / len(test_loader.dataset):})')

#指定设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 检查是否有可用的GPU（CUDA设备），如果有，则使用GPU；否则，使用CPU。
# GPU可以大幅加速模型的训练过程。
model.to(device)
# 将模型（model）转移到指定的设备（GPU或CPU）上。
# 这是为了确保模型的所有计算都在同一设备上进行，以避免数据在设备间频繁移动造成的性能损失。
for epoch in range(1,10):
# 开始训练模型。这里的循环是按照epoch进行的，即整个训练集被重复使用的次数。
# 一个epoch意味着每个训练样本（整个训练集）都已被模型看过一次。
# 迭代9次，即训练模型9个epoch。
# 训练模型的函数。这个函数需要自己定义，通常包括前向传播、计算损失、反向传播和参数更新等步骤。
# - model: 要训练的模型。
# - device: 模型和数据需要运行的设备（CPU或GPU）。
# - train_loader: 一个加载训练数据的迭代器，能够批量地提供训练数据。
# - optimizer: 优化器，用于根据计算得到的梯度更新模型的参数。
# - epoch: 当前的训练轮次。
    train(model,device,train_loader,optimizer,epoch)
# 训练模型的函数。这个函数需要自己定义，通常包括前向传播、计算损失、反向传播和参数更新等步骤。
    # - model: 要训练的模型。
    # - device: 模型和数据需要运行的设备（CPU或GPU）。
    # - train_loader: 一个加载训练数据的迭代器，能够批量地提供训练数据。
    # - optimizer: 优化器，用于根据计算得到的梯度更新模型的参数。
    # - epoch: 当前的训练轮次。
    test(model,device,test_loader)
    # 在每个epoch结束后，使用测试集评估模型的性能。
    # 这个函数也需要自己定义，通常包括前向传播和计算评估指标（如准确率）等步骤。
    # 注意，测试过程中不应该进行梯度计算和参数更新。
    # - model: 被测试的模型。
    # - device: 模型和数据运行的设备。
    # - test_loader: 一个加载测试数据的迭代器，能够批量地提供测试数据。






