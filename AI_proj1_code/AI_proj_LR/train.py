from decimal import Decimal

import torch
from torch import nn
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter
from dataloader import dataloader
from LogisticRegression import LogisticRegression
import matplotlib.pyplot as plt

total_data = dataloader(r"data/origin_breast_cancer_data.csv")
train_size = int(0.8 * len(total_data))
test_size = len(total_data) - train_size
myTrainData, myTestData = torch.utils.data.random_split(total_data, [train_size, test_size])
# myTrainData = dataloader("data/origin_breast_cancer_data.csv")
# myTestData = dataloader("data/origin_breast_cancer_data.csv")

train_batch_size = 20
epoch = 200
train_data_length = 596


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cuda = True if torch.cuda.is_available() else False

myTrainDataLoader = DataLoader(dataset=myTrainData, batch_size=train_batch_size, shuffle=True)
myTestDataLoader = DataLoader(dataset=myTestData, batch_size=train_batch_size, shuffle=True)

# 网络选择
myModel = LogisticRegression().to(device)
print(myModel)

# 损失函数,Binary crossEntropy loss
loss_fn = nn.BCELoss(size_average=False)

# 学习率
learning_rate = 1e-3
optimizer = torch.optim.SGD(myModel.parameters(), lr=learning_rate)
# descending learning rate
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 180], gamma=0.1)

total_train_step = 0
total_test_step = 0
step = 0

writer = SummaryWriter("logs")
train_loss_his = []
test_totalloss_his = []
for i in range(epoch):
    print(f"-------第{i}轮训练开始-------")
    # 这一部分是模型训练

    i_total_trainloss = []
    for data in myTrainDataLoader:
        x, y = data

        output = myModel(x)

        loss = loss_fn(output, y)

        optimizer.zero_grad()
        # 反向传播，计算梯度
        loss.backward()
        optimizer.step()
        total_train_step = total_train_step + 1
        train_loss_his.append(loss.detach().numpy())
        # 将当前的LOSS记录到tensorboard的中
        writer.add_scalar("train_loss", loss.item(), total_train_step)
        # print(f"训练次数：{total_train_step}，loss:{loss}")
        i_total_trainloss.append(loss)

    total_test_loss = 0
    # 计算F1 Score
    TP_num = 0
    TN_num = 0
    FP_num = 0
    FN_num = 0
    with torch.no_grad():
        for data in myTestDataLoader:
            test_x, test_y = data
            test_output = myModel(test_x)
            test_loss = loss_fn(test_output, test_y)
            # 这里求一个epoch的总loss
            total_test_loss = total_test_loss + test_loss

            # 计算F1_score accuracy
            for test_step in range(len(test_y)):

                if int(test_y[test_step].item()) == round(float(test_output[test_step].item())):
                    if int(test_y[test_step].item()) == 1:
                        TP_num = TP_num + 1
                    else:
                        TN_num = TN_num + 1
                else:
                    if int(test_y[test_step].item()) == 1:
                        FP_num = FP_num + 1
                    else:
                        FN_num = FN_num + 1
        if TP_num + FN_num != 0:
            Recall = TP_num / (TP_num + FN_num)
        else:
            Recall = 0
        if TP_num + FP_num != 0:
            Precision = TP_num / (TP_num + FP_num)
        else:
            Precision = 0

        Accuracy = (TP_num + TN_num) / (TP_num + TN_num + FP_num + FN_num)

        if Precision + Recall != 0:
            F1_score = 2 * (Precision * Recall) / (Precision + Recall)
        else:
            F1_score = 100

        # print test condition(maintain 4 digits)
        print(f"第{i}轮训练 loss:{sum(i_total_trainloss) / len(i_total_trainloss)}")
        print(f"测试集: loss：{total_test_loss}")
        print(f"测试Accuracy: {Decimal(Accuracy).quantize(Decimal('0.00'))}")
        print(f"测试F1_score: {Decimal(F1_score).quantize(Decimal('0.00'))}")
        print(f"测试Precision: {Decimal(Precision).quantize(Decimal('0.00'))}")
        print(f"测试Recall: {Decimal(Recall).quantize(Decimal('0.00'))}")

        test_totalloss_his.append(total_test_loss.detach().numpy())
        writer.add_scalar("test_loss", total_test_loss.item(), i)

    scheduler.step()

# 输出线性模型的两个参数，分别是权重和偏置
for parameters in myModel.parameters():
    print(parameters)

writer.close()
# 画出训练损失变化曲线
plt.plot(train_loss_his)
plt.show()
# 画出测试损失变化曲线
plt.plot(test_totalloss_his)
plt.show()
