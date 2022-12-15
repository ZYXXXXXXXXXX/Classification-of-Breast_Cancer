from torch.utils.data import Dataset
import pandas as pd
import torch


class dataloader(Dataset):
    def __init__(self, dataDir):
        self.data = pd.read_csv(dataDir)

    def __getitem__(self, idx):
        # 将数据转换成二维Tensor，并且选择asv的某一列进行提取
        x_tensor1 = torch.Tensor(self.data['radius_mean'].to_list()).reshape(-1, 1) / 2
        x_tensor2 = torch.Tensor(self.data['area_mean'].to_list()).reshape(-1, 1) / 200
        x_tensor3 = torch.Tensor(self.data['concavity_mean'].to_list()).reshape(-1, 1) * 20
        x_tensor4 = torch.Tensor(self.data['concave points_mean'].to_list()).reshape(-1, 1) * 20

        x_tensor5 = torch.Tensor(self.data['radius_worst'].to_list()).reshape(-1, 1) / 2
        x_tensor6 = torch.Tensor(self.data['area_worst'].to_list()).reshape(-1, 1) / 200
        x_tensor7 = torch.Tensor(self.data['concavity_worst'].to_list()).reshape(-1, 1) * 20
        x_tensor8 = torch.Tensor(self.data['concave points_worst'].to_list()).reshape(-1, 1) * 20

        # 标签数据
        letter_array = self.data['diagnosis'].to_list()
        good_worse = []  # 0,1结果数组
        for letter_data in letter_array:
            if letter_data == 'B':  # 如果是B的话
                good_worse.append(0)
            else:
                good_worse.append(1)

        label_tensor = torch.Tensor(good_worse).reshape(-1, 1)

        combine_data = torch.cat((x_tensor1[idx], x_tensor2[idx], x_tensor3[idx], x_tensor4[idx], x_tensor5[idx],
                                  x_tensor6[idx], x_tensor7[idx], x_tensor8[idx]), dim=0)
        # combine_data = z_tensor[idx]
        return combine_data, label_tensor[idx]

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    myTrainData = dataloader("data/origin_breast_cancer_data.csv")
    x_data, labels = myTrainData.__getitem__(1)
    print(x_data)
    print(labels)
