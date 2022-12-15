import torch.nn


class LogisticRegression(torch.nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear1 = torch.nn.Linear(8, 4)
        self.linear2 = torch.nn.Linear(4, 2)
        self.linear3 = torch.nn.Linear(2, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        res_out = self.linear1(x)
        res_out = self.linear2(res_out)
        res_out = self.linear3(res_out)
        res_out = self.sigmoid(res_out)
        return res_out


if __name__ == '__main__':
    model = LogisticRegression()
    input_x = torch.rand(31, 8)
    output = model(input_x)

    # print(output.size())
    print(model)
