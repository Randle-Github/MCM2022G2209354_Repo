import torch
from torch.nn import Module, LSTM, Linear
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch.autograd as auto
import dataloader


class lstm(Module):
    def __init__(self):
        super(lstm, self).__init__()
        self.lstm = LSTM(input_size=1, hidden_size=128, num_layers=2, batch_first=True)
        self.linear = Linear(in_features=128, out_features=1)

    def forward(self, x):
        """
        input_shape=(batch_size,seq_length,embedding_dim) # (1, n, 1)
        output_shape=(batch_size,seq_length,num_directions*hidden_size) # (1, n, 1)
        """
        lstm_out, hidden = self.lstm(x)
        linear_out = self.linear(lstm_out)
        return linear_out


def train(model, start, end, epoches=100, learning_rate=None, file=None):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()

    for epoch in range(epoches):
        x, y = dataloader.Dataloader(start, end, file)
        x = auto.Variable(x)
        y = auto.Variable(y)
        optimizer.zero_grad()
        pred_y = model(x)
        loss = criterion(pred_y, y)
        loss.backward()
        # print(loss.item())
        optimizer.step()

        x, y = dataloader.Reverse_Dataloader(start, end, file)
        x = auto.Variable(x)
        y = auto.Variable(y)
        optimizer.zero_grad()
        pred_y = model(x)
        loss = criterion(pred_y, y)
        loss.backward()
        optimizer.step()


def predict(model, start, end, t_pred, file=None):
    model.eval()
    # print(end+t_pred)
    x, _ = dataloader.Dataloader(start, end + t_pred, file)
    pred_y1 = model(x).detach()
    x, _ = dataloader.Reverse_Dataloader(start, end + t_pred, file)
    pred_y2 = model(x).detach()
    return ((pred_y1[0, end - start:end - start + t_pred, 0].flatten() + pred_y2[0, end - start:end - start + t_pred,
                                                                         0]).flatten()) / 2
    # return (pred_y1[0, end:end + t_pred, 0]).flatten()


if __name__ == "__main__":
    model = lstm()
    model.load_state_dict(torch.load("model_param.pth"))
    print(model(torch.ones((1, 10, 1))))
    torch.save(obj=model.state_dict(), f="model_param.pth")
