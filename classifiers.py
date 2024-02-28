import torch
from torch.nn.parameter import Parameter

class linear_classifier(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(linear_classifier, self).__init__()
        self.input_dim  = input_dim
        self.output_dim = output_dim
        self.fc = torch.nn.Linear(self.input_dim, self.output_dim)

    def forward(self, x):
        output = self.fc(x)
        return output

class mlp_classifier(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(mlp_classifier, self).__init__()
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.fc1 = torch.nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = torch.nn.Linear(self.hidden_dim, self.output_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        y = self.fc1(x)
        y = self.relu(y)
        output = self.fc2(y)
        return output

class mlp_classifier6(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(mlp_classifier6, self).__init__()
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.fc1 = torch.nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc4 = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc5 = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc6 = torch.nn.Linear(self.hidden_dim, self.output_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        y = self.fc1(x)
        y = self.relu(y)
        y = self.fc2(x)
        y = self.relu(y)
        y = self.fc3(x)
        y = self.relu(y)
        y = self.fc4(x)
        y = self.relu(y)
        y = self.fc5(x)
        y = self.relu(y)
        output = self.fc6(y)
        return output