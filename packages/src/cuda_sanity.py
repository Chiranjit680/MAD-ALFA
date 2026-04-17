import torch
device = torch.device("cuda:1")
print(torch.cuda.get_device_name(device))