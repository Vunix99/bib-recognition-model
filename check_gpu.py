import torch
print(torch.cuda.is_available())  # Harusnya output: True
print(torch.cuda.get_device_name(0))  # Menampilkan nama GPU

