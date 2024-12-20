import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, TensorDataset
import torchaudio
# torchtext


from datasets_preparation.titanic import TitanicDataset
from datasets_preparation.uciml import from_uciml_to_dataset
from datasets_preparation.special import get_scipyfunction_dataset, get_special_dataset_1d
from datasets_preparation.pca_dataset import get_pca_dataset
# from datasets_preparation.text import create_text_loader, get_IMDb_dataset
from datasets_preparation.audio import SubsetSC, get_SC_loader, get_US_dataset

def prepare_data(data_str, root, train_batch_size, test_batch_size, arithmetic_len: int = 10000, arithmetic_id: int = None, arithmetic_dim: int = 1, arithmetic_test_shuffle: bool = False):
    if data_str == "arithmetic":
        assert arithmetic_id is not None
        arithmetic_plan = {
            1: f1,
            2: f2,
            3: f3,
            4: f4,
            5: f5,
            6: f6
        }
        comp = arithmetic_plan[arithmetic_id]
        xs = np.random.random([arithmetic_len, arithmetic_dim, 1])
        if arithmetic_test_shuffle:
            np.random.shuffle(xs)
        ys = comp(xs)
        data_tensor = torch.from_numpy(xs).float()
        labels_tensor = torch.from_numpy(ys).float()
        dataset = TensorDataset(data_tensor, labels_tensor)
        subset_train_indices = list(range(int(0.9*arithmetic_len)))
        subset_test_indices = list(range(int(0.9*arithmetic_len),arithmetic_len))
        train_dataset = Subset(dataset, subset_train_indices)
        test_dataset = Subset(dataset, subset_test_indices)
        train_loader = DataLoader(train_dataset, train_batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, test_batch_size, shuffle=False)
        return train_loader, test_loader
    elif data_str == "mnist":
        train_loader, test_loader = prepare_mnist(root, train_batch_size, test_batch_size)
        return train_loader, test_loader
    elif data_str == "fashion":
        train_loader, test_loader = prepare_fashion(root, train_batch_size, test_batch_size)
        return train_loader, test_loader
    elif data_str == "cifar10":
        train_loader, test_loader = prepare_cifar(root, train_batch_size, test_batch_size)
        return train_loader, test_loader
    elif data_str == "svhn":
        train_loader, test_loader = prepare_svhn(root, train_batch_size, test_batch_size)
        return train_loader, test_loader
    
    elif data_str == "titanic":
        train_loader, test_loader = prepare_titanic(train_batch_size, test_batch_size)
        return train_loader, test_loader
    elif data_str== "toxicity":
        train_loader, test_loader = prepare_toxicity(train_batch_size, test_batch_size)
        return train_loader, test_loader
    elif data_str== "speech_command":
        train_loader, test_loader = prepare_speechcommand(train_batch_size, test_batch_size)
        return train_loader, test_loader
    elif data_str ==  "ag_news":
        train_loader, test_loader = prepare_ag_news(train_batch_size, test_batch_size)
        return train_loader, test_loader
    
    elif data_str ==   "bean":
        train_loader, test_loader = prepare_bean(train_batch_size, test_batch_size)
        return train_loader, test_loader
    
    elif data_str ==   "income":
        train_loader, test_loader = prepare_income(train_batch_size, test_batch_size)
        return train_loader, test_loader

        
"""
simple
"""

def f1(x):
    return np.sin(np.pi * x)

def f2(x):
    return np.exp(x)

def f3(x):
    return x * x + x + 1

def f4(x):
    y = np.sin(np.pi * x[:, [0]] + np.pi * x[:, [1]])
    return y

def f5(x):
    y = np.exp(np.sin(np.pi * x[:, [0]]) + x[:, [1]] * x[:, [1]])
    return y

def f6(x):
    y = np.exp(
        np.sin(np.pi * x[:, [0]] * x[:, [0]] + np.pi * x[:, [1]] * x[:, [1]]) +
        np.sin(np.pi * x[:, [2]] * x[:, [2]] + np.pi * x[:, [3]] * x[:, [3]])
    )
    return y

def f7(x):
    return np.sin(5 * np.pi * x) + x

"""
catastrofic forgetting
"""

def gs(x, sigma = 5):
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    x = (x - 0.5) * 2
    return np.multiply(np.power(np.sqrt(2 * np.pi) * sigma, -1), np.exp(-np.power(x, 2) / 2 * sigma ** 2))

def cf():
    xs = np.arange(0, 5000) / 5000
    ys = []
    for i in range(5):
        x = xs[i*1000: (i+1)*1000]
        y = gs(x)
        ys.append(y)
    ys = np.concatenate(ys)
    #plt.plot(xs, ys)
    #plt.show()
    ys = ys / 0.08
    xs = xs.reshape([5000, 1, 1])
    ys = ys.reshape([5000, 1, 1])
    xs = torch.Tensor(xs)
    ys = torch.Tensor(ys)
    return xs, ys

def cf_task(task_i: int):

    xs, ys = cf()

    t_xs = xs[task_i*1000: (task_i+1)*1000]
    t_ys = ys[task_i*1000: (task_i+1)*1000]
    
    return t_xs, t_ys

class OneHotEncodeTransform:
    def __init__(self, num_classes=10):
        self.num_classes = num_classes

    def __call__(self, label):
        return F.one_hot(torch.tensor(label), num_classes=self.num_classes).float()
    
# TODO:



"""
AG_NEWS
"""
def prepare_ag_news(train_batch_size=64, test_batch_size=1024):

        train_dataset = torchtext.datasets.AG_NEWS(root = "../data", split='train')
        train_dataset = [(int(item[0])-1, item[1]) for item in train_dataset]

        test_dataset = torchtext.datasets.AG_NEWS(root = "../dataset", split='test')
        test_dataset = [(int(item[0])-1, item[1]) for item in test_dataset]
        train_kwargs = {
        "batch_size": train_batch_size,
        "shuffle": True}

        test_kwargs = {
        "batch_size": train_batch_size,
        "shuffle": True}

        train_loader, vocab = create_text_loader(train_dataset, None, train_kwargs )
        test_loader, _ = create_text_loader(test_dataset, vocab, test_kwargs)
        # num_classes = 4

"""
SpeechCommand
"""

def prepare_speechcommand(train_batch_size=64, test_batch_size=1024):
#   elif args.dataset == "SpeechCommand":
        ## Without Cache
        train_kwargs = {'batch_size': train_batch_size}
        test_kwargs = {'batch_size': test_batch_size}
        train_dataset = SubsetSC("training")
        test_dataset = SubsetSC("testing")
        train_loader = get_SC_loader(train_dataset, train_kwargs)
        test_loader = get_SC_loader(test_dataset, test_kwargs)
        return train_loader, test_loader
        
# if __name__ == "__main__":
#    train, test = prepare_speechcommand()
#     waveform, sample_rate = torchaudio.load(r"C:\Users\mmiezianko\Ortho-PolyKAN\data\SpeechCommands\speech_commands_v0.02\_background_noise_\doing_the_dishes.wav")
#     print(waveform, sample_rate)
#    print(train)
#    print("done")


"""
income
"""
def prepare_income(train_batch_size=64, test_batch_size=1024):
        train_dataset, test_dataset = from_uciml_to_dataset(2)
        train_loader = DataLoader(train_dataset,batch_size=train_batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset,batch_size=test_batch_size, shuffle=False)

        return train_loader, test_loader
  
      

"""
bean
"""
def prepare_bean(train_batch_size=64, test_batch_size=1024):
        train_dataset, test_dataset = from_uciml_to_dataset(602)
        train_loader = DataLoader(train_dataset,batch_size=train_batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset,batch_size=test_batch_size, shuffle=False)

        return train_loader, test_loader
  
      


"""
toxicity
"""


def prepare_toxicity(train_batch_size=64, test_batch_size=1024):
        train_dataset, test_dataset = from_uciml_to_dataset(728)
        train_loader = DataLoader(train_dataset,batch_size=train_batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset,batch_size=test_batch_size, shuffle=False)

        return train_loader, test_loader
        
    
"""
titanic
"""

def prepare_titanic(train_batch_size=64, test_batch_size=1024):

    train_dataset, test_dataset = TitanicDataset()
    train_loader = DataLoader(train_dataset,batch_size=train_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset,batch_size=test_batch_size, shuffle=False)
        # num_classes = 2
        # input_size = 9

    return train_loader, test_loader


"""
mnist
"""

def prepare_mnist(root="./data", train_batch_size=64, test_batch_size=1024):

    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(root=root, train=True, download=True, transform=transform, target_transform=OneHotEncodeTransform())
    test_dataset = datasets.MNIST(root=root, train=False, download=True, transform=transform, target_transform=OneHotEncodeTransform())

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    return train_loader, test_loader

"""
fashion
"""

def prepare_fashion(root="./data", train_batch_size=64, test_batch_size=1024):

    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.FashionMNIST(root=root, train=True, download=True, transform=transform, target_transform=OneHotEncodeTransform())
    test_dataset = datasets.FashionMNIST(root=root, train=False, download=True, transform=transform, target_transform=OneHotEncodeTransform())

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    return train_loader, test_loader

"""
cifar10
"""

def prepare_cifar(root="./data", train_batch_size=64, test_batch_size=1024):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.CIFAR10(root=root, train=True, download=True, transform=transform, target_transform=OneHotEncodeTransform())
    test_dataset = datasets.CIFAR10(root=root, train=False, download=True, transform=transform, target_transform=OneHotEncodeTransform())

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    return train_loader, test_loader

"""
svhn
"""

def prepare_svhn(root="./data", train_batch_size=64, test_batch_size=1024):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.SVHN(root=root, split="train", download=True, transform=transform, target_transform=OneHotEncodeTransform())
    test_dataset = datasets.SVHN(root=root, split="test", download=True, transform=transform, target_transform=OneHotEncodeTransform())

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    return train_loader, test_loader
