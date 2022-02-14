from __future__ import print_function
import argparse
import os
import numpy as np
import uproot as ur
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader, random_split


class Data(Dataset):
    def __init__(self, files):
        tree = ur.concatenate(files, ["adc", "layer", "ztan", "gedep"], library='np')
        adc = torch.from_numpy(tree["adc"]).view(-1, 21, 21)[:, 7:14, 7:14].type(torch.float32).sub(75).clamp(min=0)
        layer = torch.tensordot(torch.from_numpy(tree["layer"]).type(torch.float32), torch.ones(7, 7), dims=0)
        ztan = torch.tensordot(torch.from_numpy(tree["ztan"]).type(torch.float32), torch.ones(7, 7), dims=0)
        self.input = torch.stack((adc, layer, ztan), dim=1)
        max_ind = torch.from_numpy(tree["gedep"]).view(-1, 21, 21)[:, 7:14, 7:14].flatten(start_dim=1).argmax(dim=1)
        max_col = max_ind.fmod(7)
        max_row = max_ind.sub(max_col).div(7)
        self.target = torch.stack((max_row, max_col), dim=1).type(torch.float32)

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        input = self.input[idx]
        target = self.target[idx]
        return input, target


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64*4, 128)
        self.fc2 = nn.Linear(128, 2)
        self.norm1 = nn.BatchNorm2d(32)
        self.norm2 = nn.BatchNorm2d(64)
        self.norm3 = nn.BatchNorm1d(128)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, ceil_mode=True)
        x = self.dropout1(x)
        x = x.flatten(start_dim=1)
        x = self.fc1(x)
        x = self.norm3(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = x
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # mean batch loss for each element
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader, h2_diff, print_output):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss for all elements
            test_loss += F.mse_loss(output, target, reduction='sum').item()
            diff_ind = output.sub(target).cpu().detach().numpy()
            h2, xedges, yedges = np.histogram2d(diff_ind[:,0], diff_ind[:,1], bins=11, range=[[-5.5, 5.5], [-5.5, 5.5]])
            h2_diff = np.add(h2_diff, h2)
            # get the index of the peak
            output_ind = torch.round(output).clamp(min=0, max=6)
            pred = output_ind[:,0].mul(7).add(output_ind[:,1])
            refs = target[:,0].mul(7).add(target[:,1])
            correct += pred.eq(refs).sum().item()
            if print_output:
                fig = plt.figure(figsize=(11, 11))
                ax = plt.axes()
                ax.set(xlabel=r"$\Delta\phi$ (bins)")
                ax.set(ylabel=r"$\Delta z$ (bins)")
                map = ax.imshow(h2_diff, interpolation='nearest', origin='lower',
                                extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
                fig.colorbar(map, ax=ax)
                plt.savefig("save/diff_ind.png")
                print_output = False

    # mean batch loss for each element
    test_loss /= len(test_loader.dataset) * len(test_loader.dataset[0][1].flatten())

    print("\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return h2_diff, print_output


def main():
    # training settings
    parser = argparse.ArgumentParser(description='sPHENIX TPC clustering')
    parser.add_argument('--nfiles', type=int, default=10000, metavar='N',
                        help='max number of files used for training (default: 10000)')
    parser.add_argument('--data-size', type=int, default=1, metavar='N',
                        help='number of files for each training and testing (default: 1)')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 256)')
    parser.add_argument('--test-batch-size', type=int, default=1024, metavar='N',
                        help='input batch size for testing (default: 1024)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.9, metavar='M',
                        help='learning rate step gamma (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=400, metavar='N',
                        help='how many batches to wait before logging training status (default: 400)')
    parser.add_argument('--save-interval', type=int, default=10, metavar='N',
                        help='how many groups of dataset to wait before saving the checkpoint (default: 10)')
    parser.add_argument('--print', action='store_true', default=False,
                        help='print output')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='save the current model')
    parser.add_argument('--load-model', action='store_true', default=False,
                        help='load the saved model')
    parser.add_argument('--save-checkpoint', action='store_true', default=False,
                        help='save checkpoints')
    parser.add_argument('--load-checkpoint', action='store_true', default=False,
                        help='load the saved checkpoint')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using {device} device")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 4,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    files = []
    for file in os.scandir("data"):
        if (file.name.startswith("training-") and
            file.name.endswith(".root") and
            file.is_file()):
            files.append(file.path + ":T")
    nfiles = min(args.nfiles, len(files))

    epochs_trained = 0
    sets_trained = 0
    h2_diff = np.zeros((11, 11), dtype=np.int64)
    print_output = False
    model = Net().to(device)
    if args.load_model:
        model.load_state_dict(torch.load("save/net_weights.pt"))
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    if args.load_checkpoint:
        print("\nLoading checkpoint\n")
        checkpoint = torch.load("save/checkpoint")
        epochs_trained = checkpoint['epochs_trained']
        sets_trained = checkpoint['sets_trained']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(epochs_trained + 1, args.epochs + 1):
        for iset in range(sets_trained, nfiles, args.data_size):
            ilast = min(iset + args.data_size, nfiles)
            print(f"\nDataset: {iset + 1} to {ilast}\n")
            dataset = Data(files[iset:ilast])
            nevents = len(dataset)
            ntrain = int(nevents*0.9)
            ntest = nevents - ntrain
            train_set, test_set = random_split(dataset, [ntrain, ntest])
            train_loader = DataLoader(train_set, **train_kwargs)
            test_loader = DataLoader(test_set, **test_kwargs)
            train(args, model, device, train_loader, optimizer, epoch)
            h2_diff, print_output = test(model, device, test_loader, h2_diff, print_output)
            scheduler.step()
            if (ilast - sets_trained) / args.data_size % args.save_interval == 0 or ilast == nfiles:
                if args.save_checkpoint:
                    print("\nSaving checkpoint\n")
                    torch.save({'epochs_trained': epoch,
                                'sets_trained': ilast,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                }, "save/checkpoint")
                if args.print:
                    print_output = True

    if args.save_model:
        model.cpu()
        model.eval()
        example = torch.rand(1, 3, 7, 7)
        traced_script_module = torch.jit.trace(model, example)
        traced_script_module.save("save/net_model.pt")
        torch.save(model.state_dict(), "save/net_weights.pt")
        example = torch.ones(1, 3, 7, 7)
        print(model(example))


if __name__ == '__main__':
    main()