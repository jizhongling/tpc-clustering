from __future__ import print_function
import argparse
import numpy as np
import uproot as ur
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader, random_split


class NtruthData(Dataset):
    def __init__(self):
        tree = ur.concatenate("data/training-*.root:T",
                ["adc", "layer", "ztan", "ntruth"], library='np')
        self.n = len(tree["layer"])
        self.input = []
        self.target = []
        for i in range(self.n):
            input = np.concatenate((tree["adc"][i], tree["layer"][i], tree["ztan"][i]), axis=None)
            target = min(tree["ntruth"][i], 3)
            self.input.append(np.single(input))
            self.target.append(np.int_(target))
    
    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        input = self.input[idx]
        target = self.target[idx]
        return input, target


class NtruthNet(nn.Module):
    def __init__(self):
        super(NtruthNet, self).__init__()
        self.fc1 = nn.Linear(11*11+2, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 4)
        self.norm1 = nn.BatchNorm1d(10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm1(x)
        x = F.sigmoid(x)
        x = self.fc2(x)
        x = self.norm1(x)
        x = F.sigmoid(x)
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='sPHENIX TPC clustering')
    parser.add_argument('--batch-size', type=int, default=640, metavar='N',
                        help='input batch size for training (default: 640)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--load-model', action='store_true', default=False,
                        help='For Loading the saved Model')
    parser.add_argument('--save-checkpoint', action='store_true', default=False,
                        help='For Saving the checkpoint')
    parser.add_argument('--load-checkpoint', action='store_true', default=False,
                        help='For Loading the checkpoint')
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

    dataset = NtruthData()
    nevents = len(dataset)
    ntrain = int(nevents*0.8)
    ntest = nevents - ntrain
    train_set, test_set = random_split(dataset, [ntrain, ntest])
    train_loader = DataLoader(train_set, **train_kwargs)
    test_loader = DataLoader(test_set, **test_kwargs)

    epochs_trained = 0
    model = NtruthNet().to(device)
    if args.load_model:
        model.load_state_dict(torch.load("save/ntruth_weights.pt"))
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    if args.load_checkpoint:
        print("\nLoading checkpoint\n")
        checkpoint = torch.load("save/checkpoint")
        epochs_trained = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(epochs_trained + 1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()
        if args.save_checkpoint:
            print("\nSaving checkpoint\n")
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        }, "save/checkpoint")

    if args.save_model:
        model.cpu()
        model.eval()
        example = torch.rand(1, 25)
        traced_script_module = torch.jit.trace(model, example)
        traced_script_module.save("save/ntruth_model.pt")
        torch.save(model.state_dict(), "save/ntruth_weights.pt")
        example = torch.ones(1, 25)
        print(model(example))


if __name__ == '__main__':
    main()