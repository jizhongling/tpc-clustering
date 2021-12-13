from __future__ import print_function
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader, random_split

class PosData(Dataset):
    def __init__(self, num):
        self.idx = np.arange(num)
        self.input = []
        self.output = []
        for i in range(num):
            rnd = np.random.normal(size=25)
            self.input.append(np.float32(rnd))
            rnd = np.random.normal(size=2)
            self.output.append(np.float32(rnd))
    
    def __len__(self):
        return len(self.idx)

    def __getitem__(self, idx):
        input = self.input[idx]
        output = self.output[idx]
        return input, output


class PosNet(nn.Module):
    def __init__(self):
        super(PosNet, self).__init__()
        self.fc1 = nn.Linear(25, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 2)
        self.norm1 = nn.BatchNorm1d(10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.fc3(x)
        output = x
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, target)
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
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.mse_loss(output, target, reduction='sum').item()  # sum up batch loss

    test_loss /= len(test_loader.dataset)

    print("\nTest set: Average loss: {:.4f}\n".format(
        test_loss))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
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

    nevents = 64000
    ntrain = int(nevents*0.8)
    ntest = nevents - ntrain
    dataset = PosData(nevents)
    train_set, test_set = random_split(dataset, [ntrain, ntest])
    train_loader = DataLoader(train_set, **train_kwargs)
    test_loader = DataLoader(test_set, **test_kwargs)

    epochs_trained = 0
    model = PosNet().to(device)
    if args.load_model:
        model.load_state_dict(torch.load("save/pos_weights.pt"))
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
        traced_script_module.save("save/pos_model.pt")
        torch.save(model.state_dict(), "save/pos_weights.pt")
        example = torch.ones(1, 25)
        print(model(example))


if __name__ == '__main__':
    main()