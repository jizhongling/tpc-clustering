from __future__ import print_function
import argparse
import os
import numpy as np
import uproot as ur
import matplotlib.pyplot as plt
from matplotlib import colors
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader, random_split


class Data(Dataset):
    def __init__(self, args, files):
        branch = ["adc", "layer", "ztan"]
        if args.type == 0:
            branch += ["ntruth", "nreco"]
        else:
            branch += ["truth_phi", "truth_z", "truth_edep", "reco_phi", "reco_z", "reco_adc"]
        tree = ur.concatenate(files, branch, library='np')
        if args.type == 0:
            self.target = torch.from_numpy(tree["ntruth"])[:, 5:].type(torch.int64).sum(dim=1).clamp(max=5)
            self.comp = torch.from_numpy(tree["nreco"])[:, 2:].type(torch.int64).sum(dim=1).clamp(max=5)
            batch_ind = torch.arange(len(self.target))
        else:
            e = torch.from_numpy(tree["truth_edep"])
            ind = torch.where(e > 4e-7)
            batch_ind = torch.where(ind[0].bincount() == args.type)[0]
            batch_si = torch.tensordot(batch_ind, torch.ones(args.type, dtype=torch.int64), dims=0).flatten()
            si = e[batch_ind].argsort(dim=1, descending=True)[:, :args.type].flatten()
            phi = torch.from_numpy(tree["truth_phi"])[batch_si, si].view(-1, args.type).type(torch.float32)
            z = torch.from_numpy(tree["truth_z"])[batch_si, si].view(-1, args.type).type(torch.float32)
            self.target = torch.stack((phi, z), dim=1)
            si = torch.from_numpy(tree["reco_adc"])[batch_ind].argsort(dim=1, descending=True)[:, :args.type].flatten()
            phi = torch.from_numpy(tree["reco_phi"])[batch_si, si].view(-1, args.type).type(torch.float32)
            z = torch.from_numpy(tree["reco_z"])[batch_si, si].view(-1, args.type).type(torch.float32)
            self.comp = torch.stack((phi, z), dim=1)
        if args.use_conv:
            adc = torch.from_numpy(tree["adc"])[batch_ind].view(-1, 11, 11).type(torch.float32).sub(75).clamp(min=0)
            layer = torch.tensordot(torch.from_numpy(tree["layer"])[batch_ind].type(torch.float32), torch.ones(11, 11), dims=0)
            ztan = torch.tensordot(torch.from_numpy(tree["ztan"])[batch_ind].type(torch.float32), torch.ones(11, 11), dims=0)
            self.input = torch.stack((adc, layer, ztan), dim=1)
        else:
            adc = torch.from_numpy(tree["adc"])[batch_ind].type(torch.float32).sub(75).clamp(min=0)
            layer = torch.from_numpy(tree["layer"])[batch_ind].unsqueeze(1).type(torch.float32)
            ztan = torch.from_numpy(tree["ztan"])[batch_ind].unsqueeze(1).type(torch.float32)
            self.input = torch.cat((adc, layer, ztan), dim=1)

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        input = self.input[idx]
        target = self.target[idx]
        comp = self.comp[idx]
        return input, target, comp


class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.type = args.type
        self.use_conv = args.use_conv
        if self.type == 0:
            nout = 6
        else:
            nout = 2 * self.type
        if self.use_conv:
            print("\nUsing convolutional neural network\n")
            self.conv1 = nn.Conv2d(3, 32, 3, 1)
            self.conv2 = nn.Conv2d(32, 64, 3, 1)
            self.dropout1 = nn.Dropout(0.25)
            self.dropout2 = nn.Dropout(0.5)
            self.fc1 = nn.Linear(64*16, 128)
            self.fc2 = nn.Linear(128, nout)
            self.norm1 = nn.BatchNorm2d(32)
            self.norm2 = nn.BatchNorm2d(64)
            self.norm3 = nn.BatchNorm1d(128)
        else:
            print("\nUsing linear neural network\n")
            self.fc1 = nn.Linear(11*11+2, 10)
            self.fc2 = nn.Linear(10, 10)
            self.fc3 = nn.Linear(10, nout)
            self.norm1 = nn.BatchNorm1d(10)
            self.norm2 = nn.BatchNorm1d(10)

    def forward(self, x):
        if self.use_conv:
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
        else:
            x = self.fc1(x)
            x = self.norm1(x)
            x = torch.sigmoid(x)
            x = self.fc2(x)
            x = self.norm2(x)
            x = torch.sigmoid(x)
            x = self.fc3(x)
        if self.type == 0:
            output = F.log_softmax(x, dim=1)
        else:
            output = x.view(-1, 2, self.type)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target, _) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        if args.type == 0:
            # mean batch loss for each element
            loss = F.nll_loss(output, target)
        else:
            # mean batch loss for each element
            loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                epoch + 1, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader, h_diff, h_comp, savenow):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target, comp in test_loader:
            data, target, comp = data.to(device), target.to(device), comp.to(device)
            output = model(data)
            if args.type == 0:
                # sum up batch loss for all elements
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                # get the index of the max log-probability
                pred = output.argmax(dim=1)
                refs = target.view_as(pred)
                correct += pred.eq(refs).sum().item()
                if args.print:
                    # make histogram for the difference
                    diff = pred.sub(refs).detach().cpu().numpy()
                    hist, bin_edges = np.histogram(diff, bins=9, range=(-4.5, 4.5))
                    h_diff = np.add(h_diff, hist)
                    diff = comp.sub(refs).detach().cpu().numpy()
                    hist, bin_edges = np.histogram(diff, bins=9, range=(-4.5, 4.5))
                    h_comp = np.add(h_comp, hist)
                    if savenow:
                        savenow = False
                        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                        plt.clf()
                        plt.plot(bin_centers, h_diff, drawstyle='steps-mid', label='NN')
                        plt.plot(bin_centers, h_comp, drawstyle='steps-mid', label='reco')
                        plt.legend()
                        plt.xlabel(r"$\Delta N$")
                        plt.savefig("save/diff_ntruth.png")
                        print("\nFigure saved to save/diff_ntruth.png")
            else:
                # sum up batch loss for all elements
                test_loss += F.mse_loss(output, target, reduction='sum').item()
                if args.print:
                    # make histogram for the difference
                    diff = output.sub(target).detach().cpu().numpy()
                    diff_comp = comp.sub(target).detach().cpu().numpy()
                    for itype in range(args.type):
                        hist, xedges, yedges = np.histogram2d(diff[:, 0, itype], diff[:, 1, itype],
                                                              bins=50, range=[[-5, 5], [-5, 5]])
                        h_diff[itype] = np.add(h_diff[itype], hist)
                        hist, xedges, yedges = np.histogram2d(diff_comp[:, 0, itype], diff_comp[:, 1, itype],
                                                              bins=50, range=[[-5, 5], [-5, 5]])
                        h_comp[itype] = np.add(h_comp[itype], hist)
                    if savenow:
                        savenow = False
                        plt.clf()
                        fig, (axs1, axs2) = plt.subplots(2, args.type, figsize=(12, 12))
                        if args.type == 1:
                            axs1 = np.expand_dims(axs1, axis=0)
                            axs2 = np.expand_dims(axs2, axis=0)
                        for itype in range(args.type):
                            im = axs1[itype].imshow(h_diff[itype], interpolation='nearest', origin='lower',
                                                    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                                                    norm=colors.LogNorm())
                            fig.colorbar(im, ax=axs1[itype], fraction=0.047)
                            axs1[itype].set(xlabel=r"$\Delta\phi$ (binsize)")
                            axs1[itype].set(ylabel=r"$\Delta z$ (binsize)")
                            axs1[itype].set(title=f"NN: No. {itype + 1} highest energy")
                            im = axs2[itype].imshow(h_comp[itype], interpolation='nearest', origin='lower',
                                                    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                                                    norm=colors.LogNorm())
                            fig.colorbar(im, ax=axs2[itype], fraction=0.047)
                            axs2[itype].set(xlabel=r"$\Delta\phi$ (binsize)")
                            axs2[itype].set(ylabel=r"$\Delta z$ (binsize)")
                            axs2[itype].set(title=f"reco: No. {itype + 1} highest energy")
                        fig.tight_layout()
                        plt.savefig("save/diff_position.png")
                        print("\nFigure saved to save/diff_position.png")

    # mean batch loss for each element
    test_loss /= len(test_loader.dataset) * len(test_loader.dataset[0][1].flatten())

    print("\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return h_diff, h_comp


def main():
    # training settings
    parser = argparse.ArgumentParser(description='sPHENIX TPC clustering')
    parser.add_argument('--type', type=int, default=0, metavar='N',
                        help='training type (ntruth: 0, position > 0, default: 0)')
    parser.add_argument('--use-conv', action='store_true', default=False,
                        help='use convolutional neural network')
    parser.add_argument('--nfiles', type=int, default=10000, metavar='N',
                        help='max number of files used for training (default: 10000)')
    parser.add_argument('--data-size', type=int, default=3, metavar='N',
                        help='number of files for each training (default: 3)')
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

    if args.type == 0:
        h_diff = np.zeros(9, dtype=np.int64)
        h_comp = np.zeros(9, dtype=np.int64)
    else:
        h_diff = []
        h_comp = []
        for _ in range(args.type):
            h_diff.append(np.zeros((50, 50), dtype=np.int64))
            h_comp.append(np.zeros((50, 50), dtype=np.int64))

    epochs_trained = 0
    sets_trained = 0
    model = Net(args).to(device)
    if args.load_model:
        print("\nLoading model\n")
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
    for epoch in range(epochs_trained, args.epochs):
        for iset in range(sets_trained, nfiles, args.data_size):
            ilast = min(iset + args.data_size, nfiles)
            savenow = (ilast - sets_trained) / args.data_size % args.save_interval == 0 or ilast == nfiles
            print(f"\nDataset: {iset + 1} to {ilast}\n")
            dataset = Data(args, files[iset:ilast])
            nevents = len(dataset)
            ntrain = int(nevents*0.9)
            ntest = nevents - ntrain
            train_set, test_set = random_split(dataset, [ntrain, ntest])
            train_loader = DataLoader(train_set, **train_kwargs)
            test_loader = DataLoader(test_set, **test_kwargs)
            train(args, model, device, train_loader, optimizer, epoch)
            h_diff, h_comp = test(args, model, device, test_loader, h_diff, h_comp, savenow)
            scheduler.step()
            if args.save_checkpoint and savenow:
                print("\nSaving checkpoint\n")
                torch.save({'epochs_trained': epoch + 1 if ilast == nfiles else epoch,
                            'sets_trained': 0 if ilast == nfiles else ilast,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            }, "save/checkpoint")

    if args.save_model:
        model.cpu()
        model.eval()
        if args.use_conv:
            example = torch.randn(1, 3, 11, 11)
        else:
            example = torch.randn(1, 11*11+2)
        traced_script_module = torch.jit.trace(model, example)
        traced_script_module.save("save/net_model.pt")
        torch.save(model.state_dict(), "save/net_weights.pt")
        print("\nSaved model to save/net_model.pt\n")


if __name__ == '__main__':
    main()