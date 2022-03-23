from __future__ import print_function
import argparse
import sys
import os
import numpy as np
import uproot as ur
import matplotlib.pyplot as plt
from matplotlib import colors
import torch
import torch.linalg as LA
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
        elif args.type == 1:
            branch += ["truth_phi", "truth_z", "truth_adc", "reco_phi", "reco_z", "reco_adc"]
        elif args.type == 2:
            branch += ["truth_phicov", "truth_adc"]
        elif args.type == 3:
            branch += ["truth_zcov", "truth_adc"]
        elif args.type == 4:
            branch += ["truth_adc"]
        else:
            sys.exit("\nError: Wrong type number\n")
        tree = ur.concatenate(files, branch, library='np')

        self.has_comp = False
        if args.type == 0:
            self.target = torch.from_numpy(tree["ntruth"])[:, 5:].type(torch.int64).sum(dim=1).clamp(max=args.nmax)
            self.comp = torch.from_numpy(tree["nreco"])[:, 5:].type(torch.int64).sum(dim=1).clamp(max=args.nmax)
            self.has_comp = True
            batch_ind = torch.arange(len(self.target))
        else:
            gadc = torch.from_numpy(tree["truth_adc"])
            ind = torch.where(gadc > 100)
            batch_ind = torch.where(ind[0].bincount() == args.nout)[0]
            batch_si = batch_ind.unsqueeze(1).expand(-1, args.nout).flatten()
            si = gadc[batch_ind].argsort(dim=1, descending=True)[:, :args.nout].flatten()
            if args.type == 1:
                phi = torch.from_numpy(tree["truth_phi"])[batch_si, si].type(torch.float32).view(-1, args.nout)
                z = torch.from_numpy(tree["truth_z"])[batch_si, si].type(torch.float32).view(-1, args.nout)
                self.target = torch.stack((phi, z), dim=1)
                si = torch.from_numpy(tree["reco_adc"])[batch_ind].argsort(dim=1, descending=True)[:, :args.nout].flatten()
                phi = torch.from_numpy(tree["reco_phi"])[batch_si, si].type(torch.float32).view(-1, args.nout)
                z = torch.from_numpy(tree["reco_z"])[batch_si, si].type(torch.float32).view(-1, args.nout)
                self.comp = torch.stack((phi, z), dim=1)
                self.has_comp = True
            elif args.type == 2:
                self.target = torch.from_numpy(tree["truth_phicov"])[batch_si, si].type(torch.float32).view(-1, args.nout)
            elif args.type == 3:
                self.target = torch.from_numpy(tree["truth_zcov"])[batch_si, si].type(torch.float32).view(-1, args.nout)
            elif args.type == 4:
                self.target = gadc[batch_si, si].type(torch.float32).view(-1, args.nout)

        if args.use_conv:
            adc = torch.from_numpy(tree["adc"])[batch_ind].type(torch.float32).view(-1, 11, 11).sub(75).clamp(min=0)
            layer = torch.from_numpy(tree["layer"])[batch_ind].type(torch.float32).view(-1, 1, 1).expand(-1, 11, 11)
            ztan = torch.from_numpy(tree["ztan"])[batch_ind].type(torch.float32).view(-1, 1, 1).expand(-1, 11, 11)
            self.input = torch.stack((adc, layer, ztan), dim=1)
        else:
            adc = torch.from_numpy(tree["adc"])[batch_ind].type(torch.float32).sub(75).clamp(min=0)
            layer = torch.from_numpy(tree["layer"])[batch_ind].type(torch.float32).unsqueeze(1)
            ztan = torch.from_numpy(tree["ztan"])[batch_ind].type(torch.float32).unsqueeze(1)
            self.input = torch.cat((adc, layer, ztan), dim=1)

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        input = self.input[idx]
        target = self.target[idx]
        if self.has_comp:
            return input, target, self.comp[idx]
        else:
            return input, target


class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.type = args.type
        self.nout = args.nout
        self.use_conv = args.use_conv

        if self.type == 0:
            nout = args.nmax + 1
        elif self.type == 1:
            nout = 2 * self.nout
        elif self.type <= 4:
            nout = self.nout
        else:
            sys.exit("\nError: Wrong type number\n")

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
        elif self.type == 1:
            output = x.view(-1, 2, self.nout)
        elif self.type <= 4:
            output = x.view(-1, self.nout)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, item in enumerate(train_loader):
        data, target = item[0].to(device), item[1].to(device)
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
        for item in test_loader:
            if len(item) >= 3:
                data, target, comp = item[0].to(device), item[1].to(device), item[2].to(device)
            else:
                data, target = item[0].to(device), item[1].to(device)
                args.print = False
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
                    diff = comp.view_as(refs).sub(refs).detach().cpu().numpy()
                    hist, bin_edges = np.histogram(diff, bins=9, range=(-4.5, 4.5))
                    h_comp = np.add(h_comp, hist)
            else:
                # sum up batch loss for all elements
                test_loss += F.mse_loss(output, target, reduction='sum').item()
                # count the number of improved predictions
                diff = output.sub(target)
                try:
                    comp
                except NameError:
                    pass
                else:
                    diff_comp = comp.sub(target)
                    correct += (LA.matrix_norm(diff) < LA.matrix_norm(diff_comp)).sum().item()
                if args.print:
                    # make histogram for the difference
                    diff = diff.detach().cpu().numpy()
                    diff_comp = diff_comp.detach().cpu().numpy()
                    for iout in range(args.nout):
                        hist, xedges, yedges = np.histogram2d(diff[:, 0, iout], diff[:, 1, iout],
                                                              bins=50, range=[[-5, 5], [-5, 5]])
                        h_diff[iout] = np.add(h_diff[iout], hist)
                        hist, xedges, yedges = np.histogram2d(diff_comp[:, 0, iout], diff_comp[:, 1, iout],
                                                              bins=50, range=[[-5, 5], [-5, 5]])
                        h_comp[iout] = np.add(h_comp[iout], hist)

    # mean batch loss for each element
    test_loss /= len(test_loader.dataset) * len(test_loader.dataset[0][1].flatten())

    print("\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    if args.print and savenow:
        plt.clf()
        if args.type == 0:
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            plt.plot(bin_centers, h_diff, drawstyle='steps-mid', label='NN')
            plt.plot(bin_centers, h_comp, drawstyle='steps-mid', label='reco')
            plt.legend()
            plt.xlabel(r"$\Delta N$")
        else:
            fig, (axs1, axs2) = plt.subplots(2, args.nout, figsize=(12, 12))
            if args.nout == 1:
                axs1 = np.expand_dims(axs1, axis=0)
                axs2 = np.expand_dims(axs2, axis=0)
            for iout in range(args.nout):
                im = axs1[iout].imshow(h_diff[iout], interpolation='nearest', origin='lower',
                                        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                                        norm=colors.LogNorm())
                fig.colorbar(im, ax=axs1[iout], fraction=0.047)
                axs1[iout].set(xlabel=r"$\Delta\phi$ (binsize)")
                axs1[iout].set(ylabel=r"$\Delta z$ (binsize)")
                axs1[iout].set(title=f"NN: No. {iout + 1} highest energy")
                im = axs2[iout].imshow(h_comp[iout], interpolation='nearest', origin='lower',
                                        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                                        norm=colors.LogNorm())
                fig.colorbar(im, ax=axs2[iout], fraction=0.047)
                axs2[iout].set(xlabel=r"$\Delta\phi$ (binsize)")
                axs2[iout].set(ylabel=r"$\Delta z$ (binsize)")
                axs2[iout].set(title=f"reco: No. {iout + 1} highest energy")
            fig.tight_layout()
        plt.savefig(f"save/diff-type{args.type}-nout{args.nout}.png")
        print(f"\nFigure saved to save/diff-type{args.type}-nout{args.nout}.png\n")

    return h_diff, h_comp


def main():
    # training settings
    parser = argparse.ArgumentParser(description='sPHENIX TPC clustering')
    parser.add_argument('--type', type=int, default=0, metavar='N',
                        help='training type (ntruth: 0, pos: 1, phicov: 2, zcov: 3, adc: 4, default: 0)')
    parser.add_argument('--nout', type=int, default=1, metavar='N',
                        help='number of output clusters (default: 1)')
    parser.add_argument('--nmax', type=int, default=5, metavar='N',
                        help='max number of output clusters (default: 5)')
    parser.add_argument('--nfiles', type=int, default=10000, metavar='N',
                        help='max number of files used for training (default: 10000)')
    parser.add_argument('--data-size', type=int, default=5, metavar='N',
                        help='number of files for each training (default: 5)')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 256)')
    parser.add_argument('--test-batch-size', type=int, default=1024, metavar='N',
                        help='input batch size for testing (default: 1024)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.5, metavar='M',
                        help='learning rate step gamma (default: 0.5)')
    parser.add_argument('--use-conv', action='store_true', default=False,
                        help='use convolutional neural network')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disable CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=400, metavar='N',
                        help='how many batches to wait before logging training status (default: 400)')
    parser.add_argument('--save-interval', type=int, default=5, metavar='N',
                        help='how many groups of dataset to wait before saving the checkpoint (default: 5)')
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
    print(f"\nUsing {device} device\n")

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
    data_size = min(args.data_size, len(files))

    if args.type == 0:
        h_diff = np.zeros(9, dtype=np.int64)
        h_comp = np.zeros(9, dtype=np.int64)
        args.nout = args.nmax
    else:
        h_diff = []
        h_comp = []
        for _ in range(args.nout):
            h_diff.append(np.zeros((50, 50), dtype=np.int64))
            h_comp.append(np.zeros((50, 50), dtype=np.int64))

    epochs_trained = 0
    sets_trained = 0
    model = Net(args).to(device)
    if args.load_model:
        print("\nLoading model\n")
        model.load_state_dict(torch.load(f"save/net_weights-type{args.type}-nout{args.nout}.pt"))
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    if args.load_checkpoint:
        print("\nLoading checkpoint\n")
        checkpoint = torch.load(f"save/checkpoint-type{args.type}-nout{args.nout}")
        epochs_trained = checkpoint['epochs_trained']
        sets_trained = checkpoint['sets_trained']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(epochs_trained, args.epochs):
        for iset in range(sets_trained, nfiles, data_size):
            ilast = min(iset + data_size, nfiles)
            savenow = ilast == nfiles or (ilast - sets_trained) // data_size % args.save_interval == 0
            print(f"\nDataset: {iset + 1} to {ilast}\n")
            if data_size < nfiles or epoch == epochs_trained:
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
                            }, f"save/checkpoint-type{args.type}-nout{args.nout}")

    if args.save_model:
        model.cpu()
        model.eval()
        if args.use_conv:
            example = torch.randn(1, 3, 11, 11)
        else:
            example = torch.randn(1, 11*11+2)
        traced_script_module = torch.jit.trace(model, example)
        traced_script_module.save(f"save/net_model-type{args.type}-nout{args.nout}.pt")
        torch.save(model.state_dict(), f"save/net_weights-type{args.type}-nout{args.nout}.pt")
        print(f"\nModel saved to save/net_model-type{args.type}-nout{args.nout}.pt\n")


if __name__ == '__main__':
    main()