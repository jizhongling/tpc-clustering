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
        branch = ["adc", "layer", "ztan", "ntouch"]
        if args.type == 0:
            branch += ["ntruth", "nreco"]
        elif args.type == 1:
            branch += ["truth_rphi", "truth_z", "truth_adc", "reco_rphi", "reco_z", "reco_adc"]
        elif args.type == 2:
            branch += ["truth_rphicov", "truth_adc"]
        elif args.type == 3:
            branch += ["truth_zcov", "truth_adc"]
        elif args.type == 4:
            branch += ["truth_adc"]
        elif args.type == 5:
            branch += ["track_rphi", "track_z", "reco_rphi", "reco_z"]
        else:
            sys.exit("\nError: Wrong type number\n")
        tree = ur.concatenate(files, branch, library='np')

        self.has_comp = False
        ntouch = torch.from_numpy(tree["ntouch"])
        if args.type == 0:
            batch_ind = torch.where(ntouch >= 1)[0]
            self.target = torch.from_numpy(tree["ntruth"])[batch_ind, 0:].type(torch.int64).sum(dim=1).clamp(max=args.nmax)
            self.comp = torch.from_numpy(tree["nreco"])[batch_ind, 0:].type(torch.int64).sum(dim=1).clamp(max=args.nmax)
            self.has_comp = True
        elif args.type <= 4:
            gadc = torch.from_numpy(tree["truth_adc"])
            ntouch = ntouch.unsqueeze(1).expand(-1, gadc.size(dim=1))
            ind = torch.where((gadc > 0) * (ntouch >= 1))[0]
            batch_ind = torch.where(ind.bincount() >= args.nout)[0]
            batch_si = batch_ind.unsqueeze(1).expand(-1, args.nout).flatten()
            si = gadc[batch_ind].argsort(dim=1, descending=True)[:, :args.nout].flatten()
            if args.type == 1:
                rphi = torch.from_numpy(tree["truth_rphi"])[batch_si, si].type(torch.float32).view(-1, args.nout)
                z = torch.from_numpy(tree["truth_z"])[batch_si, si].type(torch.float32).view(-1, args.nout)
                self.target = torch.stack((rphi, z), dim=1)
                si = torch.from_numpy(tree["reco_adc"])[batch_ind].argsort(dim=1, descending=True)[:, :args.nout].flatten()
                rphi = torch.from_numpy(tree["reco_rphi"])[batch_si, si].type(torch.float32).view(-1, args.nout)
                z = torch.from_numpy(tree["reco_z"])[batch_si, si].type(torch.float32).view(-1, args.nout)
                self.comp = torch.stack((rphi, z), dim=1)
                self.has_comp = True
            elif args.type == 2:
                self.target = torch.from_numpy(tree["truth_rphicov"])[batch_si, si].type(torch.float32).view(-1, args.nout)
            elif args.type == 3:
                self.target = torch.from_numpy(tree["truth_zcov"])[batch_si, si].type(torch.float32).view(-1, args.nout)
            elif args.type == 4:
                self.target = gadc[batch_si, si].type(torch.float32).view(-1, args.nout)
        elif args.type == 5:
            rphi = torch.from_numpy(tree["track_rphi"]).type(torch.float32).unsqueeze(1)
            z = torch.from_numpy(tree["track_z"]).type(torch.float32).unsqueeze(1)
            rphiz = torch.stack((rphi, z), dim=1)
            batch_ind = torch.where((ntouch >= 1) * (LA.matrix_norm(rphiz) < 1))[0]
            self.target = rphiz[batch_ind]
            rphi = torch.from_numpy(tree["reco_rphi"])[batch_ind, 0].type(torch.float32).unsqueeze(1)
            z = torch.from_numpy(tree["reco_z"])[batch_ind, 0].type(torch.float32).unsqueeze(1)
            self.comp = torch.stack((rphi, z), dim=1)
            self.has_comp = True

        if args.use_conv:
            adc = torch.from_numpy(tree["adc"])[batch_ind].type(torch.float32).view(-1, 11, 11)
            layer = torch.from_numpy(tree["layer"])[batch_ind].type(torch.float32).view(-1, 1, 1).expand(-1, 11, 11)
            ztan = torch.from_numpy(tree["ztan"])[batch_ind].type(torch.float32).view(-1, 1, 1).expand(-1, 11, 11)
            self.input = torch.stack((adc, layer, ztan), dim=1)
        else:
            adc = torch.from_numpy(tree["adc"])[batch_ind].type(torch.float32)
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
        elif self.type == 1 or self.type == 5:
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
        elif self.type == 1 or self.type == 5:
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


def test(args, model, device, test_loader, hlist, savenow):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for item in test_loader:
            nitem = len(item)
            if nitem >= 3:
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
                    try: comp
                    except NameError:
                        vlist = torch.stack((pred, refs), dim=0)
                    else:
                        vlist = torch.stack((pred, refs, comp), dim=0)
                    for i in range(nitem):
                        hvalue = vlist[i].detach().cpu().numpy()
                        hist, bin_edges = np.histogram(hvalue, bins=7, range=(-1.5, 5.5))
                        hlist[i] = np.add(hlist[i], hist)
            else:
                # sum up batch loss for all elements
                test_loss += F.mse_loss(output, target, reduction='sum').item()
                # count the number of improved predictions
                try: comp
                except NameError: pass
                else:
                    diff = output.sub(target)
                    diff_comp = comp.sub(target)
                    correct += (LA.matrix_norm(diff) < LA.matrix_norm(diff_comp)).sum().item()
                if args.print:
                    try: comp
                    except NameError:
                        vlist = torch.stack((output, target), dim=0)
                    else:
                        vlist = torch.stack((output, target, comp), dim=0)
                    for iout in range(args.nout):
                        for i in range(nitem):
                            hvalue = vlist[i].detach().cpu().numpy()
                            hist, xedges, yedges = np.histogram2d(hvalue[:, 0, iout], hvalue[:, 1, iout],
                                                                  bins=400, range=[[-2, 2], [-2, 2]])
                            hlist[iout][i] = np.add(hlist[iout][i], hist)

    # mean batch loss for each element
    test_loss /= len(test_loader.dataset) * len(test_loader.dataset[0][1].flatten())

    print("\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    if args.print and savenow:
        hlabel = ['NN', 'truth', 'reco']
        plt.clf()
        if args.type == 0:
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            for i in range(nitem):
                plt.plot(bin_centers, hlist[i], drawstyle='steps-mid', label=hlabel[i])
            plt.legend()
            plt.xlabel(r"Number of clusters")
        else:
            fig, axs = plt.subplots(args.nout, 3, figsize=(12, 12))
            if args.nout == 1:
                axs = np.expand_dims(axs, axis=0)
            for iout in range(args.nout):
                for i in range(nitem):
                    im = axs[iout, i].imshow(hlist[iout][i], interpolation='nearest', origin='lower',
                                             extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                                             norm=colors.LogNorm())
                    fig.colorbar(im, ax=axs[iout, i], fraction=0.047)
                    axs[iout, i].set(xlabel=r"$r\phi$ (cm)")
                    axs[iout, i].set(ylabel=r"$z$ (cm)")
                    axs[iout, i].set(title=f"{hlabel[i]}: No. {iout + 1} highest energy")
            fig.tight_layout()
        plt.savefig(f"save/diff-type{args.type}-nout{args.nout}.png")
        print(f"\nFigure saved to save/diff-type{args.type}-nout{args.nout}.png\n")

    return hlist


def main():
    # training settings
    parser = argparse.ArgumentParser(description='sPHENIX TPC clustering')
    parser.add_argument('--type', type=int, default=0, metavar='N',
                        help='training type (ntruth: 0, pos: 1, rphicov: 2, zcov: 3, adc: 4, track: 5, default: 0)')
    parser.add_argument('--nout', type=int, default=1, metavar='N',
                        help='number of output clusters (default: 1)')
    parser.add_argument('--nmax', type=int, default=3, metavar='N',
                        help='max number of output clusters (default: 3)')
    parser.add_argument('--dir', type=str, default='data', metavar='DIR',
                        help='directory of data (default: data)')
    parser.add_argument('--nfiles', type=int, default=10000, metavar='N',
                        help='max number of files used for training (default: 10000)')
    parser.add_argument('--data-size', type=int, default=10, metavar='N',
                        help='number of files for each training (default: 10)')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 256)')
    parser.add_argument('--test-batch-size', type=int, default=1024, metavar='N',
                        help='input batch size for testing (default: 1024)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 1)')
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
    for file in os.scandir(args.dir):
        if (file.name.startswith("training-") and
            file.name.endswith(".root") and
            file.is_file()):
            files.append(file.path + ":T")
    nfiles = min(args.nfiles, len(files))
    data_size = min(args.data_size, len(files))

    if args.type == 0:
        h = np.zeros(7, dtype=np.int64)
        hlist = [h for _ in range(3)]
        args.nout = args.nmax
    else:
        h = np.zeros((400, 400), dtype=np.int64)
        h = [h for _ in range(3)]
        hlist = [h for _ in range(args.nout)]
        if args.type == 5:
            args.nout = 1

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
            hlist = test(args, model, device, test_loader, hlist, savenow)
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