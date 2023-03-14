from __future__ import print_function
import argparse
import sys
import os
import math
import numpy as np
import uproot as ur
from scipy.optimize import curve_fit
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
            branch += ["ntruth"]
        elif args.type <= 4:
            branch += ["truth_adc"]
            if args.type <= 3:
                branch += ["truth_phi", "truth_z"]
        elif args.type == 5:
            branch += ["truhit_phi", "truhit_z"]
        else:
            sys.exit("\nError: Wrong type number\n")
        if args.type <= 4:
            branch += ["ntouch"]
        if args.type >= 1:
            branch += ["reco_adc"]
        if args.type == 1 or args.type == 5:
            branch += ["reco_phi", "reco_z"]
        tree = ur.concatenate(files, branch, library='np')

        self.has_comp = False
        if args.type <= 4:
            ntouch = torch.from_numpy(tree["ntouch"])
        if args.type >= 1:
            cadc = torch.from_numpy(tree["reco_adc"])
            ccnt = torch.where(cadc > 70)[0].bincount() >= 1
        if args.type == 0:
            batch_ind = torch.where(ntouch >= 0)[0]
            self.target = torch.from_numpy(tree["ntruth"])[batch_ind, 0:].sum(dim=1).clamp(min=0, max=1).type(torch.float32).unsqueeze(1)
        elif args.type <= 4:
            gadc = torch.from_numpy(tree["truth_adc"])
            gcnt = torch.where(gadc > 0)[0].bincount() >= args.nout
            batch_ind = torch.where(ccnt * gcnt * (ntouch >= 0))[0]
            batch_si = batch_ind.unsqueeze(1).expand(-1, args.nout).flatten()
            si = gadc[batch_ind].argsort(dim=1, descending=True)[:, :args.nout].flatten()
            if args.type <= 3:
                phi = torch.from_numpy(tree["truth_phi"])[batch_si, si].type(torch.float32).view(-1, args.nout)
                z = torch.from_numpy(tree["truth_z"])[batch_si, si].type(torch.float32).view(-1, args.nout)
                self.target = torch.stack((phi, z), dim=1)
            elif args.type == 4:
                self.target = gadc[batch_si, si].type(torch.float32).view(-1, args.nout)
        elif args.type == 5:
            phi = torch.from_numpy(tree["truhit_phi"]).type(torch.float32).unsqueeze(1)
            z = torch.from_numpy(tree["truhit_z"]).type(torch.float32).unsqueeze(1)
            phiz = torch.stack((phi, z), dim=1)
            batch_ind = torch.where(ccnt * (LA.matrix_norm(phiz) < 5))[0]
            batch_si = batch_ind
            self.target = phiz[batch_si]
        if args.type == 1 or args.type == 5:
            si = cadc[batch_ind].argsort(dim=1, descending=True)[:, :args.nout].flatten()
            phi = torch.from_numpy(tree["reco_phi"])[batch_si, si].type(torch.float32).view(-1, args.nout)
            z = torch.from_numpy(tree["reco_z"])[batch_si, si].type(torch.float32).view(-1, args.nout)
            self.comp = torch.stack((phi, z), dim=1)
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


class DataMLP(Dataset):
    def __init__(self, args, files):
        tree = ur.concatenate(files, library='np')

        self.target = torch.from_numpy(tree["ntruth"]).neg().add(2).clamp(min=0, max=1).type(torch.float32).unsqueeze(1)

        adc = torch.from_numpy(tree["e1"]).type(torch.float32).unsqueeze(1)
        for i in range(1, 3*3):
            adc = torch.cat((adc, torch.from_numpy(tree[f"e{i+1}"]).type(torch.float32).unsqueeze(1)), dim=1)
        center_x = torch.from_numpy(tree["center_x"]).type(torch.float32).unsqueeze(1)
        center_y = torch.from_numpy(tree["center_y"]).type(torch.float32).unsqueeze(1)
        if args.use_conv:
            adc = adc.view(-1, 3, 3)
            center_x = center_x.view(-1, 1, 1).expand(-1, 3, 3)
            center_y = center_y.view(-1, 1, 1).expand(-1, 3, 3)
            self.input = torch.stack((adc, center_x, center_y), dim=1)
        else:
            self.input = torch.cat((adc, center_x, center_y), dim=1)

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        input = self.input[idx]
        target = self.target[idx]
        return input, target


class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.type = args.type
        self.nout = args.nout
        self.use_conv = args.use_conv

        if self.type == 0 or self.type == 2 or self.type == 3:
            nout = 1
        elif self.type == 1 or self.type == 5:
            nout = 2 * self.nout
        elif self.type == 4:
            nout = self.nout
        else:
            sys.exit("\nError: Wrong type number\n")

        nrow = 3 if args.use_mlp else 11
        nch = 3 if args.use_mlp else 3

        if self.use_conv:
            print("\nUsing convolutional neural network\n")
            nin = nrow**2+nch+4
            self.conv1 = nn.Conv2d(nch, nch+4, 3, 1, padding='same')
            self.conv2 = nn.Conv2d(nch+4, nch+8, 3, 1, padding='same')
            self.fc1 = nn.Linear((nch+8)*math.floor(nrow/2)**2, nin)
            self.fc2 = nn.Linear(nin, nout)
            self.norm0 = nn.BatchNorm2d(nch)
            self.norm1 = nn.BatchNorm2d(nch+4)
            self.norm2 = nn.BatchNorm2d(nch+8)
            self.norm3 = nn.BatchNorm1d(nin)
            self.dropout1 = nn.Dropout(0.25)
            self.dropout2 = nn.Dropout(0.5)
        else:
            print("\nUsing linear neural network\n")
            nin = nrow**2+nch-1
            self.fc1 = nn.Linear(nin, nin+5)
            self.fc2 = nn.Linear(nin+5, nout)
            self.norm0 = nn.BatchNorm1d(nin)
            self.norm1 = nn.BatchNorm1d(nin+5)

    def forward(self, x):
        if self.use_conv:
            x = self.norm0(x)
            x = self.conv1(x)
            x = self.norm1(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = self.norm2(x)
            x = F.relu(x)
            x = F.max_pool2d(x, 2, ceil_mode=False)
            #x = self.dropout1(x)
            x = x.flatten(start_dim=1)
            x = self.fc1(x)
            x = self.norm3(x)
            x = F.relu(x)
            #x = self.dropout2(x)
            x = self.fc2(x)
        else:
            x = self.norm0(x)
            x = self.fc1(x)
            x = self.norm1(x)
            x = torch.tanh(x)
            x = self.fc2(x)

        if self.type == 0:
            output = torch.sigmoid(x)
        elif self.type == 2 or self.type == 3:
            output = x
        elif self.type == 1 or self.type == 5:
            output = x.view(-1, 2, self.nout)
        elif self.type == 4:
            output = x.view(-1, self.nout)
        return output


def gauss(x, *p):
    A, mu, sigma = p
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


def train(args, model, model_pos, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, item in enumerate(train_loader):
        data, target = item[0].to(device), item[1].to(device)
        if args.type == 2 or args.type == 3:
            target = (model_pos(data)[:, args.type-2, args.iout] - target[:, args.type-2, args.iout]).unsqueeze(1).abs()
        optimizer.zero_grad()
        output = model(data)
        # mean batch loss for each element
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                epoch + 1, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, model_pos, device, test_loader, savenow):
    model.eval()
    test_loss = 0
    correct = 0
    hlist = []
    with torch.no_grad():
        for item in test_loader:
            nitem = len(item)
            if nitem >= 3:
                data, target, comp = item[0].to(device), item[1].to(device), item[2].to(device)
            else:
                data, target = item[0].to(device), item[1].to(device)
            if args.type == 2 or args.type == 3:
                target = (model_pos(data)[:, args.type-2, args.iout] - target[:, args.type-2, args.iout]).unsqueeze(1).abs()
            output = model(data)
            # sum up batch loss for all elements
            test_loss += F.mse_loss(output, target, reduction='sum').item()
            if args.type == 0:
                # sig = 1 and bg = 0
                pred = torch.where(output > 0.5, 1, 0)
                refs = target.type(torch.int64).view_as(pred)
                correct += pred.eq(refs).sum().item()
                if args.print and savenow:
                    vlist = torch.stack((target, output), dim=0)
                    hvalue = vlist.detach().cpu().numpy()
                    hist, xedges, yedges = np.histogram2d(hvalue[0, :, 0], hvalue[1, :, 0],
                                                          bins=[2, 100], range=[[-0.5, 1.5], [0, 1]])
                    if hlist:
                        hlist[0] = np.add(hlist[0], hist)
                    else:
                        hlist.append(hist)
            else:
                diff = output.sub(target)
                # count the number of improved predictions
                try: comp
                except NameError: pass
                else:
                    diff_comp = comp.sub(target)
                    correct += (LA.matrix_norm(diff) < LA.matrix_norm(diff_comp)).sum().item()
                if args.print and savenow:
                    try: comp
                    except NameError:
                        vlist = diff.unsqueeze(0)
                    else:
                        vlist = torch.stack((diff, diff_comp), dim=0)
                    hll = []
                    for iout in range(args.nout):
                        hl = []
                        for i in range(nitem-1):
                            hvalue = vlist[i].detach().cpu().numpy()
                            hist, xedges, yedges = np.histogram2d(hvalue[:, 0, iout], hvalue[:, 1, iout],
                                                                  bins=200, range=[[-1, 1], [-1, 1]])
                            hl.append(hist)
                            if hlist:
                                hlist[iout][i] = np.add(hlist[iout][i], hist)
                        hll.append(hl)
                    if not hlist:
                        hlist = hll

    # mean batch loss for each element
    test_loss /= len(test_loader.dataset) * len(test_loader.dataset[0][1].flatten())

    print("\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    if args.print and savenow:
        plt.clf()
        if args.type == 0:
            ncum = np.cumsum(hlist[0], axis=1)
            ncum = np.expand_dims(ncum[:, -1], axis=1) - ncum
            hlabel = ['Sig', 'Bg', r'$S/\sqrt{S+B}$']
            bin_centers = (yedges[:-1] + yedges[1:]) / 2
            yvalue = [ncum[1] / ncum[1, 0], ncum[0] / ncum[0, 0],
                      ncum[1] / np.sqrt(ncum[1] + ncum[0] + 1e-6) / np.sqrt(ncum[1, 0] + 1e-6)]
            for i in range(3):
                plt.plot(bin_centers, yvalue[i], drawstyle='steps-mid', label=hlabel[i])
            plt.xlabel(r"Cut value")
            plt.ylabel(r"Efficiency")
        else:
            hlabel = ['NN', 'Reco']
            cname = ['green', 'red']
            alabel = [r"$\Delta\phi$", r"$\Delta z$"]
            fig, axs = plt.subplots(2*args.nout, nitem-1, figsize=(12*(nitem-1), 24*args.nout))
            if nitem-1 == 1:
                axs = np.expand_dims(axs, axis=1)
            for iout in range(args.nout):
                for i in range(nitem-1):
                    # numpy.histogram2d does not follow Cartesian convention (see Notes)
                    # therefore transpose for visualization purposes
                    im = axs[2*iout, i].imshow(hlist[iout][i].T, interpolation='nearest', origin='lower',
                                               extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                                               norm=colors.LogNorm())
                    fig.colorbar(im, ax=axs[iout, i], fraction=0.047)
                    axs[2*iout, i].set(xlabel=alabel[0])
                    axs[2*iout, i].set(ylabel=alabel[1])
                    axs[2*iout, i].set(title=f"{hlabel[i]}: No. {iout + 1} highest energy")
                    for j in range(2):
                        bin_centers = (xedges[:-1] + xedges[1:]) / 2
                        hproj = np.sum(hlist[iout][i], axis=1-j)
                        coeff, covar = curve_fit(gauss, bin_centers, hproj, p0=[hproj[100], 0, 0.5])
                        hfit = gauss(bin_centers, *coeff)
                        axs[2*iout+1, j].plot(bin_centers, hproj, drawstyle='steps-mid', color=cname[i], label=hlabel[i])
                        axs[2*iout+1, j].plot(bin_centers, hfit, color='black')
                        axs[2*iout+1, j].set(xlabel=alabel[j])
                        axs[2*iout+1, j].text(0.05, 0.95-0.05*i, r"{}: $\sigma$ = {:.4f}".format(hlabel[i], coeff[2]),
                                              fontsize='xx-large', transform=axs[2*iout+1, j].transAxes)
            fig.tight_layout()
        plt.legend(fontsize='xx-large')
        plt.savefig(f"save/diff-type{args.type}-nout{args.nout}.png")
        print(f"\nFigure saved to save/diff-type{args.type}-nout{args.nout}.png\n")


def main():
    # training settings
    parser = argparse.ArgumentParser(description='sPHENIX TPC clustering')
    parser.add_argument('--type', type=int, default=0, metavar='N',
                        help='training type (ntruth: 0, pos: 1, phierr: 2, zerr: 3, adc: 4, truhit: 5, default: 0)')
    parser.add_argument('--nout', type=int, default=1, metavar='N',
                        help='number of output clusters (default: 1)')
    parser.add_argument('--iout', type=int, default=0, metavar='N',
                        help='index of the output cluster used in phierr and zerr (default: 0)')
    parser.add_argument('--dir', type=str, default='data', metavar='DIR',
                        help='directory of data (default: data)')
    parser.add_argument('--nfiles', type=int, default=10000, metavar='N',
                        help='max number of files used for training (default: 10000)')
    parser.add_argument('--data-size', type=int, default=200, metavar='N',
                        help='number of files for each training (default: 200)')
    parser.add_argument('--batch-size', type=int, default=1024, metavar='N',
                        help='input batch size for training (default: 1024)')
    parser.add_argument('--test-batch-size', type=int, default=1024, metavar='N',
                        help='input batch size for testing (default: 1024)')
    parser.add_argument('--epochs', type=int, default=40, metavar='N',
                        help='number of epochs to train (default: 40)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.95, metavar='M',
                        help='learning rate step gamma (default: 0.95)')
    parser.add_argument('--use-conv', action='store_true', default=False,
                        help='use convolutional neural network')
    parser.add_argument('--use-mlp', action='store_true', default=False,
                        help='use MLP network')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disable CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=400, metavar='N',
                        help='how many batches to wait before logging training status (default: 400)')
    parser.add_argument('--save-interval', type=int, default=20, metavar='N',
                        help='how many groups of dataset to wait before saving the checkpoint (default: 20)')
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

    sets_trained = 0
    model = Net(args).to(device)
    if args.load_model:
        print("\nLoading model\n")
        model.load_state_dict(torch.load(f"save/net_weights-type{args.type}-nout{args.nout}.pt"))
    model_pos = None
    if args.type == 2 or args.type == 3:
        train_type = args.type
        args.type = 1
        model_pos = Net(args).to(device)
        print("\nLoading model\n")
        model_pos.load_state_dict(torch.load(f"save/net_weights-type{args.type}-nout{args.nout}.pt"))
        model_pos.eval()
        args.type = train_type
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    if args.load_checkpoint:
        print("\nLoading checkpoint\n")
        checkpoint = torch.load(f"save/checkpoint-type{args.type}-nout{args.nout}")
        sets_trained = checkpoint['sets_trained']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for iset in range(sets_trained, nfiles, data_size):
        ilast = min(iset + data_size, nfiles)
        print(f"\nDataset: {iset + 1} to {ilast}\n")
        dataset = DataMLP(args, files[iset:ilast]) if args.use_mlp else Data(args, files[iset:ilast])
        nevents = len(dataset)
        ntrain = int(nevents*0.9)
        ntest = nevents - ntrain
        train_set, test_set = random_split(dataset, [ntrain, ntest])
        train_loader = DataLoader(train_set, **train_kwargs)
        test_loader = DataLoader(test_set, **test_kwargs)
        for epoch in range(args.epochs):
            epoch_done = (epoch + 1) == args.epochs
            savenow = epoch_done or (epoch + 1) % args.save_interval == 0
            train(args, model, model_pos, device, train_loader, optimizer, epoch)
            test(args, model, model_pos, device, test_loader, savenow)
            scheduler.step()
            if args.save_checkpoint and savenow:
                print("\nSaving checkpoint\n")
                if epoch_done:
                    sets_trained = 0 if ilast == nfiles else ilast
                torch.save({'sets_trained': sets_trained,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            }, f"save/checkpoint-type{args.type}-nout{args.nout}")

    if args.save_model:
        model.cpu()
        model.eval()
        nrow = 3 if args.use_mlp else 11
        nch = 3 if args.use_mlp else 3
        if args.use_conv:
            example = torch.randn(1, nch, nrow, nrow)
        else:
            example = torch.randn(1, nrow**2+nch-1)
        traced_script_module = torch.jit.trace(model, example)
        traced_script_module.save(f"save/net_model-type{args.type}-nout{args.nout}.pt")
        torch.save(model.state_dict(), f"save/net_weights-type{args.type}-nout{args.nout}.pt")
        print(f"\nModel saved to save/net_model-type{args.type}-nout{args.nout}.pt\n")


if __name__ == '__main__':
    main()