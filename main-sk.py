from __future__ import print_function
import argparse
import sys
import os
import numpy as np
import pandas as pd
import uproot as ur
import matplotlib.pyplot as plt
from graphviz import Source
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


class DataHF():
    def __init__(self, _, files):
        branch = ["signal", "px", "py", "pz", "pt", "eta", "phi", "deltapt", "deltaeta", "deltaphi", "siqr", "siphi", "sithe", "six0", "siy0", "tpqr", "tpphi", "tpthe", "tpx0", "tpy0", "charge", "quality", "chisq", "ndf", "nhits", "layers", "nmaps", "nintt", "ntpc", "nmms", "ntpc1", "ntpc11", "ntpc2", "ntpc3", "nlmaps", "nlintt", "nltpc", "nlmms", "vx", "vy", "vz", "dca3dxy", "dca3dxysigma", "dca3dz", "dca3dzsigma", "pcax", "pcay", "pcaz", "npedge", "nredge", "nbig", "novlp", "merr", "msize", "nhittpcall", "nhittpcin", "nhittpcmid", "nhittpcout", "nclusall", "nclustpc", "nclusintt", "nclusmaps", "nclusmms", "clus_e_cemc", "clus_e_hcalin", "clus_e_hcalout", "clus_e_outer_cemc", "clus_e_outer_hcalin", "clus_e_outer_hcalout"]
        root_tree = ur.concatenate(files, branch, library='pd')
        self.input = root_tree.drop("signal", axis=1)
        self.target = root_tree["signal"]

    def __len__(self):
        return len(self.target)


class Tree():
    def __init__(self, args):
        if args.method == 0:
            print("\nUsing decision tree\n")
            self.estimator = DecisionTreeClassifier(max_depth=None, class_weight='balanced')
        elif args.method == 1:
            print("\nUsing random forest\n")
            self.estimator = RandomForestClassifier(n_estimators=args.ntree, random_state=args.seed, class_weight='balanced')
        elif args.method == 2:
            print("\nUsing gradient boosting decision trees\n")
            self.estimator = GradientBoostingClassifier(n_estimators=args.ntree, random_state=args.seed, learning_rate=args.lr)
        elif args.method == 3:
            print("\nUsing histogram-based gradient boosting decision trees\n")
            self.estimator = HistGradientBoostingClassifier(max_iter=args.ntree, random_state=args.seed, learning_rate=args.lr, class_weight='balanced')
        else:
            sys.exit("\nError: Wrong method number\n")


def train(_, model, X_train, y_train):
    model.fit(X_train, y_train)


def test(args, model, X_test, y_test, feature_names):
    neg_bkg, pos_bkg, neg_sig, pos_sig = confusion_matrix(y_test, model.predict(X_test)).flatten()
    test_size = neg_bkg + pos_bkg + neg_sig + pos_sig
    correct = neg_bkg + pos_sig

    if correct > 0:
        print("Accuracy: {}/{} ({:.0f}%)".format(correct, test_size, 100. * correct / test_size))
    if pos_sig + neg_sig > 0:
        print("Efficiency: {}/{} ({:.0f}%)".format(pos_sig, pos_sig + neg_sig, 100. * pos_sig / (pos_sig + neg_sig)))
    if pos_sig + pos_bkg > 0:
        print("Purity: {}/{} ({:.0f}%)".format(pos_sig, pos_sig + pos_bkg, 100. * pos_sig / (pos_sig + pos_bkg)))
    if pos_bkg > 0:
        print("Rejection: {}/{} ({:.2f})".format(pos_bkg + neg_bkg, pos_bkg, 1. * (pos_bkg + neg_bkg) / pos_bkg))

    if args.print:
        if args.method == 0:
            dot_data = export_graphviz(model, out_file=None, max_depth=3, feature_names=feature_names, filled=True)
            graph = Source(dot_data)
            graph.render("save/decision-tree", view=False)
            print("\nFigure saved to save/decision-tree.pdf\n")
        elif args.method == 1:
            importances = pd.Series(model.feature_importances_, index=feature_names)
            std = 0 #np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
            fig, ax = plt.subplots(figsize=(15, 7.5))
            importances.plot.bar(yerr=std, ax=ax)
            ax.set_ylabel("Mean decrease in impurity")
            ax.set_ylim(bottom=0)
            fig.tight_layout()
            plt.savefig("save/impurity-decrease.png")
            print("\nFigure saved to save/impurity-decrease.png\n")


def main():
    # training settings
    parser = argparse.ArgumentParser(description='sPHENIX HF electron ID')
    parser.add_argument('--type', type=int, default=0, metavar='N',
                        help='training type (hf: 0, default: 0)')
    parser.add_argument('--dir', type=str, default='data', metavar='DIR',
                        help='directory of data (default: data)')
    parser.add_argument('--nfiles', type=int, default=10000, metavar='N',
                        help='max number of files used for training (default: 10000)')
    parser.add_argument('--data-size', type=int, default=200, metavar='N',
                        help='number of files for each training (default: 200)')
    parser.add_argument('--method', type=int, default=3, metavar='N',
                        help='classifier (decision tree: 0, random forest: 1, gradient boosting: 2, hist gradient: 3, default: 3)')
    parser.add_argument('--ntree', type=int, default=100, metavar='N',
                        help='number of trees in ensemble (default: 100)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--seed', type=int, default=None, metavar='S',
                        help='random seed (default: None)')
    parser.add_argument('--print', action='store_true', default=False,
                        help='print output')
    args = parser.parse_args()

    prefix = "hf-electron"
    key = "ntp"
    files = []
    for file in os.scandir(args.dir):
        if (file.name.startswith(prefix) and
            file.name.endswith(".root") and
            file.is_file()):
            files.append(file.path + ":" + key)
    nfiles = min(args.nfiles, len(files))
    data_size = min(args.data_size, len(files))

    model = Tree(args).estimator

    for iset in range(0, nfiles, data_size):
        ilast = min(iset + data_size, nfiles)
        print(f"\nDataset: {iset + 1} to {ilast}\n")
        if args.type == 0:
            dataset = DataHF(args, files[iset:ilast])
        else:
            sys.exit("\nError: Wrong type number\n")
        X_train, X_test, y_train, y_test = train_test_split(dataset.input, dataset.target, test_size=0.2)
        feature_names = X_test.columns
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        train(args, model, X_train, y_train)
        test(args, model, X_test, y_test, feature_names)


if __name__ == '__main__':
    main()