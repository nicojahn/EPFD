# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import argparse
import time
import sys

from sklearn import datasets
from sklearn import neighbors

import numpy as np
np.random.seed(4567)

from pyensemble.classify import BaggingEnsembleAlgorithm
from data_distributed import (distributed_single_pruning,
                              distributed_pruning_methods)
from data_distributed import COMEP_Pruning, DOMEP_Pruning



def define_params(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--nb-cls", type=int, default=17,
        help='Number of individual classifiers in the ensemble')
    parser.add_argument("--nb-pru", type=int, default=5,
        help='Number of members in the pruned sub-ensemble')

    parser.add_argument("--name-pru", type=str, default='COMEP',
        choices=['ES', 'KP', 'KL', 'RE', 'OO',
                 'DREP', 'SEP', 'OEP', 'PEP',
                 'GMA', 'LCS', 'COMEP', 'DOMEP'],
        help='Name of the expected ensemble pruning method')

    parser.add_argument("--distributed", action="store_true",
        help='Whether to use EPFD (framework)')

    parser.add_argument("--lam", type=float, default=0.5,
        help="lambda")
    parser.add_argument("--m", type=int, default=2,
        help='Number of Machines')

    return parser.parse_args(args)


def load_iris():
    # import some data to play with
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    choice = list(range(len(y)))
    np.random.shuffle(choice)
    X = X[choice]
    y = y[choice]

    choice = np.random.choice([0, 1, 2])
    cindex = y != choice
    y = y[cindex]
    X = X[cindex]

    cindex = list(range(len(y)))
    np.random.shuffle(cindex)
    choice = len(cindex) // 2
    X_trn = X[: choice].tolist()
    y_trn = y[: choice].tolist()
    X_tst = X[choice :].tolist()
    y_tst = y[choice :].tolist()

    return X_trn, y_trn, X_tst, y_tst

def run_method(nb_pru, name_pru, distributed, lam, rho, m, y_validation, y_prediction):
    classifiers = []
    if name_pru not in ['COMEP', 'DOMEP']:

        if distributed:
            since = time.time()
            classifiers = Pd = distributed_pruning_methods(y_validation, y_prediction,
                nb_pru, m, name_pru, rho=rho)
            Td = time.time() - since
            print("{:5s}: {:.4f}s, get {}".format(name_pru, Td, Pd))

        else:
            since = time.time()
            classifiers = Pc = distributed_single_pruning(name_pru, y_validation, y_prediction,
                    nb_cls, nb_pru, rho=rho)
            Tc = time.time() - since
            print("{:5s}: {:.4f}s, get {}".format(name_pru, Tc, Pc))

    elif name_pru == 'COMEP':
        since = time.time()
        classifiers = Pc = COMEP_Pruning(np.array(y_prediction).T, nb_pru, y_validation, lam)
        Tc = time.time() - since
        print("{:5s}: {:.4f}s, get {}".format(name_pru, Tc, Pc))

    elif name_pru == 'DOMEP':
        since = time.time()
        classifiers = Pd = DOMEP_Pruning(np.array(y_prediction).T, nb_pru, m, y_validation, lam)
        Td = time.time() - since
        print("{:5s}: {:.4f}s, get {}".format(name_pru, Td, Pd))

    else:
        raise ValueError("Please check the `name_pru`.")

    return classifiers

def main(args, y_validation, y_prediction):
    #nb_cls = args.nb_cls # unused
    nb_pru = args.nb_pru
    name_pru = args.name_pru
    distributed = args.distributed
    lam = args.lam
    m = args.m

    nb_cls = len(y_validation)
    rho = nb_pru / nb_cls
    classifiers = run_method(nb_pru, name_pru, distributed, lam, rho, m, y_validation, y_prediction)
    return classifiers

if __name__ == "__main__":
    args = define_params(sys.argv[1:])
    main(args)