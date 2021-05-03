# EPFD

[![Build Status](https://travis-ci.org/eustomaqua/EPFD.svg?branch=master)](https://travis-ci.org/eustomaqua/EPFD) 
[![Coverage Status](https://coveralls.io/repos/github/eustomaqua/EPFD/badge.svg?branch=master)](https://coveralls.io/github/eustomaqua/EPFD?branch=master) 
[![codecov](https://codecov.io/gh/eustomaqua/EPFD/branch/master/graph/badge.svg)](https://codecov.io/gh/eustomaqua/EPFD) 
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/39ec3833188a4fefaab11a0a0df9c3b1)](https://www.codacy.com/manual/eustomaqua/EPFD?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=eustomaqua/EPFD&amp;utm_campaign=Badge_Grade) 

Codes of Paper: [Ensemble Pruning based on Objection Maximization with a General Distributed Framework](https://arxiv.org/abs/1806.04899)

Including:
- Centralized Objection Maximization for Ensemble Pruning (COMEP)
- Distributed Objection Maximization for Ensemble Pruning (DOMEP)
- Ensemble Pruning Framework in a Distributed Setting (EPFD)

## Dependencies

- Create a virtual environment
  ```shell
  $ conda create -n EPFD python=3.6
  $ source activate EPFD
  $ # source deactivate
  ```
  or
  ```shell
  $ virtualenv EPFD --python=/usr/bin/python3
  $ source EPFD/bin/activate
  $ # deactivate
  ```

- Install packages
  ```shell
  $ pip install -r requirements.txt
  $ git clone https://github.com/eustomaqua/PyEnsemble.git
  $ pip install -e ./PyEnsemble
  ```

## Examples

Dataset: iris

Optional Choices of Ensemble Pruning Methods:  
name-pru ![\Large in](https://latex.codecogs.com/svg.latex?\Large&space;\in) \['ES', 'KP', 'KL', 'RE', 'OO', 'DREP', 'SEP', 'OEP', 'PEP', 'COMEP', 'DOMEP'\]

e.g.,
```shell
$ python main.py --nb-cls 31 --nb-pru 7 --name-pru COMEP --lam 0.5 --m 2
$ python main.py --nb-cls 31 --nb-pru 7 --name-pru DOMEP --lam 0.5 --m 2
$ python main.py --nb-cls 31 --nb-pru 7 --name-pru PEP --distributed --m 2
```

## Cite
Please cite our paper if you use this repository
```bib
@article{8891828,
  title     = {Ensemble Pruning Based on Objection Maximization With a General Distributed Framework}, 
  author    = {Bian, Yijun and Wang, Yijun and Yao, Yaqiang and Chen, Huanhuan},
  journal   = {IEEE Transactions on Neural Networks and Learning Systems}, 
  year      = {2020},
  volume    = {31},
  number    = {9},
  pages     = {3766--3774},
  doi       = {10.1109/TNNLS.2019.2945116},
  publisher = {IEEE},
  url       = {https://ieeexplore.ieee.org/document/8891828},
}
```
