#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = 'bayes'
__author__ = 'JieYuan'
__mtime__ = '19-3-8'
"""
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer
from xplan.models import OOF
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold

from Santander_Customer_Transaction_Prediction.models.utils import *

"""
https://www.kaggle.com/blackblitz/gaussian-naive-bayes/comments
"""
files = ['/home/yuanjie/desktop/santander/train.csv.zip', '/home/yuanjie/desktop/santander/test.csv.zip']
train, test = read_files(files)

X = train.iloc[:, 2:]
y = train['target']
X_test = test.iloc[:, 1:]

pipeline = make_pipeline(QuantileTransformer(output_distribution='normal'), GaussianNB())

folds1 = StratifiedKFold(10, True, 2019)
folds2 = RepeatedStratifiedKFold(5, 10, 2019)

OOF(pipeline, folds1).fit(X, y, X_test)

OOF(pipeline, folds2).fit(X, y, X_test)
