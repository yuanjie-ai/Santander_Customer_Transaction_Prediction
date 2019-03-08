#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = 'utils'
__author__ = 'JieYuan'
__mtime__ = '19-3-8'
"""

import time
from functools import partial

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata
from tqdm import tqdm

# Load data
"""/home/yuanjie/desktop/santander"""


def read_files(files, **kwargs):
    """
    换行符 lineterminator='\n'
    :param files:
    :param kwargs:
    :return:
    """
    read_func = partial(pd.read_csv, **kwargs)
    return list(map(read_func, tqdm(files)))


# Submit
def submit(test, pred, path='pred.csv'):
    test[['ID_code']].assign(target=pred).to_csv(time.ctime() + ' ' + path, index=False)


# Feature engineering
## round_feats
def round_feats(df: pd.DataFrame, decimals=0):
    """
    # 0 0.8845091455713879
    # 1 0.8953696440390377
    # 2 0.8961562898660898
    # 3 0.8967054499954046
    # 4 0.8965129821339642
    """
    df = df.apply(lambda x: np.round(x, decimals))
    df.columns = df.columns + '_round_%s' % decimals
    return df


## power_rank_feats
def power_rank_feats(df):
    for col in tqdm(df.columns):
        # Normalize the data, so that it can be used in norm.cdf(),
        # as though it is a standard normal variable
        df[col] = (df[col] - df[col].mean()) / df[col].std()

        # Square
        df[col + '^2'] = df[col] ** 2

        # Cube
        df[col + '^3'] = df[col] ** 3

        # 4th power
        df[col + '^4'] = df[col] ** 4

        # Cumulative percentile (not normalized)
        df[col + '_cp'] = rankdata(df[col])

        # Cumulative normal percentile
        df[col + '_cnp'] = norm.cdf(df[col])
    return df

## 行聚合

## 行正/负个数: 加减乘除

## 聚类特征

## 异常值特征

## 半监督

## 高阶特征: nn/遗传编码

## 当回归做

## rank融合
