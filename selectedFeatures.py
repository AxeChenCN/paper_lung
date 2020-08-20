#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author:Axe Chen
import xlrd
from scipy.io import loadmat
from scipy import stats
import numpy as np
import logging

# use logging module for easy debug
logging.basicConfig(format='%(asctime)s %(levelname)8s: %(message)s', datefmt='%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')

features = ["max_CT", "min_CT", "mean_CT", "std_CT", "skewness_CT", "kurtosis_CT", "upper quartile_CT", "lower quartile_CT", "IQR_CT",
            "max_PET", "min_PET", "mean_PET", "std_PET", "skewness_PET", "kurtosis_PET", "upper quartile_PET", "lower quartile_PET", "IQR_PET",
            "energy0_CT", "contrast0_CT", "entropy0_CT", "homogeneity0_CT",
            "energy45_CT", "contrast45_CT", "entropy45_CT", "homogeneity45_CT",
            "energy90_CT", "contrast90_CT", "entropy90_CT", "homogeneity90_CT",
            "energy135_CT", "contrast135_CT", "entropy135_CT", "homogeneity135_CT",
            "energy0_PET", "contrast0_PET", "entropy0_PET", "homogeneity0_PET",
            "energy45_PET", "contrast45_PET", "entropy45_PET", "homogeneity45_PET",
            "energy90_PET", "contrast90_PET", "entropy90_PET", "homogeneity90_PET",
            "energy135_PET", "contrast135_PET", "entropy135_PET", "homogeneity135_PET",
            "shape1", "shape2", "shape3", "shape4", "shape5", "shape6", "shape7", "shape8"] + list(range(1024))
countMap = {}
for feature in features:
    countMap[feature] = 0
    
for iterId in range(10):
    for foldId in range(10):
        select1 = loadmat("D:\PycharmProjects\lung\selected/selected1_{iter}_{fold}.mat".format(iter=iterId, fold=foldId))["data"].reshape(-1)
        select2 = loadmat("D:\PycharmProjects\lung\selected/selected2_{iter}_{fold}.mat".format(iter=iterId, fold=foldId))["data"].reshape(-1)
        selectC1 = np.argwhere(select1 == 1).reshape(-1)
        selectC2 = np.argwhere(select2 == 1).reshape(-1)
        selected = selectC1[selectC2].copy()
        for selectId in selected:
            countMap[features[selectId]] += 1
after = sorted(countMap.items(), key=lambda item: item[1], reverse=True)
print(after)

# 读被试表
workbook = xlrd.open_workbook("subjects.xls", 'r')
sheet_pt = workbook.sheet_by_name("pt")
sheet_spn = workbook.sheet_by_name("spn")
subject_pt = sheet_pt.col_values(1)
subject_spn = sheet_spn.col_values(1)
subjectList = subject_pt + subject_spn

pt = []
spn = []
features_intensity_PET = [3, 2, 6, 7]
for feature in features_intensity_PET:
    for subject in subject_pt:
        data = loadmat("intensity/{sub}_PET.mat".format(sub=subject))["data"].reshape(-1)
        pt.append(data[feature])
    for subject in subject_spn:
        data = loadmat("intensity/{sub}_PET.mat".format(sub=subject))["data"].reshape(-1)
        spn.append(data[feature])
    Ttest_indResult = stats.ttest_ind(np.array(pt), np.array(spn))
    print(feature, Ttest_indResult[1], Ttest_indResult[0] > 0)

features_texture_CT = [8, 9]
for feature in features_texture_CT:
    for subject in subject_pt:
        data = loadmat("texture/{sub}_CT.mat".format(sub=subject))["data"].reshape(-1)
        pt.append(data[feature])
    for subject in subject_spn:
        data = loadmat("texture/{sub}_CT.mat".format(sub=subject))["data"].reshape(-1)
        spn.append(data[feature])
    Ttest_indResult = stats.ttest_ind(np.array(pt), np.array(spn))
    print(feature, Ttest_indResult[1], Ttest_indResult[0] > 0)