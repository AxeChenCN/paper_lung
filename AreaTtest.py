#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author:Axe Chen
import xlrd
import numpy as np
from scipy import stats
import pydicom
import logging

# use logging module for easy debug
logging.basicConfig(format='%(asctime)s %(levelname)8s: %(message)s', datefmt='%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')

# 读被试表
workbook = xlrd.open_workbook("subjects.xls", 'r')
sheet_pt = workbook.sheet_by_name("pt")
sheet_spn = workbook.sheet_by_name("spn")
subject_pt = sheet_pt.col_values(1)
subject_spn = sheet_spn.col_values(1)
subjectList = subject_pt + subject_spn

pt = []
spn = []
for subject in subjectList:
    shape = []
    dcm = pydicom.dcmread("cropped/{subject}_CT.dcm".format(subject=subject))
    test = dcm.PixelSpacing
    area_sum = 64 * 64 * dcm.PixelSpacing[0]
    img = dcm.pixel_array
    pixel_num = len(img[img > 0])
    proportion = pixel_num / (64*64)
    area = proportion * area_sum
    if subject.startswith("sub_0"):
        pt.append(area)
    else:
        spn.append(area)
rvs0 = np.array(pt)
rvs1 = np.array(spn)
print("pt的area最大最小值为：{max}, {min}".format(max=max(rvs0), min=min(rvs0)))
print("{num}个pt的area均值和标准差为：".format(num=len(pt)))
print(rvs0.mean(), "±", rvs0.std(ddof=1))
print("spn的area最大最小值为：{max}, {min}".format(max=max(rvs1), min=min(rvs1)))
print("{num}个spn的area均值和标准差为：".format(num=len(spn)))
print(rvs1.mean(), "±", rvs1.std(ddof=1))
LeveneResult = stats.levene(rvs0, rvs1)
if LeveneResult[1] > 0.05:
    Ttest_indResult = stats.ttest_ind(rvs0, rvs1)
    print("pt和spn间FD值t检验的p值为："+str(Ttest_indResult[1]))
else:
    print("两对照组不具有方差齐性")