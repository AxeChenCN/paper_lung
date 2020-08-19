#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author:Axe Chen
import os
import xlrd
from scipy.io import loadmat, savemat
import logging

# use logging module for easy debug
logging.basicConfig(format='%(asctime)s %(levelname)8s: %(message)s', datefmt='%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')

if not os.path.exists("deep"):
    os.mkdir("deep")

# 读被试表
workbook = xlrd.open_workbook("subjects.xls", 'r')
sheet_pt = workbook.sheet_by_name("pt")
sheet_spn = workbook.sheet_by_name("spn")
subject_pt = sheet_pt.col_values(1)
subject_spn = sheet_spn.col_values(1)
subjectList = subject_pt + subject_spn

deeps = loadmat("feature1024_2.mat")['feature']
for i in range(len(deeps[:47])):
    deep = deeps[i].reshape(-1)
    savemat("deep/{sub}.mat".format(sub=subject_spn[i]), {"data": deep})
for j in range(len(deeps[47:])):
    deep = deeps[47+j].reshape(-1)
    savemat("deep/{sub}.mat".format(sub=subject_pt[j]), {"data": deep})