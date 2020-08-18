#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author:Axe Chen
import os
import xlrd
import pydicom
from tools import get_hu, get_suv
import numpy as np
import pandas as pd
from scipy.io import savemat
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

if not os.path.exists("intensity"):
    os.mkdir("intensity")
for subject in subjectList:
    for mol in ["CT", "PET"]:
        intensity = []
        dcm = pydicom.dcmread("cropped/{subject}_{mol}.dcm".format(subject=subject, mol=mol))
        if mol == "CT":
            img = get_hu(dcm)
        else:
            img = get_suv(dcm)
        