#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author:Axe Chen
import os
import xlrd
import pydicom
import numpy as np
from scipy.io import savemat
from skimage.feature import greycomatrix, greycoprops
from skimage.measure import shannon_entropy
import skimage
from PIL import Image
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

if not os.path.exists("texture"):
    os.mkdir("texture")
for subject in subjectList:
    for mol in ["CT", "PET"]:
        texture = []
        dcm = pydicom.dcmread("cropped/{subject}_{mol}.dcm".format(subject=subject, mol=mol))
        img = dcm.pixel_array
        # window = skimage.exposure.equalize_hist(img, nbins=65536)
        # inds = np.digitize(img, bins=np.arange(0, 65536, 32))
        img8 = Image.fromarray(np.uint8(img))
        img8.save("cropped/grey/{sub}_{mol}.jpg".format(sub=subject, mol=mol))
        img8 = np.uint8(img)
        glcm = greycomatrix(img8, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=np.max(img8)+1, normed=True, symmetric=False)
        for energy in greycoprops(glcm, 'energy')[0]:
            texture.append(energy)
        for contrast in greycoprops(glcm, 'contrast')[0]:
            texture.append(contrast)
        for i in range(4):
            texture.append(shannon_entropy(glcm[:, :, 0, i]))
        for homogeneity in greycoprops(glcm, 'homogeneity')[0]:
            texture.append(homogeneity)
        savemat("texture/{sub}_{mol}.mat".format(sub=subject, mol=mol), {"data": texture})