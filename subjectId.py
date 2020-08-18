#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author:Axe Chen
import os
import xlwt
import cv2
import numpy as np
import pydicom
import logging

# use logging module for easy debug
logging.basicConfig(format='%(asctime)s %(levelname)8s: %(message)s', datefmt='%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')


if not os.path.exists("cropped"):
    os.mkdir("cropped")
workbook = xlwt.Workbook()
for subject_class in ["pt", "spn"]:
    contents = []
    subjects = os.listdir(subject_class)
    sheet = workbook.add_sheet(subject_class, cell_overwrite_ok=True)
    for i in range(len(subjects)):
        subject = subjects[i]
        if subject_class == "pt":
            subject_id = "sub_" + str(i+1).rjust(3, "0")
        else:
            subject_id = "sub_" + str(i + 101).rjust(3, "0")
        contents.append((subject, subject_id))
        for modality in ["CT", "PET"]:
            dcm = pydicom.dcmread("{cla}/{subject}/a{mol}.dcm".format(cla=subject_class, subject=subject, mol=modality))
            locations = np.argwhere(dcm.pixel_array > 0)
            location_max = np.max(locations, axis=0)
            location_min = np.min(locations, axis=0)
            central = (location_max + location_min)//2
            y_max = central[0] + 32
            y_min = central[0] - 32
            x_max = central[1] + 32
            x_min = central[1] - 32
            cropped = dcm.pixel_array[y_min:y_max, x_min:x_max]
            dcm.Rows, dcm.Columns = cropped.shape
            dcm.PixelData = cropped.tobytes()
            # cropped[cropped == -1024] = 255
            dcm.save_as("cropped/{sub}_{mol}.dcm".format(sub=subject_id, mol=modality))
    for i in range(len(contents)):
        for col in range(len(contents[i])):
            sheet.write(i, col, contents[i][col])
workbook.save('subjects.xls')