#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author:Axe Chen
import xlrd
import os
import cv2
import numpy as np
from scipy.io import savemat
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

MIN_DESCRIPTOR = 8


def find_contours(laplacian):
    """获取连通域
    :param: 输入Laplacian算子（空间锐化滤波）
    :return: 最大连通域
    """
    # binaryimg = cv2.Canny(res, 50, 200) #二值化，canny检测
    h = cv2.findContours(laplacian, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)   # 寻找轮廓
    contour = h[0]
    contour = sorted(contour, key=cv2.contourArea, reverse=True)    # 对一系列轮廓点坐标按它们围成的区域面积进行排序
    return contour


def truncate_descriptor(fourier_result):
    """截短傅里叶描述子
    :param res: 输入傅里叶描述子
    :return: 截短傅里叶描述子
    """
    descriptors_in_use = np.fft.fftshift(fourier_result)

    # 取中间的MIN_DESCRIPTOR项描述子
    center_index = int(len(descriptors_in_use) / 2)
    low, high = center_index - int(MIN_DESCRIPTOR / 2), center_index + int(MIN_DESCRIPTOR / 2)
    descriptors_in_use = descriptors_in_use[low:high]
    
    descriptors_in_use = np.fft.ifftshift(descriptors_in_use)
    return descriptors_in_use


def fourier_descriptor(img):
    """计算傅里叶描述子
    :param res: 输入图片
    :return: 图像，描述子点
    """
    img[img > 0] = 1
    img = img.astype("uint8")
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    '''M = cv2.moments(contours[0])  # 计算第一条轮廓的各阶矩,字典形式
    center_x = int(M["m10"] / M["m00"])
    center_y = int(M["m01"] / M["m00"])'''  # 中心检测
    contour_array = contours[0][:, 0, :]     # 注意这里只保留区域面积最大的轮廓点坐标
    contours_complex = np.empty(contour_array.shape[:-1], dtype=complex)
    contours_complex.real = contour_array[:, 0]      # 横坐标作为实数部分
    contours_complex.imag = contour_array[:, 1]      # 纵坐标作为虚数部分
    fourier_result = np.fft.fft(contours_complex)       # 进行傅里叶变换
    # fourier_result = np.fft.fftshift(fourier_result)
    descirptor_in_use = truncate_descriptor(fourier_result)     # 截短傅里叶描述子
    return descirptor_in_use


if not os.path.exists("shape"):
    os.mkdir("shape")
for subject in subjectList:
    shape = []
    dcm = pydicom.dcmread("cropped/{subject}_CT.dcm".format(subject=subject))
    img = dcm.pixel_array
    fourier_results = abs(fourier_descriptor(img))
    fourier_results = (fourier_results-np.min(fourier_results)) / (np.max(fourier_results)-np.min(fourier_results))
    for descriptor in fourier_results:
        shape.append(descriptor)
    savemat("shape/{sub}.mat".format(sub=subject), {"data": shape})