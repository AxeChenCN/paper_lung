#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author:Axe Chen
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
import logging

# use logging module for easy debug
logging.basicConfig(format='%(asctime)s %(levelname)8s: %(message)s', datefmt='%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')

names = ["intensity_CT", "intensity_PET", "texture_CT", "texture_PET", "shape", "deep",
         "intensity_CT+PET_concatenating", "texture_CT+PET_concatenating", "intensity+texture+shape_concatenating", "proposed_concatenating",
         "intensity_CT+PET_svoting", "texture_CT+PET_svoting", "intensity+texture+shape_svoting", "proposed_svoting",
         "intensity_CT+PET_wsvoting", "texture_CT+PET_wsvoting", "intensity+texture+shape_wsvoting", "proposed_wsvoting",
         "intensity_CT+PET_stacking", "texture_CT+PET_stacking", "intensity+texture+shape_stacking", "proposed_stacking"]
fig = plt.figure(figsize=(20, 9))
ax = plt.subplot(111)
for name in names:
    fpr = loadmat("ROC/{name}_fpr.mat".format(name=name))["data"].reshape(-1)
    tpr = loadmat("ROC/{name}_tpr.mat".format(name=name))["data"].reshape(-1)
    if name.startswith("proposed"):
        ax.plot(fpr, tpr, lw=5, label=name, linestyle="-")
    elif name.startswith("intensity+texture+shape"):
        ax.plot(fpr, tpr, lw=5, label=name, linestyle="--")
    else:
        ax.plot(fpr, tpr, lw=5, label=name, linestyle="-.")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., ncol=4)
fig.subplots_adjust(right=0.5)
plt.savefig("D:/paper_lung/ROC.png", dpi=1000)
plt.show()