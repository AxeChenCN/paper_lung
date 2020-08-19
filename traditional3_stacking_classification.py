#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author:Axe Chen
import xlwt
import xlrd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE, f_regression, SelectKBest
from sklearn import svm
from scipy import interp
from scipy.io import loadmat, savemat
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import os
import logging
import warnings

# use logging module for easy debug
logging.basicConfig(format='%(asctime)s %(levelname)8s: %(message)s', datefmt='%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')
warnings.filterwarnings("ignore")

# 读被试表
workbook = xlrd.open_workbook("subjects.xls", 'r')
sheet_pt = workbook.sheet_by_name("pt")
sheet_spn = workbook.sheet_by_name("spn")
subject_pt = sheet_pt.col_values(1)
subject_spn = sheet_spn.col_values(1)
subjectList = subject_pt + subject_spn

full_name = "intensity+texture+shape_stacking"


# 解析.mat文件，组合xy
def generate_x_y():
    x1 = np.empty((len(subjectList), 9))
    x2 = np.empty((len(subjectList), 9))
    x3 = np.empty((len(subjectList), 16))
    x4 = np.empty((len(subjectList), 16))
    x5 = np.empty((len(subjectList), 8))
    for subject_id in range(len(subjectList)):
        subject = subjectList[subject_id]
        data1 = loadmat("intensity/{sub}_CT.mat".format(sub=subject))["data"].reshape(-1)
        data2 = loadmat("intensity/{sub}_PET.mat".format(sub=subject))["data"].reshape(-1)
        data3 = loadmat("texture/{sub}_CT.mat".format(sub=subject))["data"].reshape(-1)
        data4 = loadmat("texture/{sub}_PET.mat".format(sub=subject))["data"].reshape(-1)
        data5 = loadmat("shape/{sub}.mat".format(sub=subject))["data"].reshape(-1)
        x1[subject_id] = data1.copy()
        x2[subject_id] = data2.copy()
        x3[subject_id] = data3.copy()
        x4[subject_id] = data4.copy()
        x5[subject_id] = data5.copy()
    y = np.asarray([1]*len(subject_pt) + [0]*len(subject_spn)).astype('float32')
    return [x1, x2, x3, x4, x5], y


def generate_train_test(x, y, seed):
    skf = StratifiedKFold(n_splits=10, random_state=seed, shuffle=True)
    train_test_index = skf.split(x, y)
    return train_test_index


def model_evaluate(test_label, res_label):
    tp = tn = fp = fn = 0
    for j in range(len(test_label)):
        if test_label[j] == 0 and res_label[j] == 0:
            tn += 1
        if test_label[j] == 0 and res_label[j] == 1:
            fp += 1
        if test_label[j] == 1 and res_label[j] == 0:
            fn += 1
        if test_label[j] == 1 and res_label[j] == 1:
            tp += 1
    acc = (tp+tn)/(tp+fp+fn+tn)
    sen = tp/(tp+fn)
    spe = tn/(fp+tn)
    return acc, sen, spe


mean_fpr = np.linspace(0, 1, 100)
ACCs = []
SENs = []
SPEs = []
AUCs = []
tprs = []
contents = []
for iterCount in range(10):
    X, Y = generate_x_y()
    trainAndTestIndex = generate_train_test(X[0], Y, iterCount)
    fold = 0
    for trainIndex, testIndex in trainAndTestIndex:         # 十折交叉验证
        train_reses = np.empty((len(X), len(trainIndex)))
        test_reses = np.empty((len(X), len(testIndex)))
        y_train, y_test = Y[trainIndex], Y[testIndex]
        for feature_id in range(len(X)):
            X_train, X_test = X[feature_id][trainIndex], X[feature_id][testIndex]
            if feature_id != 4:
                mean = np.mean(X_train, axis=0)     # 列归一化
                std = np.std(X_train, axis=0)
                X_train_nol = (X_train - mean)/std
                X_test_nol = (X_test - mean)/std
            else:
                X_train_nol = X_train.copy()
                X_test_nol = X_test.copy()
            acc_max = 0
            sen_max = 0
            featureNumOp = 10
            COp = 1
            if not os.path.exists("selected"):
                os.mkdir("selected")
            for featureNum in range(2, X_train_nol.shape[1], 1):
                for C in np.logspace(-4, 4, 9, base=2):
                    lr = LinearRegression()
                    rfe = RFE(lr, n_features_to_select=featureNum)
                    rfe.fit(X_train_nol, y_train)
                    selected = rfe.support_
                    clf = svm.SVC(kernel='linear', C=C, max_iter=15000).fit(X_train_nol[:, selected], y_train)
                    score = clf.score(X_test_nol[:, selected], y_test)
                    y_score = clf.decision_function(X_test_nol[:, selected])
                    res = clf.predict(X_test_nol[:, selected])
                    ACC, SEN, SPE = model_evaluate(test_label=y_test, res_label=res)
                    if score > acc_max:
                        COp = C
                        featureNumOp = featureNum
                        acc_max = score
                        sen_max = SEN
                    if score == acc_max:
                        if SEN > sen_max:
                            COp = C
                            featureNumOp = featureNum
                            acc_max = score
                            sen_max = SEN
            lr = LinearRegression()
            rfe = RFE(lr, n_features_to_select=featureNumOp)
            rfe.fit(X_train_nol, y_train)
            selected = rfe.support_
            clf = svm.SVC(kernel='linear', C=COp, max_iter=15000).fit(X_train_nol[:, selected], y_train)
            test_res = clf.predict(X_test_nol[:, selected])
            test_reses[feature_id] = test_res
            train_res = clf.predict(X_train_nol[:, selected])
            train_reses[feature_id] = train_res
        acc_max = 0
        sen_max = 0
        COp = 1
        gammaOp = 1
        X_train_fuse = train_reses.T
        X_test_fuse = test_reses.T
        for C in np.logspace(-4, 4, 9, base=2):
            clf = svm.SVC(kernel='linear', C=C, max_iter=15000).fit(X_train_fuse, y_train)
            score = clf.score(X_test_fuse, y_test)
            res = clf.predict(X_test_fuse)
            ACC, SEN, SPE = model_evaluate(test_label=y_test, res_label=res)
            if score > acc_max:
                COp = C
                acc_max = score
                sen_max = SEN
            if score == acc_max:
                if SEN > sen_max:
                    COp = C
                    acc_max = score
                    sen_max = SEN
        clf = svm.SVC(kernel='linear', C=COp, max_iter=15000).fit(X_train_fuse, y_train)
        score = clf.score(X_test_fuse, y_test)
        y_score = clf.decision_function(X_test_fuse)
        res = clf.predict(X_test_fuse)
        ACC, SEN, SPE = model_evaluate(test_label=y_test, res_label=res)
        fpr, tpr, threshold = roc_curve(y_test, y_score)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        logger.debug("ACC={ACC}, SEN={SEN}, SPE={SPE}, AUC={AUC}".format(ACC=ACC, SEN=SEN, SPE=SPE, AUC=roc_auc))
        content = [ACC, SEN, SPE, roc_auc]
        contents.append(content)
        ACCs.append(ACC)
        SENs.append(SEN)
        SPEs.append(SPE)
        AUCs.append(roc_auc)
        fold += 1
        '''plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.show()'''
logger.debug("mean: ACC={ACC}±{ACC_std}, SEN={SEN}±{SEN_std}, SPE={SPE}±{SPE_std}, AUC={AUC}±{AUC_std}".format(ACC=sum(ACCs)/len(ACCs), ACC_std=np.std(ACCs),
                                                                                                               SEN=sum(SENs)/len(SENs), SEN_std=np.std(SENs),
                                                                                                               SPE=sum(SPEs)/len(SPEs), SPE_std=np.std(SPEs),
                                                                                                               AUC=sum(AUCs)/len(AUCs), AUC_std=np.std(AUCs)))
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8)
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
plt.plot(mean_fpr, mean_tpr, color='b', lw=2, alpha=.8)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.show()
if not os.path.exists("ROC"):
    os.mkdir("ROC")
savemat("ROC/{f}_fpr.mat".format(f=full_name), {'data': mean_fpr})
savemat("ROC/{f}_tpr.mat".format(f=full_name), {'data': mean_tpr})

workbook = xlwt.Workbook()
sheet = workbook.add_sheet('sheet1', cell_overwrite_ok=True)
head = ('ACC', 'SEN', 'SPE', 'AUC')
for col in range(len(head)):
    sheet.write(0, col, head[col])
for i in range(len(contents)):
    for col in range(len(contents[i])):
        sheet.write(i+1, col, contents[i][col])
workbook.save('Results/{f}.xls'.format(f=full_name))