#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author:Axe Chen
import math
import numpy as np
import logging

# use logging module for easy debug
logging.basicConfig(format='%(asctime)s %(levelname)8s: %(message)s', datefmt='%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')


def get_hu(dcm):
    img = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
    return img


def get_suv(dcm):
    dose = float(dcm.RadiopharmaceuticalInformationSequence._list[0].RadionuclideTotalDose)
    half_life = float(dcm.RadiopharmaceuticalInformationSequence._list[0].RadionuclideHalfLife)
    acq_time = float(dcm.AcquisitionTime)
    start_time = float(dcm.RadiopharmaceuticalInformationSequence._list[0].RadiopharmaceuticalStartTime)
    weight = dcm.PatientWeight
    use_time = acq_time - start_time
    actual_activity = dose * (np.exp(-use_time * math.log(2) / half_life))
    suv_factor = weight * 1000 / actual_activity
    img = (dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept) * suv_factor
    return img