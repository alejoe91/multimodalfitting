import numpy as np
import json
import os

from bluepyopt.ephys.extra_features_utils import calculate_features, all_1D_features
import time


def compute_extra_features(eap, fs, upsample=1, feature_list=None):
    if feature_list is None:
        feature_list = all_1D_features
    else:
        for feat_name in feature_list:
            assert feat_name in all_1D_features, f"{feat_name} is not in available feature list {all_1D_features}"

    features = calculate_features(eap, sampling_frequency=fs, feature_names=feature_list)

    return features
