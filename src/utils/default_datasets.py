"""
This module contains the definitions of datasets used in the project.
"""

# Dictionary mapping dataset names to their paths and structure
image_datasets = {
    "f76iP12w": {"rel_path": "/fake_phone/HPI76_printrun1_session2_InvercoteG_EHPI55/iPhone12Pro_LAB_run1_scale3_wide_2x_RC_office/rcod", "structure": "default"},
    "f76iP15w": {"rel_path": "/fake_phone/HPI76_printrun1_session2_InvercoteG_EHPI55/iPhone15ProMax_RC_run1_scale2_wide_2.5x_RC_office/rcod", "structure": "default"},
    "f76iP14w": {"rel_path": "/fake_phone/HPI76_printrun1_session2_InvercoteG_EHPI55/iPhone14Pro_LAB_run1_scale2_wide_3x_RC_office/rcod", "structure": "default"},
    "f76iP15m": {"rel_path": "/fake_phone/HPI76_printrun1_session2_InvercoteG_EHPI55/iPhone15ProMax_RC_run1_scale5_uwide_2.5x_RC_lamp/rcod", "structure": "default"},
    "f76iP14m": {"rel_path": "/fake_phone/HPI76_printrun1_session2_InvercoteG_EHPI55/iPhone14Pro_LAB_run1_scale5_uwide_2.5x_RC_overhead/rcod", "structure": "default"},
    "f76iPXSo": {"rel_path": "/fake_phone/HPI76_printrun1_session2_InvercoteG_EHPI55/iPhoneXS_LAB_run1_scale3_wide_2x_RC_office/rcod", "structure": "default"},
    
    "f55iP12w": {"rel_path": "/fake_phone/HPI55_printrun1_session2_InvercoteG_EHPI76/iPhone12Pro_LAB_run1_scale3_wide_2x_RC_office/rcod", "structure": "default"},
    "f55iP15w": {"rel_path": "/fake_phone/HPI55_printrun1_session2_InvercoteG_EHPI76/iPhone15ProMax_RC_run1_scale2_wide_2.5x_RC_office/rcod", "structure": "default"},
    "f55iP14w": {"rel_path": "/fake_phone/HPI55_printrun1_session2_InvercoteG_EHPI76/iPhone14Pro_LAB_run1_scale2_wide_3x_RC_office/rcod", "structure": "default"},
    "f55iP15m": {"rel_path": "/fake_phone/HPI55_printrun1_session2_InvercoteG_EHPI76/iPhone15ProMax_RC_run1_scale5_uwide_2.5x_RC_lamp/rcod", "structure": "default"},
    "f55iP14m": {"rel_path": "/fake_phone/HPI55_printrun1_session2_InvercoteG_EHPI76/iPhone14Pro_LAB_run1_scale5_uwide_2.5x_RC_overhead/rcod", "structure": "default"},
    "f55iPXSo": {"rel_path": "/fake_phone/HPI55_printrun1_session2_InvercoteG_EHPI76/iPhoneXS_LAB_run1_scale3_wide_2x_RC_office/rcod", "structure": "default"},

    "o76iP12w": {"rel_path": "/orig_phone/HPI76_printrun1_session2_InvercoteG/iPhone12Pro_LAB_run1_scale3_wide_2x_RC_office/rcod", "structure": "default", "cond_idx": 13},
    "o76iP15w": {"rel_path": "/orig_phone/HPI76_printrun1_session2_InvercoteG/iPhone15ProMax_RC_run1_scale2_wide_2.5x_RC_office/rcod", "structure": "default", "cond_idx": 14},
    "o76iP14w": {"rel_path": "/orig_phone/HPI76_printrun1_session2_InvercoteG/iPhone14Pro_LAB_run1_scale2_wide_3x_RC_office/rcod", "structure": "default", "cond_idx": 15},
    "o76iP15m": {"rel_path": "/orig_phone/HPI76_printrun1_session2_InvercoteG/iPhone15ProMax_RC_run1_scale5_uwide_2.5x_RC_lamp/rcod", "structure": "default", "cond_idx": 16},
    "o76iP14m": {"rel_path": "/orig_phone/HPI76_printrun1_session2_InvercoteG/iPhone14Pro_LAB_run1_scale5_uwide_2.5x_RC_overhead/rcod", "structure": "default", "cond_idx": 17},
    "o76iPXSo": {"rel_path": "/orig_phone/HPI76_printrun1_session2_InvercoteG/iPhoneXS_LAB_run1_scale3_wide_2x_RC_office/rcod", "structure": "default", "cond_idx": 18},
    
    "o55iP12w": {"rel_path": "/orig_phone/HPI55_printrun1_session2_InvercoteG/iPhone12Pro_LAB_run1_scale3_wide_2x_RC_office/rcod", "structure": "default", "cond_idx": 3},
    "o55iP15w": {"rel_path": "/orig_phone/HPI55_printrun1_session2_InvercoteG/iPhone15ProMax_RC_run1_scale2_wide_2.5x_RC_office/rcod", "structure": "default", "cond_idx": 4},
    "o55iP14w": {"rel_path": "/orig_phone/HPI55_printrun1_session2_InvercoteG/iPhone14Pro_LAB_run1_scale2_wide_3x_RC_office/rcod", "structure": "default", "cond_idx": 5},
    "o55iP15m": {"rel_path": "/orig_phone/HPI55_printrun1_session2_InvercoteG/iPhone15ProMax_RC_run1_scale5_uwide_2.5x_RC_lamp/rcod", "structure": "default", "cond_idx": 6},
    "o55iP14m": {"rel_path": "/orig_phone/HPI55_printrun1_session2_InvercoteG/iPhone14Pro_LAB_run1_scale5_uwide_2.5x_RC_overhead/rcod", "structure": "default", "cond_idx": 7},
    "o55iPXSo": {"rel_path": "/orig_phone/HPI55_printrun1_session2_InvercoteG/iPhoneXS_LAB_run1_scale3_wide_2x_RC_office/rcod", "structure": "default", "cond_idx": 8},

    "o55Epson": {"rel_path": "/orig_scan/HPI55_printrun1_session2_InvercoteG/all_runs", "structure": "default", "cond_idx": 1},
    "o76Epson": {"rel_path": "/orig_scan/HPI76_printrun1_session2_InvercoteG/all_runs", "structure": "default", "cond_idx": 11},

    "f55Epson": {"rel_path": "/fake_scan/HPI55_printrun1_session2_InvercoteG_EHPI76/EpsonV850_run1_scandpi2400_scale3/rcod", "structure": "default"},
    "f76Epson": {"rel_path": "/fake_scan/HPI76_printrun1_session2_InvercoteG_EHPI55/EpsonV850_run1_scandpi2400_scale3/rcod", "structure": "default"},

    "tem": {"rel_path": "orig_template/rcod", "structure": "template", "cond_idx": 0},
    
    #original dataset for many experiments below
    "o55Epson_prun1_sc1": {"rel_path": "orig_scan/HPI55_printdpi812.8_printrun1_session1_InvercoteG/scanrun1_scandpi2400/rcod", "structure": "template"},
    
    # possible reference ensimble using multiple prints
    "o55Epson_prun2_sc1": {"rel_path": "orig_scan/HPI55_printdpi812.8_printrun2_session1_InvercoteG/scanrun1_scandpi2400/rcod", "structure": "template"},
    "o55Epson_prun3_sc1": {"rel_path": "orig_scan/HPI55_printdpi812.8_printrun3_session1_InvercoteG/scanrun1_scandpi2400/rcod", "structure": "template"},
    "o55Epson_prun4_sc1": {"rel_path": "orig_scan/HPI55_printdpi812.8_printrun4_session1_InvercoteG/scanrun1_scandpi2400/rcod", "structure": "template"},
    "o55Epson_prun5_sc1": {"rel_path": "orig_scan/HPI55_printdpi812.8_printrun5_session1_InvercoteG/scanrun1_scandpi2400/rcod", "structure": "template"},
    "o55Epson_prun6_sc1": {"rel_path": "orig_scan/HPI55_printdpi812.8_printrun6_session1_InvercoteG/scanrun1_scandpi2400/rcod", "structure": "template"},
    "o55Epson_prun7_sc1": {"rel_path": "orig_scan/HPI55_printdpi812.8_printrun7_session1_InvercoteG/scanrun1_scandpi2400/rcod", "structure": "template"},
    "o55Epson_prun8_sc1": {"rel_path": "orig_scan/HPI55_printdpi812.8_printrun8_session1_InvercoteG/scanrun1_scandpi2400/rcod", "structure": "template"},
    "o55Epson_prun9_sc1": {"rel_path": "orig_scan/HPI55_printdpi812.8_printrun9_session1_InvercoteG/scanrun1_scandpi2400/rcod", "structure": "template"},
    
    # possible reference ensemble using multiple scans
    "o55Epson_prun1_sc2": {"rel_path": "orig_scan/HPI55_printdpi812.8_printrun1_session1_InvercoteG/scanrun2_scandpi2400/rcod", "structure": "template"},
    "o55Epson_prun1_sc3": {"rel_path": "orig_scan/HPI55_printdpi812.8_printrun1_session1_InvercoteG/scanrun3_scandpi2400/rcod", "structure": "template"},
    "o55Epson_prun1_sc4": {"rel_path": "orig_scan/HPI55_printdpi812.8_printrun1_session1_InvercoteG/scanrun4_scandpi2400/rcod", "structure": "template"},
    
    # Two possible fakes
    "f55Epson_prun1_sc1": {"rel_path": "fake_scan/HPI55_printdpi812.8_printrun1_session1_InvercoteG_EHPI55/scanrun1_scandpi2400/rcod", "structure": "template"},
    "f55Epson_prun1_sc1_76": {"rel_path": "fake_scan/HPI55_printdpi812.8_printrun1_session1_InvercoteG_EHPI76/scanrun1_scandpi2400/rcod", "structure": "template"},
}

# Maps original dataset names to their corresponding fake dataset names
all_original_to_fake_pairs = {
    "o76iP12w": "f76iP12w",
    "o76iP15w": "f76iP15w",
    "o76iP14w": "f76iP14w",
    "o76iP15m": "f76iP15m",
    "o76iP14m": "f76iP14m",
    "o76iPXSo": "f76iPXSo",
    "o55iP12w": "f55iP12w",
    "o55iP15w": "f55iP15w",
    "o55iP14w": "f55iP14w",
    "o55iP15m": "f55iP15m",
    "o55iP14m": "f55iP14m",
    "o55iPXSo": "f55iPXSo",
    "o55Epson": "f55Epson",
    "o76Epson": "f76Epson",
}

# Maps fake dataset names to their corresponding original dataset names
all_fake_to_original_pairs = {fake: original for original, fake in all_original_to_fake_pairs.items()}

# Helper function to get the corresponding dataset name
def get_corresponding_dataset(dataset_name):
    """
    Get the corresponding dataset name (original→fake or fake→original).
    
    Args:
        dataset_name (str): The dataset name to find the corresponding pair for.
        
    Returns:
        str or None: The corresponding dataset name, or None if no pair exists.
    """
    if dataset_name.startswith('o'):
        return original_to_fake_pairs.get(dataset_name)
    elif dataset_name.startswith('f'):
        return fake_to_original_pairs.get(dataset_name)
    return None


def get_default_class_mapping():
    """
    Get the default class mapping for dataset suffixes.
    
    Args:
        allowed_suffixes (list, optional): List of allowed suffixes.
        fake_suffixes (list, optional): List of fake suffixes.
        
    Returns:
        dict: Mapping of class labels to integers.
    """
    class_mapping={}
    for image_dataset_name, image_dataset in image_datasets.items():
        if "cond_idx" in image_dataset:
            class_mapping[image_dataset_name] = image_dataset["cond_idx"]
    return class_mapping