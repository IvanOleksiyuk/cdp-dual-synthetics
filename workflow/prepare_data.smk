### Data processing rules ###
datadir = config["datadir"]
prepared_dataset_dir = config["prepared_dataset_dir"]

rule all:
    input:
        f"{prepared_dataset_dir}cdp_transit_dataset_blockified.DONE"
    output:
        f"{prepared_dataset_dir}prepare_data.DONE"
    shell:
        f"touch {{output}}"

rule rearrange_scans:
    output:
        f"{datadir}rearranged_data.DONE",
        f"{datadir}wifs2024dataset/wifs2024dataset/orig_scan/HPI55_printrun1_session2_InvercoteG/all_runs/000145/0001.tiff",
        f"{datadir}wifs2024dataset/wifs2024dataset/orig_scan/HPI55_printrun1_session2_InvercoteG/all_runs/000145/0002.tiff"
    shell:
        f"python src/helper_scripts/rearrange_scan_runs.py \"{datadir}wifs2024dataset/wifs2024dataset/orig_scan/HPI55_printrun1_session2_InvercoteG/\" \n"
        f"python src/helper_scripts/rearrange_scan_runs.py \"{datadir}wifs2024dataset/wifs2024dataset/orig_scan/HPI76_printrun1_session2_InvercoteG/\" \n"
        f"touch {{output}}"

rule restructure_dataset_for_training:
    input:
        f"{datadir}rearranged_data.DONE",
        f"{datadir}wifs2024dataset/wifs2024dataset/orig_scan/HPI55_printrun1_session2_InvercoteG/all_runs/000145/0001.tiff",
        f"{datadir}wifs2024dataset/wifs2024dataset/orig_scan/HPI55_printrun1_session2_InvercoteG/all_runs/000145/0002.tiff"
    output:
        f"{prepared_dataset_dir}cdp_transport_dataset_default.DONE"
    shell:
        f"mkdir -p {prepared_dataset_dir} \n "
        f"python src/helper_scripts/restructure_dataset.py --original_dataset_path {datadir}wifs2024dataset/wifs2024dataset "
        f"--path_to_new_dataset {prepared_dataset_dir} "
        f"--train_range {config['train_range'][0]},{config['train_range'][1]} "
        f"--valid_range {config['valid_range'][0]},{config['valid_range'][1]} "
        f"--test_range {config['test_range'][0]},{config['test_range'][1]} \n"
        f"touch {{output}}"

rule blockify_dataset:
    input:
        f"{prepared_dataset_dir}cdp_transport_dataset_default.DONE"
    output:
        f"{prepared_dataset_dir}cdp_transit_dataset_blockified.DONE"
    shell:
        f"python src/helper_scripts/blockify_dataset.py --dataset_path={prepared_dataset_dir} \n"
        f"touch {{output}}"

### Rest of the pipeline ###

