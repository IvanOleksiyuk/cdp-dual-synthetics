#!/usr/bin/env python3
import os
import shutil
import argparse

def rearrange_scan_runs(root_dir):
    all_runs_dir = os.path.join(root_dir, "all_runs")
    os.makedirs(all_runs_dir, exist_ok=True)

    run_dirs = [d for d in os.listdir(root_dir)
                if os.path.isdir(os.path.join(root_dir, d))
                and d.startswith("EpsonV850_run")]

    for run_dir in run_dirs:
        rcod_path = os.path.join(root_dir, run_dir, "rcod")
        if not os.path.isdir(rcod_path):
            continue

        for subfolder in os.listdir(rcod_path):
            source_subfolder_path = os.path.join(rcod_path, subfolder)
            if not os.path.isdir(source_subfolder_path):
                continue

            # Prepare destination folder
            dest_folder = os.path.join(all_runs_dir, subfolder)
            os.makedirs(dest_folder, exist_ok=True)

            # Determine the next available file index in destination
            existing_files = sorted([
                f for f in os.listdir(dest_folder) if f.lower().endswith(".tiff")
            ])
            next_index = len(existing_files) + 1

            for filename in sorted(os.listdir(source_subfolder_path)):
                if filename.lower().endswith(".tiff"):
                    src_file = os.path.join(source_subfolder_path, filename)
                    dest_filename = f"{next_index:04}.tiff"
                    dest_file = os.path.join(dest_folder, dest_filename)

                    shutil.copy2(src_file, dest_file)
                    print(f"Copied {src_file} → {dest_file}")
                    next_index += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rearrange scan runs into an 'all_runs' directory under the provided root directory.")
    parser.add_argument("root_directory", help="Path to the root directory containing EpsonV850_run* folders.")
    args = parser.parse_args()

    rearrange_scan_runs(args.root_directory)


#python src/helper_scripts/rearrange_scan_runs.py --root_directory="data/wifs2024dataset/wifs2024dataset/orig_scan/HPI55_printrun1_session2_InvercoteG/"
#python src/helper_scripts/rearrange_scan_runs.py --root_directory="data/wifs2024dataset/wifs2024dataset/orig_scan/HPI76_printrun1_session2_InvercoteG/"