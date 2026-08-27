#imports
import os
import yaml
from pathlib import Path
from omegaconf import ListConfig, DictConfig

#includes
include: "rules/utils.smk"

if "config_dir" not in config:
    config["config_dir"] = os.path.join(os.getcwd(), "config")
configfile: os.path.join(config["config_dir"], "main.yaml")

if "train_datadir" not in config:
    config["train_datadir"] = "cdp_transport_dataset_default"

config = DictConfig(config)
hydra_config_dir = config.get("hydra_config_dir", False)

debug = config["debug"]

if debug:
    print("Running in DEBUG mode")
    config["wrkspdir"] = config["wrkspdir"][:-1] + "debug/"
    config['test_range'][0] = 145
    config['test_range'][1] = 150

weak_gpu = config.get("weak_gpu", False) if not debug else False
srcdir = config["srcdir"] 
wrkspdir = config["wrkspdir"]
datadir = config["datadir"]
wandb_key = config.get("wandb_key", False)
do_cleanup = config.get("do_cleanup", False)
print(f"W&B Key: {wandb_key}")

captures_used = config["captures_used"]
# Accept one or multiple modes (space-separated string or list)
_reference_eval_mode_raw = config.get("reference_eval_mode", "blocks_combine-1")
if isinstance(_reference_eval_mode_raw, (list, ListConfig)):
    reference_eval_modes = [str(x) for x in _reference_eval_mode_raw]
elif isinstance(_reference_eval_mode_raw, str):
    reference_eval_modes = _reference_eval_mode_raw.split()
else:
    reference_eval_modes = ["blocks_combine-1"]
# Keep first for any legacy fallback
reference_eval_mode = reference_eval_modes[0]

# I am doing this like this for now which is a quick and dirty way to handle ensemble seeds
# This should be improved in the future
n_ensemble_seeds = config.get("n_ensemble_seeds", 10)
ensemble_mode = config.get("ensemble_mode", "per_pixel_closest_reference")

models = config["models"]

# Define all reference result files in one place (per mode)
ref_results_by_mode = {bd: [] for bd in reference_eval_modes}
ref_exports = []  # (model_name, from_class, to_class) or (model_name, capture_item)

# Add digital template results if enabled
if config.get("do_digital_template_results", False):
    for bd in reference_eval_modes:
        ref_results_by_mode[bd] += [
            f"{wrkspdir}tem_reference/refperf_{bd}/refperf_tem_ref_{capture_item}.pkl"
            for capture_item in captures_used
        ]

if config.get("do_cross_camera_auth_results", False):
    for bd in reference_eval_modes:
        ref_results_by_mode[bd] += [
            f"{wrkspdir}cam_reference/refperf_{bd}/refperf_refcam_{capture_item}_allused.pkl"
            for capture_item in captures_used
        ]


# Add model-generated reference results for all models
for model_name, model_config in models.items():
    templates = model_config.templates

    # Identify the mode of tempalte generation
    if isinstance(templates, DictConfig):
        template_mode = templates.get("template_mode", None)
    elif isinstance(templates, ListConfig) and all(isinstance(t, ListConfig) and len(t) == 2 for t in templates):
        template_mode = "transport_old"
    elif templates == "singular":
        template_mode = "singular"
    elif isinstance(templates, ListConfig) or (isinstance(templates, list) and all(isinstance(t, str) for t in templates)):
        template_mode = "conditional_old" #this mode has to be abandoned in the future


    # Generation: singular or list of single classes
    if template_mode == "singular":
        for bd in reference_eval_modes:
            ref_results_by_mode[bd].append(f"{wrkspdir}{model_name}/refperf_{bd}/refperf_{model_name}_ref_allused.pkl")
            ref_exports.append((model_name, None, None))
    elif template_mode == "singular_seeds_pixel_min":
        for bd in reference_eval_modes:
            ref_results_by_mode[bd].append(f"{wrkspdir}{model_name}/refperf_{bd}/refperf_{model_name}_ref_allused_EEpixel_minEE.pkl")
            ref_exports.append((model_name, None, None))
    # Old config structure
    elif template_mode == "transport_old":
        for from_to in templates:
            from_class, to_class = from_to
            for bd in reference_eval_modes:
                ref_results_by_mode[bd].append(
                    f"{wrkspdir}{model_name}/refperf_{bd}/refperf_{model_name}_from_{from_class}_to_{to_class}_ref_allused.pkl"
                )
            ref_exports.append((model_name, from_class, to_class))
    elif template_mode == "conditional_old":
        for capture_item in templates:
            for bd in reference_eval_modes:
                ref_results_by_mode[bd].append(
                    f"{wrkspdir}{model_name}/refperf_{bd}/refperf_{model_name}_class_{capture_item}_ref_allused.pkl"
                )
            ref_exports.append((model_name, capture_item, None))
    # New config structure 
    elif template_mode == "conditional":
        for capture_item in templates.conditions:
            for bd in reference_eval_modes:
                ref_results_by_mode[bd].append(
                    f"{wrkspdir}{model_name}/refperf_{bd}/refperf_{model_name}_class_{capture_item}_ref_allused.pkl"
                )
            ref_exports.append((model_name, capture_item, None))
    elif template_mode == "transport":
        for from_to in templates.pairs:
            from_class, to_class = from_to
            for bd in reference_eval_modes:
                ref_results_by_mode[bd].append(
                    f"{wrkspdir}{model_name}/refperf_{bd}/refperf_{model_name}_from_{from_class}_to_{to_class}_ref_allused.pkl"
                )
            ref_exports.append((model_name, from_class, to_class))
    else:
        raise ValueError(f"Unknown template type for model {model_name}: {template_mode}")

#print(ref_results_by_mode)
# Define truncated reference result files per mode
ref_results_truncated_by_mode = {
    bd: [f"{wrkspdir}refperf_truncated/{bd}/{Path(ref).name}" for ref in refs]
    for bd, refs in ref_results_by_mode.items()
}
# Helper maps to address single truncated targets by basename
basenames_by_mode = {bd: [Path(p).name for p in refs] for bd, refs in ref_results_by_mode.items()}
input_map_by_mode = {bd: {Path(p).name: p for p in refs} for bd, refs in ref_results_by_mode.items()}
# These result will then be aggregated in one table at the end

# Use requested modes for per-mode targets
blocks_dirs = reference_eval_modes

# Targets for LaTeX tables (AUC and mean_ref_vs_orig) per blocks_combine-* folder
auc_tex_tables = [f"{wrkspdir}main_results/{bd}/auc_table.tex" for bd in blocks_dirs]
meanref_tex_tables = [f"{wrkspdir}main_results/{bd}/mean_ref_vs_orig_table.tex" for bd in blocks_dirs]

# DEV version
# rule all:
#     input:
#         f"{wrkspdir}l2cond_EpsoniP12iP14_dev/training.DONE",
#         f"{wrkspdir}l2cond_EpsoniP12iP14_dev/export.DONE"
#         #*[f"{wrkspdir}{model_name}/refperf_{model_name}_ref_allused.pkl" for model_name in config["models"]]

# MAIN version

all_input = []
if len(models)>0:
    all_input += [f"{wrkspdir}main_results/metric_summaries/min_val_loss_vs_model.png"]

rule all:
    input:
        # per-mode main plots
        expand(f"{wrkspdir}main_results/{{bd}}/auc_comparison_SSIM.png", bd=blocks_dirs),
        expand(f"{wrkspdir}main_results/{{bd}}/auc_comparison_PCC.png", bd=blocks_dirs),
        expand(f"{wrkspdir}main_results/{{bd}}/mean_ref_vs_orig_comparison_SSIM.png", bd=blocks_dirs),
        expand(f"{wrkspdir}main_results/{{bd}}/mean_ref_vs_orig_comparison_PCC.png", bd=blocks_dirs),
        expand(f"{wrkspdir}main_results/{{bd}}/sim_auc.tex", bd=blocks_dirs),
        # per-mode truncated plots
        expand(f"{wrkspdir}main_results_truncated/{{bd}}/auc_comparison_SSIM.png", bd=blocks_dirs),
        expand(f"{wrkspdir}main_results_truncated/{{bd}}/auc_comparison_PCC.png", bd=blocks_dirs),
        expand(f"{wrkspdir}main_results_truncated/{{bd}}/mean_ref_vs_orig_comparison_SSIM.png", bd=blocks_dirs),
        # tables
        *auc_tex_tables,
        *meanref_tex_tables,
        *all_input,
        # COVERAGE
        f"{wrkspdir}main_results_truncated/final_plots/metric_vs_coverage_all_plots.DONE"
        #*[f"{wrkspdir}{model_name}/refperf_{model_name}_ref_allused.pkl" for model_name in config["models"]]

# Generate LaTeX table from AUC summary JSON per blocks_combine-* folder
rule make_auc_table_tex:
    input:
        script=f"{srcdir}src/scripts/jsonresilts2latex.py",
        summary_json=lambda wildcards: f"{wrkspdir}main_results/{wildcards.blocks_dir}/auc_comparison_summary.json"
    output:
        table=f"{wrkspdir}main_results/{{blocks_dir}}/auc_table.tex"
    shell:
        f"python {{input.script}} --input {{input.summary_json}} --output {{output.table}}"

# Generate LaTeX table from mean_ref_vs_orig summary JSON per blocks_combine-* folder
rule make_meanref_table_tex:
    input:
        script=f"{srcdir}src/scripts/jsonresilts2latex.py",
        summary_json=lambda wildcards: f"{wrkspdir}main_results/{wildcards.blocks_dir}/mean_ref_vs_orig_comparison_summary.json"
    output:
        table=f"{wrkspdir}main_results/{{blocks_dir}}/mean_ref_vs_orig_table.tex"
    shell:
        f"python {{input.script}} --input {{input.summary_json}} --output {{output.table}}"

rule merge_tex_tables:
    input:
        tables=lambda wildcards: expand(f"{wrkspdir}main_results/{{blocks_dir}}/{{table}}", blocks_dir=blocks_dirs, table=[f"{wildcards.table}.tex"])
    output:
        merged_table=f"{wrkspdir}main_results/final_tables/{{table}}.tex"
    shell:
        """
        python merge_latex_tables.py \
            --table1 table1.tex \
            --table2 table2.tex \
            --out merged_table.tex \
            --label tab:sim_auc \
            --caption "Average similarity metrics and ROC AUCs..." \
            --xs_title "iPhone XS wide" \
            --p15_title "iPhone 15 Pro macro"
        """

# Merge mean_ref_vs_orig_table.tex and auc_table.tex into sim_auc.tex inside the same blocks_dir folder
rule make_sim_auc_table_tex:
    input:
        script=f"{srcdir}src/scripts/merge_latex_tables.py",
        mean_ref_table=f"{wrkspdir}main_results/{{blocks_dir}}/mean_ref_vs_orig_table.tex",
        auc_table=f"{wrkspdir}main_results/{{blocks_dir}}/auc_table.tex"
    output:
        sim_auc_table=f"{wrkspdir}main_results/{{blocks_dir}}/sim_auc.tex"
    shell:
        # Using example command provided by user, generalized with wildcards
        f"""
        python {{input.script}} \
          --table1 {{input.mean_ref_table}} \
          --table2 {{input.auc_table}} \
          --out {{output.sim_auc_table}} \
          --label tab:sim_auc \
          --xs_title \"iPhone XS wide\" \
          --p15_title \"iPhone 15 Pro macro\"
        """

rule cleanup:
    output:
        f"{wrkspdir}{{model_name_wc}}/cleanup.DONE"
    shell:
        "rm -r a.txt"


rule train_model:
    # This rule trains a model using the specified parameters
    threads: 16
    input:
        srcdir+"src/scripts/train_ptl.py",
    output:
        f"{wrkspdir}{{model_name_wc}}/checkpoints/best_model.ckpt",
        f"{wrkspdir}{{model_name_wc}}/checkpoints/last.ckpt",
        f"{wrkspdir}{{model_name_wc}}/training.DONE"
    resources:
        gpu=1,
        gres="gpu:1",
        slurm_extra="--gres=gpu:1 --nodes=1",
        cores=16,
        slurm_partition="gpu",  # adjust to your cluster's partition name
        mem_mb=parse_byte_count("20GB"),
        runtime=parse_timespan("20m"),
    shell:
        #f"{'HYDRA_FULL_ERROR=1' if debug else ''} " 
        f"CUDA_VISIBLE_DEVICES=0 python {srcdir}src/scripts/train_ptl.py"
        f" --config-dir={hydra_config_dir if hydra_config_dir else 'src/configs'}" 
        " --config-name={wildcards.model_name_wc}"
        f" paths.workdir_path={wrkspdir}"
        " paths.model_name={wildcards.model_name_wc}"
        f" debug={debug}"
        " export.reference_name={wildcards.model_name_wc}"
        f" ++wandb_key={wandb_key}"
        f" ++deterministic={config.get('deterministic', True)}"
        # Debug flag dependent part
        f" paths.dataset_path={datadir}{'cdp_transit_debug' if debug else config['train_datadir']}/"
        f" {'trainer.max_epochs=1' if debug else ''}"
        f" {'data.datamodule.batch_size=8' if debug else ''}"
        f" {'++tags=debug' if debug else config.get('model_tags_cmd', '')}"
        f" {'data.datamodule.batch_size=16' if weak_gpu else ''}"
        f" {'model.learning_rate=0.0005' if weak_gpu else ''}"
        f" {'++model.discriminator_lr=0.0005' if weak_gpu else ''}"

rule export_test_reference_singular:
    threads: 16
    input:
        f"{wrkspdir}{{model_name}}/checkpoints/last.ckpt",
        f"{wrkspdir}{{model_name}}/checkpoints/best_model.ckpt",
        srcdir+"src/scripts/export_test_reference.py",
        f"{wrkspdir}{{model_name}}/training.DONE"
    output:
        f"{wrkspdir}{{model_name}}/outputs/test_transformed/reference_{{model_name}}/export.DONE"
    resources:
        gpu=1,
        gres="gpu:1",
        slurm_extra="--gres=gpu:1 --nodes=1",
        cores=16,
        slurm_partition="gpu",  # adjust to your cluster's partition name
        mem_mb=parse_byte_count("20GB"),
        runtime=parse_timespan("20m"),
    shell:
        f"CUDA_VISIBLE_DEVICES=0 python {srcdir}src/scripts/export_test_reference.py"
        f" --config-dir={hydra_config_dir if hydra_config_dir else 'src/configs'}"
        f" --config-name={{wildcards.model_name}}"
        f" paths.workdir_path={wrkspdir}"
        f" paths.model_name={{wildcards.model_name}}"
        f" export.reference_name=reference_{{wildcards.model_name}}"
        f" ++wandb_key={wandb_key}"
        f" ++test_range=[{config['test_range'][0]},{config['test_range'][1]}]"
        f" paths.dataset_path={datadir}cdp_transit_{'debug' if debug else 'dataset'}/"
        f" ++block_side={config['eval_block_side']}"
        f" ++block_stride={config['eval_block_stride']}"
        f" ++image_side={config['eval_image_side']}"

rule export_test_reference_singular_seeds:
    threads: 16
    input:
        f"{wrkspdir}{{model_name}}/checkpoints/last.ckpt",
        f"{wrkspdir}{{model_name}}/checkpoints/best_model.ckpt",
        srcdir+"src/scripts/export_test_reference.py",
        f"{wrkspdir}{{model_name}}/training.DONE"
    output:
        f"{wrkspdir}{{model_name}}/outputs/test_transformed/reference_{{model_name}}/seed_{{generation_seed}}/export.DONE"
    resources:
        gpu=1,
        gres="gpu:1",
        slurm_extra="--gres=gpu:1 --nodes=1",
        cores=16,
        slurm_partition="gpu",  # adjust to your cluster's partition name
        mem_mb=parse_byte_count("20GB"),
        runtime=parse_timespan("20m"),
    shell:
        f"CUDA_VISIBLE_DEVICES=0 python {srcdir}src/scripts/export_test_reference.py"
        f" --config-dir={hydra_config_dir if hydra_config_dir else 'src/configs'}"
        f" --config-name={{wildcards.model_name}}"
        f" paths.workdir_path={wrkspdir}"
        f" paths.model_name={{wildcards.model_name}}"
        f" export.reference_name=reference_{{wildcards.model_name}}/seed_{{wildcards.generation_seed}}"
        f" ++wandb_key={wandb_key}"
        f" ++test_range=[{config['test_range'][0]},{config['test_range'][1]}]"
        f" paths.dataset_path={datadir}cdp_transit_{'debug' if debug else 'dataset'}/"
        f" ++export.generation_seed={{wildcards.generation_seed}}"
        f" ++block_side={config['eval_block_side']}"
        f" ++block_stride={config['eval_block_stride']}"
        f" ++image_side={config['eval_image_side']}"

rule export_test_reference_conditional:
    threads: 16
    input:
        f"{wrkspdir}{{model_name}}/checkpoints/last.ckpt",
        srcdir+"src/scripts/export_test_reference.py",
        f"{wrkspdir}{{model_name}}/training.DONE"
    output:
        f"{wrkspdir}{{model_name}}/outputs/test_transformed/reference_{{model_name}}_class_{{capture_item}}/export.DONE"
    resources:
        gpu=1,
        gres="gpu:1",
        slurm_extra="--gres=gpu:1 --nodes=1",
        cores=16,
        slurm_partition="gpu",  # adjust to your cluster's partition name
        mem_mb=parse_byte_count("20GB"),
        runtime=parse_timespan("20m"),
    shell:
        f"CUDA_VISIBLE_DEVICES=0 python {srcdir}src/scripts/export_test_reference.py"
        f" --config-dir={hydra_config_dir if hydra_config_dir else 'src/configs'}"
        f" --config-name={{wildcards.model_name}}"
        f" paths.workdir_path={wrkspdir}"
        f" paths.model_name={{wildcards.model_name}}"
        f" export.reference_name=reference_{{wildcards.model_name}}"
        f" ++export.capture_item={{wildcards.capture_item}}"
        f" ++wandb_key={wandb_key}"
        f" ++test_range=[{config['test_range'][0]},{config['test_range'][1]}]"
        f" paths.dataset_path={datadir}cdp_transit_{'debug' if debug else 'dataset'}/"
        f" ++block_side={config['eval_block_side']}"
        f" ++block_stride={config['eval_block_stride']}"
        f" ++image_side={config['eval_image_side']}"

rule export_test_reference_transport:
    threads: 16
    input:
        f"{wrkspdir}{{model_name}}/checkpoints/last.ckpt",
        srcdir+"src/scripts/export_test_reference_transport.py",
        f"{wrkspdir}{{model_name}}/training.DONE"
    output:
        f"{wrkspdir}{{model_name}}/outputs/test_transformed/reference_{{model_name}}_from_{{from_class}}_to_{{to_class}}/export.DONE",
    resources:
        gpu=1,
        gres="gpu:1",
        slurm_extra="--gres=gpu:1 --nodes=1",
        cores=16,
        slurm_partition="gpu",  # adjust to your cluster's partition name
        mem_mb=parse_byte_count("20GB"),
        runtime=parse_timespan("20m"),
    shell:
        f"CUDA_VISIBLE_DEVICES=0 python {srcdir}src/scripts/export_test_reference_transport.py"
        f" --config-dir={hydra_config_dir if hydra_config_dir else 'src/configs'}"
        f" --config-name={{wildcards.model_name}}"
        f" paths.workdir_path={wrkspdir}"
        f" paths.model_name={{wildcards.model_name}}"
        f" export.reference_name=reference_{{wildcards.model_name}}_from_{{wildcards.from_class}}_to_{{wildcards.to_class}}"
        f" ++wandb_key={wandb_key}"
        f" ++test_range=[{config['test_range'][0]},{config['test_range'][1]}]"
        f" paths.dataset_path={datadir}cdp_transit_{'debug' if debug else 'dataset'}/"
        f" +export.transport_pairs='[[\"{{wildcards.from_class}}\",\"{{wildcards.to_class}}\"]]'"
        f" ++block_side={config['eval_block_side']}"
        f" ++block_stride={config['eval_block_stride']}"
        f" ++image_side={config['eval_image_side']}"

rule referenece_eval_model:
    # This rule evaluates the performance of the trained model using the generated reference dataset
    input:
        f"{srcdir}src/scripts/reference_eval.py",
        f"{wrkspdir}{{model_name_wc}}/outputs/test_transformed/reference_{{reference_name}}/export.DONE"
    output:
        f"{wrkspdir}{{model_name_wc}}/refperf_{{blocks_dir}}/refperf_{{reference_name}}_ref_allused.pkl"
    shell:
        f"python {srcdir}src/scripts/reference_eval.py"
        f" --reference {wrkspdir}{{wildcards.model_name_wc}}/outputs/test_transformed/reference_{{wildcards.reference_name}}/"
        f" --output_dir {wrkspdir}{{wildcards.model_name_wc}}/refperf_{{wildcards.blocks_dir}}/"
        f" --plot_output_dir {wrkspdir}{{wildcards.model_name_wc}}/ref_eval_plots_{{wildcards.blocks_dir}}/reference_{{wildcards.reference_name}}/"
        f" --output_name refperf_{{wildcards.reference_name}}_ref_allused"
        f" --originals {' '.join(captures_used)}"
        f" --mode {{wildcards.blocks_dir}}"
        " --shot_probe=1"
        f" --reference_data_structure generated"
        f" --no_plots"
        f" --test_uid_range {config['test_range'][0]},{config['test_range'][1]}"
        f" --dataset_path {datadir}/wifs2024dataset/wifs2024dataset/"
        f" --block_side {config['eval_block_side']}"
        f" --block_stride {config['eval_block_stride']}"
        f" --image_side {config['eval_image_side']}"

rule referenece_eval_model_ensemble:
    # This rule evaluates the performance of the trained model using the generated reference dataset
    input:
        f"{srcdir}src/scripts/reference_eval.py",
        [f"{wrkspdir}{{model_name_wc}}/outputs/test_transformed/reference_{{reference_name}}/seed_{seed}/export.DONE" for seed in range(n_ensemble_seeds)]
    output:
        f"{wrkspdir}{{model_name_wc}}/refperf_{{blocks_dir}}/refperf_{{reference_name}}_ref_allused_EEpixel_minEE.pkl"
    shell:
        f"python {srcdir}src/scripts/reference_eval.py"
        f" --reference {' '.join([f'{wrkspdir}{{wildcards.model_name_wc}}/outputs/test_transformed/reference_{{wildcards.reference_name}}/seed_{seed}/' for seed in range(n_ensemble_seeds)])}/"
        f" --output_dir {wrkspdir}{{wildcards.model_name_wc}}/refperf_{{wildcards.blocks_dir}}/"
        f" --plot_output_dir {wrkspdir}{{wildcards.model_name_wc}}/ref_eval_plots_{{wildcards.blocks_dir}}/reference_{{wildcards.reference_name}}/"
        f" --output_name refperf_{{wildcards.reference_name}}_ref_allused_EEpixel_minEE"
        f" --originals {' '.join(captures_used)}"
        f" --mode {{wildcards.blocks_dir}}"
        " --shot_probe=1"
        f" --reference_data_structure generated"
        f" --no_plots"
        f" --test_uid_range {config['test_range'][0]},{config['test_range'][1]}"
        f" --dataset_path {datadir}/wifs2024dataset/wifs2024dataset/"
        f" --ensemble_mode {ensemble_mode}"
        f" --block_side {config['eval_block_side']}"
        f" --block_stride {config['eval_block_stride']}"
        f" --image_side {config['eval_image_side']}"

rule referene_eval_template:
    input:
        f"{srcdir}src/scripts/reference_eval.py",
    output:
        f"{wrkspdir}tem_reference/refperf_{{blocks_dir}}/refperf_tem_ref_{{capture}}.pkl"
    shell:
        f"python {srcdir}src/scripts/reference_eval.py"
        f" --reference tem"
        f" --output_dir {wrkspdir}tem_reference/refperf_{{wildcards.blocks_dir}}/"
        f" --plot_output_dir {wrkspdir}tem_reference/refperf_{{wildcards.blocks_dir}}/plots_{{wildcards.capture}}/"
        " --originals {wildcards.capture}"
        " --output_name refperf_tem_ref_{wildcards.capture}"
        f" --mode {{wildcards.blocks_dir}}"
        f" --no_plots"
        f" --shot_probe=1"
        f" --test_uid_range {config['test_range'][0]},{config['test_range'][1]}"
        f" --reference_data_structure template"
        f" --dataset_path {datadir}/wifs2024dataset/wifs2024dataset/"
        f" --block_side {config['eval_block_side']}"
        f" --block_stride {config['eval_block_stride']}"
        f" --image_side {config['eval_image_side']}"

rule reference_eval_other_camera:
    input:
        f"{srcdir}src/scripts/reference_eval.py",
    output:
        f"{wrkspdir}cam_reference/refperf_{{blocks_dir}}/refperf_refcam_{{capture}}_allused.pkl"
    shell:
        f"python {srcdir}src/scripts/reference_eval.py"
        " --reference {wildcards.capture}DUP"
        f" --output_dir {wrkspdir}cam_reference/refperf_{{wildcards.blocks_dir}}/"
        f" --originals {' '.join(captures_used)}"
        " --output_name refperf_refcam_{wildcards.capture}_allused"
        f" --mode {{wildcards.blocks_dir}}"
        " --shot_probe=1"
        " --shot_reference=2"
        #f" --no_plots"
        f" --test_uid_range {config['test_range'][0]},{config['test_range'][1]}"
        f" --reference_data_structure default"
        f" --dataset_path {datadir}/wifs2024dataset/wifs2024dataset/"
        f" --block_side {config['eval_block_side']}"
        f" --block_stride {config['eval_block_stride']}"
        f" --image_side {config['eval_image_side']}"

rule ref_eval_combination:
    # This rule combines the reference evaluation results from different references
    # For example it can summarize the results for several references of the same model for different captures
    input:
        refs=lambda wildcards: ref_results_by_mode[wildcards.blocks_dir]
    output:
        f"{wrkspdir}main_results/{{blocks_dir}}/refperf_combined.pkl"
    shell:
        f"python {srcdir}src/scripts/reference_eval_combination.py"
        f" --input_pickles {{input.refs}}"
        f" --output_dir {wrkspdir}main_results/{{wildcards.blocks_dir}}/"
        f" --baseline {config.get('baseline','none')}"
        f" --dedicated=1"
        f" --block_side {config['eval_block_side']}"
        f" --block_stride {config['eval_block_stride']}"
        f" --image_side {config['eval_image_side']}"

rule metric_comparison_plot_truncated:
    # This rule generates a plot comparing different metrics
    input:
        script=srcdir+"src/scripts/metric_comparison_plot.py",
        refs_trunc=lambda wildcards: [
            f"{wrkspdir}refperf_truncated/{wildcards.blocks_dir}/{bn}"
            for bn in basenames_by_mode[wildcards.blocks_dir]
        ]
    output:
        f"{wrkspdir}main_results_truncated/{{blocks_dir}}/auc_comparison_SSIM.png",
        f"{wrkspdir}main_results_truncated/{{blocks_dir}}/auc_comparison_PCC.png",
        f"{wrkspdir}main_results_truncated/{{blocks_dir}}/auc_comparison_summary.json",
    shell:
        f"python {{input.script}}"
        f" --input_pickles {{input.refs_trunc}}"
        f" --output_dir {wrkspdir}main_results_truncated/{{wildcards.blocks_dir}}/"
        f" --baseline {config.get('baseline','none')}"
        f" --dedicated=1"
        f" --captures_used {' '.join(captures_used)}"

rule metric_comparison_plot_mean_ref_vs_orig_truncated:
    # This rule generates a plot comparing mean reference vs original metrics (truncated version)
    input:
        script=srcdir+"src/scripts/metric_comparison_plot.py",
        refs_trunc=lambda wildcards: [
            f"{wrkspdir}refperf_truncated/{wildcards.blocks_dir}/{bn}"
            for bn in basenames_by_mode[wildcards.blocks_dir]
        ]
    output:
        f"{wrkspdir}main_results_truncated/{{blocks_dir}}/mean_ref_vs_orig_comparison_SSIM.png",
        f"{wrkspdir}main_results_truncated/{{blocks_dir}}/mean_ref_vs_orig_comparison_PCC.png",
        f"{wrkspdir}main_results_truncated/{{blocks_dir}}/mean_ref_vs_orig_comparison_summary.json",
    shell:
        f"python {{input.script}}"
        f" --input_pickles {{input.refs_trunc}}"
        f" --output_dir {wrkspdir}main_results_truncated/{{wildcards.blocks_dir}}/"
        f" --baseline {config.get('baseline','none')}"
        f" --dedicated=1"
        f" --metric_types mean_ref_vs_orig"
        f" --captures_used {' '.join(captures_used)}"

rule metric_comparison_plot_mean_ref_vs_orig:
    # This rule generates a plot comparing mean reference vs original metrics
    input:
        script=srcdir+"src/scripts/metric_comparison_plot.py",
        refs=lambda wildcards: ref_results_by_mode[wildcards.blocks_dir]
    output:
        f"{wrkspdir}main_results/{{blocks_dir}}/mean_ref_vs_orig_comparison_SSIM.png",
        f"{wrkspdir}main_results/{{blocks_dir}}/mean_ref_vs_orig_comparison_PCC.png",
        f"{wrkspdir}main_results/{{blocks_dir}}/mean_ref_vs_orig_comparison_MSE.png",
        f"{wrkspdir}main_results/{{blocks_dir}}/mean_ref_vs_orig_comparison_summary.json",
    shell:
        f"python {{input.script}}"
        f" --input_pickles {{input.refs}}"
        f" --output_dir {wrkspdir}main_results/{{wildcards.blocks_dir}}/"
        f" --baseline {config.get('baseline','none')}"
        f" --dedicated=0"
        f" --metric_types mean_ref_vs_orig"
        f" --captures_used {' '.join(captures_used)}"

rule metric_comparison_plot:
    # This rule generates a plot comparing different metrics
    input:
        script=srcdir+"src/scripts/metric_comparison_plot.py",
        refs=lambda wildcards: ref_results_by_mode[wildcards.blocks_dir]
    output:
        f"{wrkspdir}main_results/{{blocks_dir}}/auc_comparison_SSIM.png",
        f"{wrkspdir}main_results/{{blocks_dir}}/auc_comparison_PCC.png",
        f"{wrkspdir}main_results/{{blocks_dir}}/auc_comparison_summary.json"
    shell:
        f"python {{input.script}}"
        f" --input_pickles {{input.refs}}"
        f" --output_dir {wrkspdir}main_results/{{wildcards.blocks_dir}}/"
        f" --baseline {config.get('baseline','none')}"
        f" --dedicated=0"
        f" --captures_used {' '.join(captures_used)}"

rule truncate_single_metric:
    # Truncate one reference pickle into the per-mode truncated dir
    input:
        src=lambda wildcards: input_map_by_mode[wildcards.blocks_dir][wildcards.base]
    output:
        f"{wrkspdir}refperf_truncated/{{blocks_dir}}/{{base}}"
    shell:
        f"python {srcdir}src/scripts/dedicated_truncation.py --input_pkl {{input.src}} --output_pkl {{output}}"

rule collect_important_metrics:
    input:
        trained=f"{wrkspdir}{{model_name}}/training.DONE",
        script=f"{srcdir}src/scripts/collect_important_metrics_run.py"
    output:
        metrics=f"{wrkspdir}{{model_name}}/metrics.json"
    shell:
        f"python {srcdir}src/scripts/collect_important_metrics_run.py {wrkspdir}{{wildcards.model_name}}"

rule collect_metrics_summary:
    input:
        metrics=expand(f"{wrkspdir}" + "{model_name}/metrics.json", model_name=list(models.keys())),
        script=f"{srcdir}src/scripts/collect_metrics_summary.py"
    output:
        summary_plot=f"{wrkspdir}main_results/metric_summaries/min_val_loss_vs_model.png"
    shell:
        (
            "python {srcdir}src/scripts/collect_metrics_summary.py "
            "{input.metrics} --output {output.summary_plot}"
        )

rule metric_vs_coverage_plot:
    input:
        expand(f"{wrkspdir}main_results_truncated/{{bd}}/auc_comparison_SSIM.png", bd=blocks_dirs),
        expand(f"{wrkspdir}main_results_truncated/{{bd}}/auc_comparison_PCC.png", bd=blocks_dirs),
        expand(f"{wrkspdir}main_results_truncated/{{bd}}/mean_ref_vs_orig_comparison_SSIM.png", bd=blocks_dirs),
        expand(f"{wrkspdir}main_results_truncated/{{bd}}/mean_ref_vs_orig_comparison_PCC.png", bd=blocks_dirs),
        srcdir+"src/scripts/metric_vs_size_plot.py",
    output:
        f"{wrkspdir}main_results_truncated/final_plots/auc_SSIM_ALL_{{dataset_pair}}_vs_blocksize.png",
        f"{wrkspdir}main_results_truncated/final_plots/auc_PCC_ALL_{{dataset_pair}}_vs_blocksize.png"
    shell:
        f"python {srcdir}src/scripts/metric_vs_size_plot.py"
        f" --root_dir={wrkspdir}main_results_truncated"
        f" --metric_type=auc"
        f" --metric=SSIM"
        f" --reference=all"
        f" --dataset_pair={{wildcards.dataset_pair}}"
        f" --output_dir={wrkspdir}main_results_truncated/final_plots"
        f" --block_size={config['eval_block_side']}"
        f" --block_stride={config['eval_block_stride']}"
        f" --auc_log1m"
        f" --truey"
        f" --do_legend_outside_plot={int(config.get('do_legend_outside_plot', False))}"
        f" --do_combine_transport={int(config.get('do_combine_transport', False))}"
        f" --do_combine_other_camera={int(config.get('do_combine_other_camera', False))}"
        f" \n "
        f"python {srcdir}src/scripts/metric_vs_size_plot.py"
        f" --root_dir={wrkspdir}main_results_truncated"
        f" --metric_type=auc"
        f" --metric=PCC"
        f" --reference=all"
        f" --dataset_pair={{wildcards.dataset_pair}}"
        f" --output_dir={wrkspdir}main_results_truncated/final_plots"
        f" --block_size={config['eval_block_side']}"
        f" --block_stride={config['eval_block_stride']}"
        f" --auc_log1m"
        f" --truey"
        f" --do_legend_outside_plot={int(config.get('do_legend_outside_plot', False))}"
        f" --do_combine_transport={int(config.get('do_combine_transport', False))}"
        f" --do_combine_other_camera={int(config.get('do_combine_other_camera', False))}"
        f" \n "
        f"python {srcdir}src/scripts/metric_vs_size_plot.py"
        f" --root_dir={wrkspdir}main_results_truncated"
        f" --metric_type=auc"
        f" --metric=MSE"
        f" --reference=all"
        f" --dataset_pair={{wildcards.dataset_pair}}"
        f" --output_dir={wrkspdir}main_results_truncated/final_plots"
        f" --block_size={config['eval_block_side']}"
        f" --block_stride={config['eval_block_stride']}"
        f" --auc_log1m"
        f" --truey"
        f" --do_legend_outside_plot={int(config.get('do_legend_outside_plot', False))}"
        f" --do_combine_transport={int(config.get('do_combine_transport', False))}"
        f" --do_combine_other_camera={int(config.get('do_combine_other_camera', False))}"


metric_vs_coverage_all_plots_list=[]
for capture in captures_used:
    metric_vs_coverage_all_plots_list.append(
        f"{wrkspdir}main_results_truncated/final_plots/auc_SSIM_ALL_{capture}_vs_f{capture[1:]}_vs_blocksize.png"
    )
    metric_vs_coverage_all_plots_list.append(
        f"{wrkspdir}main_results_truncated/final_plots/auc_PCC_ALL_{capture}_vs_f{capture[1:]}_vs_blocksize.png"
    )

rule metric_vs_coverage_all_plots:
    input:
        *metric_vs_coverage_all_plots_list
    output:
        f"{wrkspdir}main_results_truncated/final_plots/metric_vs_coverage_all_plots.DONE"
    shell:
        "touch {output}"
