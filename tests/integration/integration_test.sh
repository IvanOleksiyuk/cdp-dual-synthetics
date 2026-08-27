python -m snakemake -s workflow/main.smk --cores="all" \
--configfile workflow/config/main_integration_test.yaml \
--rerun-incomplete -p \
--until train_model