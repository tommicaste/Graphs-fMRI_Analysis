import argparse, yaml
from pathlib import Path
from static_models.pipeline import run_from_config

cli = argparse.ArgumentParser(description="Run one experiment from a YAML config")
cli.add_argument("-c", "--config", default="config.yaml", help="Path to YAML configuration file")
args = cli.parse_args()

cfg = yaml.safe_load(Path(args.config).read_text())
run_from_config(cfg)
