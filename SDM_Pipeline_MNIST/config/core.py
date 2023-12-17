from pathlib import Path
from typing import Dict
from pydantic import BaseModel
from strictyaml import YAML, load
from yaml.loader import FullLoader
import yaml
import SDM_Pipeline_MNIST

# Project Directories
PACKAGE_ROOT = Path(SDM_Pipeline_MNIST.__file__).resolve().parent
ROOT = PACKAGE_ROOT.parent
CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yml"


class AppConfig(BaseModel):
    package_name: str
    save_file: str


class SDMPipelineMNIST(BaseModel):
    UnetSP: Dict
    UnetTR: Dict


class SDMTransformer(BaseModel):
    UnetTR: Dict


class AutoEncoder(BaseModel):
    AE: Dict


class Config(BaseModel):
    """Master config object."""

    model_UnetSP: SDMPipelineMNIST
    model_UnetTR: SDMTransformer
    model_AE: AutoEncoder
    app_config: AppConfig


def find_config_file() -> Path:
    """Locate the configuration file."""
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    raise Exception(f"Config not found at {CONFIG_FILE_PATH!r}")


def fetch_config_from_yaml(cfg_path: Path = None) -> YAML:
    """Parse YAML containing the package configuration."""

    if not cfg_path:
        cfg_path = find_config_file()

    with open(cfg_path, "r") as stream:
        try:
            # Converts yaml document to python object
            parsed_config = yaml.load(stream, Loader=FullLoader)
            return parsed_config
        except yaml.YAMLError as e:
            print(e)


def create_and_validate_config(parsed_config: YAML = None) -> Config:
    """Run validation on config values."""
    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()
        for k, v in parsed_config.items():
            if k == "UnetSP":
                parsed_config[k]["batch_size"] = int(parsed_config[k]["batch_size"])
                parsed_config[k]["num_epochs"] = int(parsed_config[k]["num_epochs"])
                parsed_config[k]["sigma"] = float(parsed_config[k]["sigma"])
                parsed_config[k]["euler_maruyam_num_steps"] = int(
                    parsed_config[k]["euler_maruyam_num_steps"]
                )
                parsed_config[k]["eps_stab"] = float(parsed_config[k]["eps_stab"])
                parsed_config[k]["lr"] = float(parsed_config[k]["lr"])
                parsed_config[k]["use_unet_score_based"] = bool(
                    parsed_config[k]["use_unet_score_based"]
                )
            elif k == "UnetTR":
                parsed_config[k]["batch_size"] = int(parsed_config[k]["batch_size"])
                parsed_config[k]["num_epochs"] = int(parsed_config[k]["num_epochs"])
                parsed_config[k]["sigma"] = float(parsed_config[k]["sigma"])
                parsed_config[k]["euler_maruyam_num_steps"] = int(
                    parsed_config[k]["euler_maruyam_num_steps"]
                )
                parsed_config[k]["eps_stab"] = float(parsed_config[k]["eps_stab"])
                parsed_config[k]["lr"] = float(parsed_config[k]["lr"])
                parsed_config[k]["use_latent_unet"] = bool(parsed_config[k]["use_latent_unet"])
                parsed_config[k]["autoencoder_model"] = str(parsed_config[k]["autoencoder_model"])
            elif k == "AE":
                parsed_config[k]["batch_size"] = int(parsed_config[k]["batch_size"])
                parsed_config[k]["num_epochs"] = int(parsed_config[k]["num_epochs"])
                parsed_config[k]["lr"] = float(parsed_config[k]["lr"])
            else:
                Exception("No configuration in config file.")

    _config = Config(
        app_config=AppConfig(**parsed_config),
        model_UnetSP=SDMPipelineMNIST(**parsed_config),
        model_UnetTR=SDMTransformer(**parsed_config),
        model_AE=AutoEncoder(**parsed_config),
    )

    return _config


config = create_and_validate_config()
