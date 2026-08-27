import omegaconf

def load_config(config_path: str = "./desktop_env/env_config.yaml"):
    return omegaconf.OmegaConf.load(config_path)
