from models.basic_model import Net
from models.siggraph import siggraph17

model_collection = {"basic" : Net, "siggraph" : siggraph17}

def build_model(cfg):
    return model_collection[cfg.MODEL.NAME](color_space=cfg.INPUT.COLOR_SPACE,download=cfg.MODEL.DOWNLOAD_PRETRAINED)
