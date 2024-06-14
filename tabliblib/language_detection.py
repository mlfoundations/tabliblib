import os
import warnings
from typing import Dict, Union

import fasttext

models = {"low_mem": None, "high_mem": None}

cache_dir = os.path.dirname(__file__)


def get_or_load_model(low_memory=False):
    if low_memory:
        model = models.get("low_mem", None)
        if not model:
            model_path = os.path.join(cache_dir, "lid.176.ftz")
            model = fasttext.load_model(model_path)
            models["low_mem"] = model
        return model
    else:
        model = models.get("high_mem", None)
        if not model:
            model_path = os.path.join(cache_dir, "lid.176.bin")
            # suppress Warning : `load_model` does not return WordVectorModel or
            # SupervisedModel any more, but a `FastText` object which is very similar.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                model = fasttext.load_model(model_path)
            models["high_mem"] = model
        return model


def detect(text: str, low_memory=False) -> Dict[str, Union[str, float]]:
    model = get_or_load_model(low_memory)
    labels, scores = model.predict(text)
    label = labels[0].replace("__label__", '')
    score = min(float(scores[0]), 1.0)
    return {
        "lang": label,
        "score": score,
    }
