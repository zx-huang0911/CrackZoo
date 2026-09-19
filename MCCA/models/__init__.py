from functools import partial

from .mcca import MCCA

MODEL_REGISTRY = {
    "mcca": partial(MCCA, model_variant="full_mcca"),
    "full_mcca": partial(MCCA, model_variant="full_mcca"),
    "baseline_plain": partial(MCCA, model_variant="baseline_plain"),
    "baseline_deepsup": partial(MCCA, model_variant="baseline_deepsup"),
    "mcca_no_mscfm": partial(MCCA, model_variant="mcca_no_mscfm"),
    "mcca_no_cam": partial(MCCA, model_variant="mcca_no_cam"),
}


def create_model(model_name: str, **kwargs):
    key = model_name.lower()
    if key not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(f"Unknown model '{model_name}'. Available: {available}")
    return MODEL_REGISTRY[key](**kwargs)


def available_models():
    return sorted(MODEL_REGISTRY.keys())
