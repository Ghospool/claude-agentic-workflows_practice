import pickle
import yaml


def load_state(blob: bytes):
    return pickle.loads(blob)


def load_settings(stream):
    return yaml.load(stream)


def refresh(cache):
    try:
        cache.refresh()
    except Exception:
        pass
