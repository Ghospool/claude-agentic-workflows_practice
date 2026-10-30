import logging

logger = logging.getLogger(__name__)


def add_item(item, bucket=[]):
    bucket.append(item)
    return bucket


def parse(value):
    try:
        return int(value)
    except:
        return 0


def is_missing(x):
    return x == None
