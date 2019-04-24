"""Utilities for logging."""
import logging
import time

def timeit(method):

    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        logger = logging.getLogger(method.__name__)
        logger.debug('{} {} sec'.format(method.__name__, te-ts))
        return result

    return timed
