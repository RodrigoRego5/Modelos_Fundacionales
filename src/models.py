from lightgbm import LGBMClassifier

import tabpfn_client
from tabpfn_client import TabPFNClassifier

def build_lightgbm(params=None):
    return LGBMClassifier(**(params or {}))


def build_tabpfn(params=None,apikey=None):
    tabpfn_client.set_access_token(apikey)
    return TabPFNClassifier(**(params or {}))
