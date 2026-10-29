import os

def build_lightgbm(params=None):
    from lightgbm import LGBMClassifier
    return LGBMClassifier(**{"random_state": 42, "verbosity": -1, **(params or {})})


def build_tabpfn(params=None,apikey=None):
    import tabpfn_client
    from tabpfn_client import TabPFNClassifier
    apikey = apikey or os.getenv("TABPFN_API_KEY")
    if apikey:
        tabpfn_client.set_access_token(apikey)
    return TabPFNClassifier(**(params or {}))
