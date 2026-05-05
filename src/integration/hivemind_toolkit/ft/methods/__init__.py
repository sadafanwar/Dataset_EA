from .qlora import QLoRATrainer


class LoRATrainer:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("LoRATrainer is not included in this local setup. Use method='qlora'.")


class DoRATrainer:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("DoRATrainer is not included in this local setup. Use method='qlora'.")