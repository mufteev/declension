"""
Система склонения русских слов, словосочетаний и фраз — v2 (GPU).

Фаза 1: Словоуровневый конвейер (pymorphy3 → fallback)
Фаза 2: Именованные сущности (ФИО, организации, числительные)
Фаза 3: Фразовый движок (dependency parsing + agreement propagation)
GPU:    ruT5, BERT validator, animacy classifier, meta-ensemble
"""
__version__ = "0.4.0"
