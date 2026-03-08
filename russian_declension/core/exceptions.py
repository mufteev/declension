"""
Пользовательские исключения системы склонения.

Иерархия исключений позволяет API-слою корректно маппить ошибки на HTTP-коды:
  - DeclensionError (400) — ошибка в запросе или невозможность склонения
  - EngineError (503)     — сбой конкретного engine
  - WordNotFoundError     — слово не найдено ни в одном engine
"""


class DeclensionError(Exception):
    """Базовое исключение системы склонения."""
    pass


class WordNotFoundError(DeclensionError):
    """Слово не найдено ни одним engine в цепочке."""
    pass


class EngineError(DeclensionError):
    """Ошибка в конкретном engine (pymorphy не инициализирован, ONNX упал и т.д.)."""

    def __init__(self, engine_name: str, message: str):
        self.engine_name = engine_name
        super().__init__(f"[{engine_name}] {message}")


class InvalidWordError(DeclensionError):
    """Слово не является существительным или не может быть склонено."""
    pass
