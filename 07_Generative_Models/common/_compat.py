"""跨家族复用 02_CNN_Family 的 MNIST 加载（sys.path 注入）。"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from importlib import import_module
    cnn_data = import_module("02_CNN_Family.common.data")
except Exception as _e:  # pragma: no cover
    raise ImportError(f"无法加载 02_CNN_Family.common.data: {_e}；请确认 02 家族缓存存在") from _e
