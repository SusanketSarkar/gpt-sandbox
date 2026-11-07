"""gpt-sandbox: a small, hackable lab for autoregressive GPT experiments.

Plugin authors need only::

    from gpt_sandbox import register, Ctx, sdpa_causal, layer_component
"""
__version__ = "0.1.0"

from .registry import build, get, names, register  # noqa: F401
from .model.context import Ctx  # noqa: F401
from .model.spec import layer_selected, select_spec  # noqa: F401
from .model.build import layer_component, layer_norm  # noqa: F401
from .components.attention import repeat_kv, sdpa_causal  # noqa: F401
from .components.block import BlockBase  # noqa: F401
from .components.norm import LayerNorm, RMSNorm  # noqa: F401
from .components.posenc import PosEncBase  # noqa: F401
