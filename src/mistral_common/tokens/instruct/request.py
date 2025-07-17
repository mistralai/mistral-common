# this file is deprecated
import warnings

from mistral_common.protocol.fim.request import FIMRequest  # noqa: F401
from mistral_common.protocol.instruct.request import InstructRequest  # noqa: F401

warnings.warn(
    "This file is deprecated and will be deleted in v1.9.0, please import as follows instead: \n"
    "`from mistral_common.protocol.fim.request import FimRequest` \n or \n"
    "`from mistral_common.protocol.instruct.request import InstructRequest`",
    FutureWarning,
)
