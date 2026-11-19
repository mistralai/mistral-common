r"""Neutral, domain-local support for chat-template tests.

Both the unit lane (`tests/unit/integrations/chat_templates/`) and the
integration lane (`tests/integration/chat_templates/`) import from this
package. The legacy plural-root modules under `tests/integrations/` re-export
from here for their remaining parity and Transformers consumers during the
overlap. Support modules never import test modules.
"""
