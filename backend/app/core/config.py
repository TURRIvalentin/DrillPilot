"""App-wide settings. No pydantic-settings dependency added for this -- the only
configurable value today is the model URI (useful for tests/local overrides via env
var), so a plain class reading os.environ is enough; adding a dependency for one
optional string would be over-engineering for what M7 needs.
"""

from __future__ import annotations

import os

from ml.inference.predict import MODEL_URI as _DEFAULT_MODEL_URI


class Settings:
    model_uri: str = os.environ.get("DRILLPILOT_MODEL_URI", _DEFAULT_MODEL_URI)
    log_level: str = os.environ.get("DRILLPILOT_LOG_LEVEL", "INFO")


settings = Settings()
