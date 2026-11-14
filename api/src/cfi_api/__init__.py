"""The Customer Fit Intelligence API: one Flask service under /api/v1.

create_app() builds it (T201). flask-smorest documents every endpoint as
OpenAPI at /api/v1/openapi.json, browsable at /api/v1/docs. Run it locally
with: flask --app cfi_api run --port 8000
"""

from cfi_api.app import create_app

__version__ = "0.1.0"

__all__ = ["create_app"]
