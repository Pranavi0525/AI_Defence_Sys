"""
api/settings.py
=================
Environment-driven configuration for productionization (Phase 4).

Deliberately separate from config.py: config.py is about WHERE files live
on disk (report paths, stage catalog) and never changes between dev and
prod. This module is about HOW THE SERVER BEHAVES per environment (CORS,
logging verbosity, which host/port to bind) and is exactly what needs to
differ between a laptop and a Render deployment.

Reads from process environment variables, with defaults that reproduce
today's local-dev behavior exactly -- setting zero env vars must not
change anything for someone running this on their laptop.

    ENVIRONMENT              "development" | "staging" | "production"
                             default: "development"
    CORS_ALLOWED_ORIGINS     comma-separated list of allowed origins.
                             default in development: "*" (matches the
                             current wide-open behavior, documented in
                             app.py as a known local-hackathon shortcut).
                             In staging/production, "*" is REJECTED at
                             startup (fail loudly, not silently insecure)
                             unless CORS_ALLOW_ALL_ORIGINS_UNSAFE=true is
                             also set explicitly.
    API_HOST / API_PORT      default 127.0.0.1:8000 (run_api.py's own
                             argparse defaults still take precedence when
                             passed explicitly on the command line --
                             these env vars are the fallback for
                             deployment platforms like Render that set
                             $PORT and expect the process to honor it).
    MODEL_VERSION            optional human label for the deployed model
                             (e.g. a release tag). Falls back to the
                             git-commit-based version computed from the
                             saved artifacts if unset -- see
                             inference.py's ModelRegistry.
    LOG_LEVEL                default "INFO".
"""
from __future__ import annotations

import os


class ConfigError(RuntimeError):
    """Raised when the process environment is unsafe/invalid to start with."""


def _split_origins(raw: str) -> list[str]:
    return [o.strip() for o in raw.split(",") if o.strip()]


class Settings:
    def __init__(self) -> None:
        self.environment: str = os.environ.get("ENVIRONMENT", "development").strip().lower()
        if self.environment not in {"development", "staging", "production"}:
            raise ConfigError(
                f"ENVIRONMENT={self.environment!r} is not one of "
                f"'development'/'staging'/'production'."
            )

        raw_origins = os.environ.get("CORS_ALLOWED_ORIGINS")
        allow_all_unsafe = os.environ.get("CORS_ALLOW_ALL_ORIGINS_UNSAFE", "").lower() == "true"

        if raw_origins is None:
            if self.environment == "development":
                self.cors_allowed_origins = ["*"]
            else:
                raise ConfigError(
                    f"CORS_ALLOWED_ORIGINS must be set explicitly when "
                    f"ENVIRONMENT={self.environment!r} (comma-separated list "
                    f"of allowed frontend origins, e.g. "
                    f"'https://your-frontend.vercel.app'). Refusing to start "
                    f"with a wide-open CORS policy outside development."
                )
        elif raw_origins.strip() == "*":
            if self.environment != "development" and not allow_all_unsafe:
                raise ConfigError(
                    "CORS_ALLOWED_ORIGINS='*' is not allowed outside "
                    "development. Set explicit origins, or set "
                    "CORS_ALLOW_ALL_ORIGINS_UNSAFE=true if you genuinely "
                    "intend to allow every origin in this environment."
                )
            self.cors_allowed_origins = ["*"]
        else:
            self.cors_allowed_origins = _split_origins(raw_origins)

        self.host: str = os.environ.get("API_HOST", "127.0.0.1")
        # Render (and most PaaS) inject $PORT and expect the process to bind
        # to it; fall back to 8000 for local dev.
        self.port: int = int(os.environ.get("PORT", os.environ.get("API_PORT", "8000")))
        self.model_version_override: str | None = os.environ.get("MODEL_VERSION") or None
        self.log_level: str = os.environ.get("LOG_LEVEL", "INFO").upper()

    def as_dict(self) -> dict:
        return {
            "environment": self.environment,
            "cors_allowed_origins": self.cors_allowed_origins,
            "host": self.host,
            "port": self.port,
            "model_version_override": self.model_version_override,
            "log_level": self.log_level,
        }


def load_settings() -> Settings:
    return Settings()
