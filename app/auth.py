from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from urllib.parse import urlsplit

import jwt
from dotenv import load_dotenv
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt import PyJWKClient
from jwt.exceptions import PyJWKClientConnectionError, PyJWKClientError


ENV_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
load_dotenv(ENV_FILE, override=False)

bearer_scheme = HTTPBearer(auto_error=False)


class AuthenticationConfigurationError(RuntimeError):
    """Raised when OIDC authentication is not configured correctly."""


@dataclass(frozen=True)
class AuthenticatedPrincipal:
    principal_id: str


@dataclass(frozen=True)
class OIDCSettings:
    issuer: str
    audience: str
    jwks_url: str


def _require_https_url(name: str, value: str) -> str:
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.fragment
    ):
        raise AuthenticationConfigurationError(
            f"{name} must be an absolute HTTPS URL"
        )
    return value


def _oidc_settings() -> OIDCSettings:
    issuer = os.getenv("OIDC_ISSUER", "").strip()
    audience = os.getenv("OIDC_AUDIENCE", "").strip()
    jwks_url = os.getenv("OIDC_JWKS_URL", "").strip()
    if not issuer or not audience or not jwks_url:
        raise AuthenticationConfigurationError(
            "Set OIDC_ISSUER, OIDC_AUDIENCE, and OIDC_JWKS_URL"
        )
    return OIDCSettings(
        issuer=_require_https_url("OIDC_ISSUER", issuer),
        audience=audience,
        jwks_url=_require_https_url("OIDC_JWKS_URL", jwks_url),
    )


@lru_cache(maxsize=4)
def _jwks_client(jwks_url: str) -> PyJWKClient:
    return PyJWKClient(jwks_url, timeout=5)


def require_authenticated_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> AuthenticatedPrincipal:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Bearer token required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        settings = _oidc_settings()
    except AuthenticationConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OIDC authentication is not configured",
        ) from exc

    try:
        signing_key = _jwks_client(
            settings.jwks_url
        ).get_signing_key_from_jwt(credentials.credentials)
    except PyJWKClientConnectionError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Could not retrieve OIDC signing keys",
        ) from exc
    except PyJWKClientError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    try:
        claims = jwt.decode(
            credentials.credentials,
            signing_key.key,
            algorithms=["RS256"],
            issuer=settings.issuer,
            audience=settings.audience,
            options={"require": ["exp", "iss", "aud", "sub"]},
        )
    except jwt.InvalidTokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    subject = claims.get("sub")
    if not isinstance(subject, str) or not subject.strip() or len(subject) > 512:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="The authenticated identity has no valid subject",
        )
    return AuthenticatedPrincipal(
        principal_id=f"{settings.issuer}\0{subject}"
    )
