import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import jwt
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import HTTPException
from fastapi.testclient import TestClient
from fastapi.security import HTTPAuthorizationCredentials
from jwt.exceptions import PyJWKClientConnectionError

from app.auth import AuthenticatedUser, require_authenticated_user
from app.main import app


class AuthenticationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )

    def setUp(self):
        self.environment = patch.dict(
            "os.environ",
            {
                "OIDC_ISSUER": "https://identity.example/",
                "OIDC_AUDIENCE": "getnet-api",
                "OIDC_JWKS_URL": "https://identity.example/.well-known/jwks.json",
                "OIDC_USER_ID_CLAIM": "customer_id",
            },
            clear=True,
        )
        self.environment.start()
        self.addCleanup(self.environment.stop)

        self.jwks_client = patch("app.auth._jwks_client")
        self.mock_jwks_client = self.jwks_client.start()
        self.addCleanup(self.jwks_client.stop)
        self.mock_jwks_client.return_value.get_signing_key_from_jwt.return_value = (
            SimpleNamespace(key=self.private_key.public_key())
        )

    def _token(self, **overrides):
        claims = {
            "iss": "https://identity.example/",
            "aud": "getnet-api",
            "sub": "identity-subject-123",
            "customer_id": "customer-123",
            "exp": int(time.time()) + 300,
        }
        claims.update(overrides)
        return jwt.encode(claims, self.private_key, algorithm="RS256")

    def _credentials(self, token):
        return HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)

    def test_derives_customer_identity_from_verified_claim(self):
        user = require_authenticated_user(self._credentials(self._token()))

        self.assertEqual(user, AuthenticatedUser(user_id="customer-123"))
        self.mock_jwks_client.assert_called_once_with(
            "https://identity.example/.well-known/jwks.json"
        )

    def test_rejects_token_with_wrong_issuer(self):
        with self.assertRaises(HTTPException) as raised:
            require_authenticated_user(
                self._credentials(self._token(iss="https://attacker.example/"))
            )

        self.assertEqual(raised.exception.status_code, 401)

    def test_rejects_token_with_wrong_audience(self):
        with self.assertRaises(HTTPException) as raised:
            require_authenticated_user(
                self._credentials(self._token(aud="another-api"))
            )

        self.assertEqual(raised.exception.status_code, 401)

    def test_rejects_invalid_signature(self):
        other_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )
        token = jwt.encode(
            {
                "iss": "https://identity.example/",
                "aud": "getnet-api",
                "sub": "identity-subject-123",
                "customer_id": "customer-123",
                "exp": int(time.time()) + 300,
            },
            other_key,
            algorithm="RS256",
        )

        with self.assertRaises(HTTPException) as raised:
            require_authenticated_user(self._credentials(token))

        self.assertEqual(raised.exception.status_code, 401)

    def test_rejects_expired_token(self):
        with self.assertRaises(HTTPException) as raised:
            require_authenticated_user(
                self._credentials(self._token(exp=int(time.time()) - 1))
            )

        self.assertEqual(raised.exception.status_code, 401)

    def test_rejects_identity_without_customer_mapping(self):
        with self.assertRaises(HTTPException) as raised:
            require_authenticated_user(
                self._credentials(self._token(customer_id=""))
            )

        self.assertEqual(raised.exception.status_code, 403)

    def test_returns_unavailable_when_signing_key_service_is_down(self):
        self.mock_jwks_client.return_value.get_signing_key_from_jwt.side_effect = (
            PyJWKClientConnectionError("JWKS unavailable")
        )

        with self.assertRaises(HTTPException) as raised:
            require_authenticated_user(self._credentials(self._token()))

        self.assertEqual(raised.exception.status_code, 503)

    def test_fails_closed_when_oidc_is_not_configured(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(HTTPException) as raised:
                require_authenticated_user(self._credentials(self._token()))

        self.assertEqual(raised.exception.status_code, 503)

    def test_chat_endpoint_rejects_missing_bearer_token(self):
        response = TestClient(app).post(
            "/chat",
            json={"message": "Show my account details"},
        )

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.headers["www-authenticate"], "Bearer")


if __name__ == "__main__":
    unittest.main()
