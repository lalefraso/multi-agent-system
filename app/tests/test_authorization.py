import hashlib
import unittest
from unittest.mock import ANY, patch

from redis.exceptions import RedisError

from app.authorization import (
    CUSTOMER_MEMBERSHIP_PREFIX,
    CUSTOMER_OWNER_PREFIX,
    CustomerAlreadyAssignedError,
    CustomerAuthorizationError,
    CustomerMembershipNotFoundError,
    CustomerNotAuthorizedError,
    CustomerSelectionRequiredError,
    _membership_key,
    _customer_owner_key,
    _principal_hash,
    authorize_customer_access,
    grant_customer_membership,
    revoke_customer_membership,
)
from app.manage_authorization import _principal_id


class CustomerAuthorizationTests(unittest.TestCase):
    def test_membership_key_uses_a_hash_of_the_oidc_principal(self):
        principal_id = "https://identity.example/\0subject-123"
        expected_hash = hashlib.sha256(
            principal_id.encode("utf-8")
        ).hexdigest()

        self.assertEqual(
            _membership_key(principal_id),
            f"{CUSTOMER_MEMBERSHIP_PREFIX}{expected_hash}:customers",
        )
        self.assertNotIn(principal_id, _membership_key(principal_id))

    def test_owner_key_uses_a_hash_of_the_customer_id(self):
        customer_id = "private-customer-id"
        customer_hash = hashlib.sha256(customer_id.encode()).hexdigest()

        self.assertEqual(
            _customer_owner_key(customer_id),
            f"{CUSTOMER_OWNER_PREFIX}{customer_hash}:owner",
        )
        self.assertNotIn(customer_id, _customer_owner_key(customer_id))

    def test_management_principal_uses_issuer_and_subject(self):
        self.assertEqual(
            _principal_id("https://identity.example/", "subject-123"),
            "https://identity.example/\0subject-123",
        )
        with self.assertRaises(ValueError):
            _principal_id("http://identity.example/", "subject-123")

    @patch("app.authorization.Redis.from_url")
    @patch("app.authorization._resolve_redis_url", return_value="redis://test")
    def test_selects_the_only_authorized_customer_implicitly(
        self, _resolve_url, redis_from_url
    ):
        client = redis_from_url.return_value
        client.smembers.return_value = {"customer-1"}
        client.get.return_value = _principal_hash("principal-1")

        result = authorize_customer_access("principal-1", None)

        self.assertEqual(result, "customer-1")
        client.smembers.assert_called_once_with(_membership_key("principal-1"))
        client.close.assert_called_once_with()

    @patch("app.authorization.Redis.from_url")
    @patch("app.authorization._resolve_redis_url", return_value="redis://test")
    def test_accepts_an_explicit_member_of_the_principal(
        self, _resolve_url, redis_from_url
    ):
        redis_from_url.return_value.smembers.return_value = {
            "customer-1",
            "customer-2",
        }
        redis_from_url.return_value.get.return_value = _principal_hash(
            "principal-1"
        )

        result = authorize_customer_access("principal-1", "customer-2")

        self.assertEqual(result, "customer-2")

    @patch("app.authorization.Redis.from_url")
    @patch("app.authorization._resolve_redis_url", return_value="redis://test")
    def test_requires_selection_when_principal_has_multiple_customers(
        self, _resolve_url, redis_from_url
    ):
        redis_from_url.return_value.smembers.return_value = {
            "customer-1",
            "customer-2",
        }

        with self.assertRaises(CustomerSelectionRequiredError):
            authorize_customer_access("principal-1", None)

    @patch("app.authorization.Redis.from_url")
    @patch("app.authorization._resolve_redis_url", return_value="redis://test")
    def test_rejects_customer_not_in_membership_set(
        self, _resolve_url, redis_from_url
    ):
        redis_from_url.return_value.smembers.return_value = {"customer-1"}

        with self.assertRaises(CustomerNotAuthorizedError):
            authorize_customer_access("principal-1", "customer-2")

    @patch("app.authorization.Redis.from_url")
    @patch("app.authorization._resolve_redis_url", return_value="redis://test")
    def test_denies_membership_if_reverse_owner_is_another_principal(
        self, _resolve_url, redis_from_url
    ):
        client = redis_from_url.return_value
        client.smembers.return_value = {"customer-1"}
        client.get.return_value = _principal_hash("principal-2")

        with self.assertRaises(CustomerNotAuthorizedError):
            authorize_customer_access("principal-1", "customer-1")

        client.close.assert_called_once_with()

    @patch("app.authorization.Redis.from_url")
    @patch("app.authorization._resolve_redis_url", return_value="redis://test")
    def test_rejects_principal_without_customer_memberships(
        self, _resolve_url, redis_from_url
    ):
        redis_from_url.return_value.smembers.return_value = set()

        with self.assertRaises(CustomerNotAuthorizedError):
            authorize_customer_access("principal-1", None)

    @patch("app.authorization.Redis.from_url")
    @patch("app.authorization._resolve_redis_url", return_value="redis://test")
    def test_fails_closed_for_invalid_membership_data(
        self, _resolve_url, redis_from_url
    ):
        redis_from_url.return_value.smembers.return_value = {""}

        with self.assertRaisesRegex(
            CustomerAuthorizationError, "Invalid customer authorization"
        ):
            authorize_customer_access("principal-1", None)

    @patch("app.authorization.Redis.from_url")
    @patch("app.authorization._resolve_redis_url", return_value="redis://test")
    def test_reports_redis_failures(self, _resolve_url, redis_from_url):
        redis_from_url.return_value.smembers.side_effect = RedisError(
            "Redis unavailable"
        )

        with self.assertRaisesRegex(
            CustomerAuthorizationError, "Could not retrieve"
        ):
            authorize_customer_access("principal-1", None)

        redis_from_url.return_value.close.assert_called_once_with()

    @patch("app.authorization.Redis.from_url")
    @patch("app.authorization._resolve_redis_url", return_value="redis://test")
    def test_grants_membership_atomically_and_idempotently(
        self, _resolve_url, redis_from_url
    ):
        client = redis_from_url.return_value
        client.eval.return_value = 1

        grant_customer_membership("principal-1", "customer-1")
        grant_customer_membership("principal-1", "customer-1")

        self.assertEqual(client.eval.call_count, 2)
        client.eval.assert_called_with(
            ANY,
            2,
            _membership_key("principal-1"),
            _customer_owner_key("customer-1"),
            _principal_hash("principal-1"),
            "customer-1",
        )
        client.close.assert_called_with()

    @patch("app.authorization.Redis.from_url")
    @patch("app.authorization._resolve_redis_url", return_value="redis://test")
    def test_rejects_grant_if_customer_belongs_to_another_principal(
        self, _resolve_url, redis_from_url
    ):
        redis_from_url.return_value.eval.return_value = 0

        with self.assertRaises(CustomerAlreadyAssignedError):
            grant_customer_membership("principal-2", "customer-1")
        self.assertIn(
            "SREM", redis_from_url.return_value.eval.call_args.args[0]
        )

    @patch("app.authorization.Redis.from_url")
    @patch("app.authorization._resolve_redis_url", return_value="redis://test")
    def test_revokes_membership_atomically(self, _resolve_url, redis_from_url):
        redis_from_url.return_value.eval.return_value = 1

        revoke_customer_membership("principal-1", "customer-1")

        self.assertIn(
            "SREM", redis_from_url.return_value.eval.call_args.args[0]
        )
        self.assertIn(
            "DEL", redis_from_url.return_value.eval.call_args.args[0]
        )

    @patch("app.authorization.Redis.from_url")
    @patch("app.authorization._resolve_redis_url", return_value="redis://test")
    def test_rejects_revocation_by_a_different_principal(
        self, _resolve_url, redis_from_url
    ):
        redis_from_url.return_value.eval.return_value = 0

        with self.assertRaises(CustomerMembershipNotFoundError):
            revoke_customer_membership("principal-2", "customer-1")


if __name__ == "__main__":
    unittest.main()
