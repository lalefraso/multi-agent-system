import fnmatch
import hashlib
import json
import unittest
from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient
from redis.exceptions import RedisError, WatchError

from app.auth import AuthenticatedPrincipal, require_authenticated_user
from app.authorization import _principal_hash
from app.escalation import EscalationDraft, create_escalation
from app.main import app
from app.operator_authorization import OPERATOR_SET_KEY
from app.manage_operators import main as manage_operators
from app.schemas import EscalationReason
from app.tests.test_escalation import FakeRedis


class OperatorRedis(FakeRedis):
    def __init__(self):
        super().__init__()
        self.sets = {}
        self.versions = {}
        self.fail_reads = False
        self.conflict_once = False
        self.conflict_callback = None

    def get(self, key):
        if self.fail_reads:
            raise RedisError("unavailable")
        return super().get(key)

    def set(self, key, value, nx=False, ex=None):
        result = super().set(key, value, nx=nx, ex=ex)
        if result:
            self.versions[key] = self.versions.get(key, 0) + 1
        return result

    def scan_iter(self, match, count):
        return iter([key for key in self.data if fnmatch.fnmatch(key, match)])

    def sadd(self, key, value):
        self.sets.setdefault(key, set()).add(value)

    def srem(self, key, value):
        self.sets.setdefault(key, set()).discard(value)

    def sismember(self, key, value):
        if self.fail_reads:
            raise RedisError("unavailable")
        return value in self.sets.get(key, set())

    def pipeline(self):
        return FakePipeline(self)


class FakePipeline:
    def __init__(self, client):
        self.client = client
        self.key = None
        self.version = None
        self.pending = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def watch(self, key):
        self.key = key
        self.version = self.client.versions.get(key)

    def get(self, key):
        return self.client.get(key)

    def multi(self):
        pass

    def set(self, key, value):
        self.pending = (key, value)

    def execute(self):
        if self.client.conflict_once:
            self.client.conflict_once = False
            if self.client.conflict_callback:
                self.client.conflict_callback()
            raise WatchError()
        if self.version != self.client.versions.get(self.key):
            raise WatchError()
        self.client.set(*self.pending)


class OperatorAPITests(unittest.TestCase):
    def setUp(self):
        self.redis = OperatorRedis()
        self.principal = "https://issuer.example/\0operator-a"
        self.redis.sadd(OPERATOR_SET_KEY, _principal_hash(self.principal))
        self.auth = patch("app.operator_authorization._client", return_value=self.redis)
        self.auth.start()
        self.addCleanup(self.auth.stop)
        app.dependency_overrides[require_authenticated_user] = lambda: AuthenticatedPrincipal(
            principal_id=self.principal
        )
        from app.operator_api import escalation_client
        app.dependency_overrides[escalation_client] = lambda: self.redis
        self.addCleanup(app.dependency_overrides.clear)
        self.client = TestClient(app)
        self.record = create_escalation(
            self.redis, uuid4(), "private-owner", "customer-1",
            EscalationReason.HUMAN_REQUESTED,
            EscalationDraft(summary="Customer requested human help."),
        )
        self.path = f"/operator/escalations/{self.record.escalation_id}"

    def test_list_detail_and_owned_transitions(self):
        response = self.client.get("/operator/escalations?status=pending")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["items"][0]["escalation_id"], str(self.record.escalation_id))
        self.assertIsNone(response.json()["next_offset"])
        detail = self.client.get(self.path)
        self.assertEqual(detail.status_code, 200)
        self.assertEqual(detail.json()["summary"], "Customer requested human help.")
        self.assertEqual(detail.json()["customer_id"], "customer-1")
        self.assertIsNone(detail.json()["conversation"])
        self.assertIsNone(detail.json()["proposed_action"])
        self.assertIsNone(detail.json()["sources"])
        self.assertNotIn("owner_token", detail.json())
        self.assertEqual(self.client.patch(self.path + "/status", json={"status": "closed"}).status_code, 409)

        self.redis.conflict_once = True
        taken = self.client.patch(self.path + "/status", json={"status": "taken"})
        self.assertEqual(taken.status_code, 200)
        self.assertEqual(taken.json()["assigned_to"], _principal_hash(self.principal))
        self.assertIsNotNone(taken.json()["taken_at"])
        self.assertEqual(self.client.patch(self.path + "/status", json={"status": "taken"}).json(), taken.json())
        self.assertEqual(self.client.get("/operator/escalations").json()["items"], [])
        self.assertEqual(len(self.client.get("/operator/escalations?status=taken").json()["items"]), 1)

        self.principal = "https://issuer.example/\0operator-b"
        self.redis.sadd(OPERATOR_SET_KEY, _principal_hash(self.principal))
        self.assertEqual(self.client.patch(self.path + "/status", json={"status": "taken"}).status_code, 409)
        self.assertEqual(self.client.patch(self.path + "/status", json={"status": "closed"}).status_code, 409)
        self.principal = "https://issuer.example/\0operator-a"
        closed = self.client.patch(self.path + "/status", json={"status": "closed"})
        self.assertEqual(closed.status_code, 200)
        self.assertIsNotNone(closed.json()["closed_at"])
        self.assertEqual(self.client.patch(self.path + "/status", json={"status": "closed"}).json(), closed.json())
        self.assertEqual(self.client.patch(self.path + "/status", json={"status": "taken"}).status_code, 409)

    def test_pagination_and_old_records(self):
        key = next(iter(self.redis.data))
        old_record = json.loads(self.redis.data[key])
        for field in ("taken_by", "taken_at", "closed_at"):
            del old_record[field]
        self.redis.set(key, json.dumps(old_record))
        create_escalation(
            self.redis, uuid4(), "owner-2", "customer-2",
            EscalationReason.UNRESOLVED, EscalationDraft(summary="Terminal offline."),
        )
        first = self.client.get("/operator/escalations?limit=1").json()
        second = self.client.get(f"/operator/escalations?limit=1&offset={first['next_offset']}").json()
        self.assertEqual(len(first["items"]), 1)
        self.assertEqual(len(second["items"]), 1)
        self.assertNotEqual(first["items"][0]["escalation_id"], second["items"][0]["escalation_id"])
        self.assertIsNone(second["next_offset"])

    def test_competing_operator_wins_watch_race(self):
        other_hash = _principal_hash("https://issuer.example/\0operator-b")
        key = next(iter(self.redis.data))

        def claim_by_other():
            record = json.loads(self.redis.get(key))
            record["status"] = "taken"
            record["taken_by"] = other_hash
            self.redis.set(key, json.dumps(record))

        self.redis.conflict_once = True
        self.redis.conflict_callback = claim_by_other
        response = self.client.patch(self.path + "/status", json={"status": "taken"})
        self.assertEqual(response.status_code, 409)
        self.assertEqual(json.loads(self.redis.get(key))["taken_by"], other_hash)

    def test_closed_case_still_blocks_automated_chat(self):
        customer_principal = "https://issuer.example/\0customer"
        key = next(iter(self.redis.data))
        record = json.loads(self.redis.get(key))
        record["owner_token"] = hashlib.sha256(
            f"{customer_principal}\0customer-1".encode()
        ).hexdigest()
        self.redis.set(key, json.dumps(record))
        self.assertEqual(self.client.patch(self.path + "/status", json={"status": "taken"}).status_code, 200)
        self.assertEqual(self.client.patch(self.path + "/status", json={"status": "closed"}).status_code, 200)
        app.dependency_overrides[require_authenticated_user] = lambda: AuthenticatedPrincipal(
            principal_id=customer_principal
        )
        with (
            patch("app.main.authorize_customer_access", return_value="customer-1"),
            patch("app.conversation_memory._resolve_redis_url", return_value="redis://test"),
            patch("app.conversation_memory.Redis.from_url", return_value=self.redis),
            patch("app.main._router_agent") as router_agent,
        ):
            response = self.client.post("/chat", json={
                "message": "Another question",
                "conversation_id": str(self.record.conversation_id),
            })
        self.assertEqual(response.status_code, 409)
        router_agent.assert_not_called()

    def test_unavailable_or_corrupt_store_fails_explicitly(self):
        self.redis.data[next(iter(self.redis.data))] = "not-json"
        self.assertEqual(self.client.get("/operator/escalations").status_code, 503)
        self.assertEqual(self.client.get(self.path).status_code, 503)

    def test_auth_failures_and_bad_requests(self):
        self.principal = "unauthorized"
        self.assertEqual(self.client.get("/operator/escalations").status_code, 403)
        self.assertEqual(self.client.get(self.path).status_code, 403)
        self.assertEqual(self.client.patch(self.path + "/status", json={"status": "taken"}).status_code, 403)
        self.redis.fail_reads = True
        self.assertEqual(self.client.get("/operator/escalations").status_code, 503)
        self.redis.fail_reads = False
        self.principal = "https://issuer.example/\0operator-a"
        self.assertEqual(self.client.get(f"/operator/escalations/{uuid4()}").status_code, 404)
        self.assertEqual(self.client.patch(self.path + "/status", json={"status": "pending"}).status_code, 422)
        self.assertEqual(self.client.patch(self.path + "/status", json={"status": "taken", "operator_id": "other"}).status_code, 422)
        self.assertEqual(self.client.get("/operator/escalations?status=unknown").status_code, 422)

    def test_no_bearer_token_is_rejected(self):
        del app.dependency_overrides[require_authenticated_user]
        self.assertEqual(self.client.get("/operator/escalations").status_code, 401)

    def test_cli_grants_and_revokes_verified_identity(self):
        with patch("app.operator_authorization._client", return_value=self.redis):
            args = ["--issuer", "https://issuer.example/", "--subject", "operator-b"]
            self.assertEqual(manage_operators(["grant", *args]), 0)
            self.assertIn(
                _principal_hash("https://issuer.example/\0operator-b"),
                self.redis.sets[OPERATOR_SET_KEY],
            )
            self.assertEqual(manage_operators(["revoke", *args]), 0)
            self.assertNotIn(
                _principal_hash("https://issuer.example/\0operator-b"),
                self.redis.sets[OPERATOR_SET_KEY],
            )
            self.assertEqual(manage_operators(["grant", "--issuer", "http://bad", "--subject", "x"]), 1)


if __name__ == "__main__":
    unittest.main()
