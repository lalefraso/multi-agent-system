import fnmatch
import unittest
from unittest.mock import Mock, patch

from redis.exceptions import RedisError

from getnet_ingestion.migrate_redis_keys import (
    RedisMigrationError,
    has_old_keys,
    main,
    migrate_keys,
)


class FakeRedis:
    def __init__(self, data):
        self.data = dict(data)

    def scan_iter(self, *, match, count):
        return iter(key for key in list(self.data) if fnmatch.fnmatchcase(key, match))

    def exists(self, key):
        return key in self.data

    def renamenx(self, source, destination):
        if destination in self.data:
            return False
        self.data[destination] = self.data.pop(source)
        return True


class MigrationTests(unittest.TestCase):
    def test_dry_run_then_move_all_namespaces_and_repeat(self):
        original = {
            "customers:customer:1": ("string", "customer", None),
            "getnet_users:customer:2": ("hash", {"name": "legacy"}, None),
            "customers:transactions:1": ("string", "[]", None),
            "customers:terminals:1": ("string", "[]", None),
            "getnet:chat:abc": ("list", ["message"], 600),
            "getnet:chat:abc:lock": ("string", "token", 60),
            "getnet:conversation-owner:uuid": ("string", "owner", 600),
            "getnet:authz:principal:hash:customers": ("set", {"1"}, None),
            "getnet:authz:customer:hash:owner": ("string", "hash", None),
            "getnet:authz:operators": ("set", {"operator"}, None),
            "getnet:escalation:conversation:uuid": ("string", "case", None),
            "getnet:section:document": ("hash", {"text": "knowledge"}, None),
        }
        client = FakeRedis(original)
        self.assertTrue(has_old_keys(client))
        self.assertEqual(migrate_keys(client), 11)
        self.assertEqual(client.data, original)
        self.assertEqual(migrate_keys(client, apply=True), 11)
        expected = {
            "getnet:customers:1": original["customers:customer:1"],
            "getnet:customers:2": original["getnet_users:customer:2"],
            "getnet:transactions:1": original["customers:transactions:1"],
            "getnet:terminals:1": original["customers:terminals:1"],
            "chat:history:abc": original["getnet:chat:abc"],
            "chat:history:abc:lock": original["getnet:chat:abc:lock"],
            "chat:conversations:uuid:owner": original["getnet:conversation-owner:uuid"],
            "chat:authz:principals:hash:customers": original["getnet:authz:principal:hash:customers"],
            "chat:authz:customers:hash:owner": original["getnet:authz:customer:hash:owner"],
            "chat:authz:operators": original["getnet:authz:operators"],
            "chat:escalation:conversation:uuid": original["getnet:escalation:conversation:uuid"],
            "getnet:section:document": original["getnet:section:document"],
        }
        self.assertEqual(client.data, expected)
        self.assertFalse(has_old_keys(client))
        self.assertEqual(migrate_keys(client, apply=True), 0)

    def test_moves_previous_chat_namespaces_without_changing_values_or_ttls(self):
        original = {
            "chat:conversation-owner:uuid": ("string", "owner", 600),
            "chat:authz:principal:hash:customers": ("set", {"1"}, None),
            "chat:authz:customer:hash:owner": ("string", "hash", None),
            "chat:authz:operators": ("set", {"operator"}, None),
        }
        client = FakeRedis(original)
        self.assertEqual(migrate_keys(client, apply=True), 3)
        self.assertEqual(client.data, {
            "chat:conversations:uuid:owner": original["chat:conversation-owner:uuid"],
            "chat:authz:principals:hash:customers": original["chat:authz:principal:hash:customers"],
            "chat:authz:customers:hash:owner": original["chat:authz:customer:hash:owner"],
            "chat:authz:operators": original["chat:authz:operators"],
        })
        self.assertFalse(has_old_keys(client))

    def test_versions_of_same_authorization_conflict_before_writes(self):
        original = {
            "getnet:authz:principal:hash:customers": "old",
            "chat:authz:principal:hash:customers": "previous",
        }
        client = FakeRedis(original)
        with self.assertRaisesRegex(RedisMigrationError, "Conflicting"):
            migrate_keys(client, apply=True)
        self.assertEqual(client.data, original)

    def test_existing_destination_rejects_all_moves(self):
        original = {
            "customers:customer:1": "old",
            "customers:transactions:1": "old transaction",
            "getnet:transactions:1": "new transaction",
        }
        client = FakeRedis(original)
        with self.assertRaisesRegex(RedisMigrationError, "Conflicting"):
            migrate_keys(client, apply=True)
        self.assertEqual(client.data, original)

    def test_overlapping_old_customers_reject_all_moves(self):
        original = {
            "customers:customer:1": "current",
            "getnet_users:customer:1": "legacy",
        }
        client = FakeRedis(original)
        with self.assertRaisesRegex(RedisMigrationError, "Conflicting"):
            migrate_keys(client, apply=True)
        self.assertEqual(client.data, original)

    def test_race_and_redis_errors_are_reported(self):
        client = Mock()
        client.scan_iter.side_effect = lambda **kwargs: iter(
            ["customers:customer:1"] if kwargs["match"] == "customers:customer:*" else []
        )
        client.exists.return_value = False
        client.renamenx.return_value = False
        with self.assertRaisesRegex(RedisMigrationError, "some keys may"):
            migrate_keys(client, apply=True)
        client.renamenx.side_effect = RedisError("offline")
        with self.assertRaisesRegex(RedisMigrationError, "Redis migration failed"):
            migrate_keys(client, apply=True)

    def test_interrupted_migration_can_be_retried(self):
        client = FakeRedis({
            "customers:customer:1": "customer",
            "customers:transactions:1": "transactions",
        })
        rename = client.renamenx
        calls = 0

        def interrupt_after_first(source, destination):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RedisError("offline")
            return rename(source, destination)

        with patch.object(client, "renamenx", side_effect=interrupt_after_first):
            with self.assertRaisesRegex(RedisMigrationError, "some keys may"):
                migrate_keys(client, apply=True)
        self.assertEqual(client.data["getnet:customers:1"], "customer")
        self.assertEqual(migrate_keys(client, apply=True), 1)
        self.assertEqual(client.data["getnet:transactions:1"], "transactions")
        self.assertFalse(has_old_keys(client))

    @patch("getnet_ingestion.migrate_redis_keys.Redis.from_url")
    @patch("getnet_ingestion.migrate_redis_keys._resolve_redis_url", return_value="redis://test")
    @patch("sys.argv", ["migrate_redis_keys"])
    def test_cli_defaults_to_dry_run(self, _resolve, from_url):
        client = from_url.return_value
        client.scan_iter.return_value = iter(())
        self.assertEqual(main(), 0)
        client.renamenx.assert_not_called()
        client.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
