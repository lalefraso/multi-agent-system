import json
import unittest
from unittest.mock import Mock, patch

from getnet_ingestion.schema import CUSTOMER_KEY_PREFIX, TERMINALS_KEY_PREFIX, TRANSACTIONS_KEY_PREFIX
from getnet_ingestion.seed import (
    SYNTHETIC_CUSTOMERS,
    SYNTHETIC_TERMINALS,
    SYNTHETIC_TRANSACTIONS,
    RedisSeedError,
    seed_redis,
    seed_synthetic_customers,
    seed_synthetic_support_data,
)


class SeedTests(unittest.TestCase):
    def test_seeds_synthetic_customers_without_overwriting_existing_records(self):
        client = Mock()
        client.set.side_effect = [True, False, True, False, True]

        result = seed_synthetic_customers(client)

        self.assertEqual(result, (3, 2))
        self.assertEqual(client.set.call_count, 5)
        first_call = client.set.call_args_list[0]
        self.assertEqual(
            first_call.args[0],
            f"{CUSTOMER_KEY_PREFIX}cliente1988",
        )
        self.assertEqual(json.loads(first_call.args[1]), SYNTHETIC_CUSTOMERS[0])
        self.assertIs(first_call.kwargs["nx"], True)
        self.assertTrue(all(customer["synthetic"] for customer in SYNTHETIC_CUSTOMERS))

    def test_support_datasets_are_seeded_without_overwriting(self):
        client = Mock()
        client.set.side_effect = [True, False] * 5

        self.assertEqual(seed_synthetic_support_data(client), (5, 5))
        self.assertEqual(client.set.call_count, 10)
        first_call = client.set.call_args_list[0]
        self.assertEqual(first_call.args[0], TRANSACTIONS_KEY_PREFIX + "cliente1988")
        self.assertEqual(json.loads(first_call.args[1]), SYNTHETIC_TRANSACTIONS["cliente1988"])
        self.assertTrue(all(call.kwargs["nx"] for call in client.set.call_args_list))
        self.assertEqual(
            client.set.call_args_list[5].args[0], TERMINALS_KEY_PREFIX + "cliente1988"
        )
        self.assertEqual(set(SYNTHETIC_TERMINALS), set(SYNTHETIC_TRANSACTIONS))

    def test_seed_adds_support_data_to_existing_accounts_idempotently(self):
        stored = {CUSTOMER_KEY_PREFIX + "cliente1988": "existing account"}

        def set_if_absent(key, value, *, nx):
            self.assertTrue(nx)
            if key in stored:
                return False
            stored[key] = value
            return True

        client = Mock()
        client.set.side_effect = set_if_absent
        self.assertEqual(seed_synthetic_customers(client), (4, 1))
        self.assertEqual(seed_synthetic_support_data(client), (10, 0))
        self.assertEqual(seed_synthetic_support_data(client), (0, 10))
        self.assertEqual(stored[CUSTOMER_KEY_PREFIX + "cliente1988"], "existing account")
        self.assertEqual(
            json.loads(stored[TRANSACTIONS_KEY_PREFIX + "cliente1988"]),
            SYNTHETIC_TRANSACTIONS["cliente1988"],
        )

    @patch("getnet_ingestion.seed.Redis.from_url")
    @patch("getnet_ingestion.seed.persist_corpora", return_value=42)
    @patch("getnet_ingestion.seed.embeddings_for_locale")
    @patch("getnet_ingestion.seed._resolve_redis_url", return_value="redis://test")
    def test_seed_redis_loads_all_datasets(
        self, _resolve, embeddings, persist, redis_from_url
    ):
        client = redis_from_url.return_value
        client.set.return_value = True
        client.scan_iter.return_value = iter(())
        self.assertEqual(seed_redis(), (5, 0, 42, 10, 0))
        self.assertEqual(embeddings.call_count, 2)
        persist.assert_called_once()
        client.ping.assert_called_once()
        self.assertEqual(client.set.call_count, 15)
        client.close.assert_called_once()

    @patch("getnet_ingestion.seed.Redis.from_url")
    @patch("getnet_ingestion.seed.persist_corpora")
    @patch("getnet_ingestion.seed._resolve_redis_url", return_value="redis://test")
    def test_seed_refuses_old_keys_before_writing(self, _resolve, persist, redis_from_url):
        client = redis_from_url.return_value
        client.scan_iter.return_value = iter(["customers:customer:cliente1988"])

        with self.assertRaisesRegex(RedisSeedError, "Old Redis keys detected"):
            seed_redis()

        persist.assert_not_called()
        client.set.assert_not_called()
        client.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
