import json
import unittest
from unittest.mock import Mock

from getnet_ingestion.schema import CUSTOMER_KEY_PREFIX
from getnet_ingestion.seed import SYNTHETIC_CUSTOMERS, seed_synthetic_customers


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


if __name__ == "__main__":
    unittest.main()
