import unittest

from app.agents.router import ROUTER_SYSTEM_PROMPT


class RouterPromptTests(unittest.TestCase):
    def test_prompt_answers_directly_on_follow_up_questions(self):
        self.assertIn("Lead with the direct practical answer", ROUTER_SYSTEM_PROMPT)
        self.assertIn(
            "synthesize the available evidence instead\nof repeating caveats",
            ROUTER_SYSTEM_PROMPT,
        )


if __name__ == "__main__":
    unittest.main()
