"""Fake LLM backend for tests only: answers the v4 constrained schemas without a model."""

import re


class FakeBackend:
    """Answers the constrained schema like a model would: picks the first facts with their first allowed label,
    and answers the synthesis correctly from the counts in its prompt."""

    name = "fake"

    def __init__(self, correct_synthesis=True, slow=0.0):
        self.calls, self.correct, self.slow = [], correct_synthesis, slow

    def complete_json(self, system, prompt, schema=None, should_cancel=None, on_token=None):
        import time

        self.calls.append(prompt)
        for i in range(int(self.slow * 20)):
            time.sleep(0.05)
            if should_cancel and should_cancel():
                from riskcausal_agents.backends import Cancelled

                raise Cancelled("x")
            if on_token:
                on_token(i)
        p = schema["properties"]
        if "key_facts" in p:
            ids = p["key_facts"]["items"]["enum"]
            n, k = map(int, re.search(r"Counted facts: (\d+); unusual: (\d+)", prompt).groups())
            ov = "not enough evidence" if not n else "mostly unusual" if k / n >= .6 else "mostly typical" if k / n <= .3 else "mixed"
            miss = re.search(r"Agents with no evidence: (.*)", prompt).group(1)
            return {"key_facts": ids[:2], "overall": ov if self.correct else "mixed",
                    "missing_agents": [] if miss == "none" else miss.split(", "), "note": ""}, {}
        alts = p["claims"]["items"]["anyOf"]
        return {"claims": [{"fact_id": a["properties"]["fact_id"]["enum"][0], "reading": "",
                            "label": a["properties"]["label"]["enum"][0], "note": ""} for a in alts[:3]],
                "abstain": False, "abstain_reason": ""}, {}
