"""Tests for the shared closed-set scoring toolkit.

Pure-stdlib (unittest), deterministic, no network. Covers the edge cases that
keep the optimizer's accept/reject gate honest: empty sets, hallucinated IDs,
case/whitespace normalization, partial overlap, and answer extraction.
"""
import unittest

from skillopt.envs._shared.setscore import (
    build_eval_detail,
    extract_answer,
    normalize_id,
    parse_id_set,
    set_em,
    set_f1,
)


class TestExtractAnswer(unittest.TestCase):
    def test_last_answer_block_wins(self):
        text = "first <answer>AC-1</answer> then <answer>AC-7</answer>"
        self.assertEqual(extract_answer(text), "AC-7")

    def test_case_insensitive_tags(self):
        self.assertEqual(extract_answer("<ANSWER>CKV_AWS_19</ANSWER>"), "CKV_AWS_19")

    def test_fallback_to_last_nonempty_line(self):
        self.assertEqual(extract_answer("reasoning\n\nCWE-79\n"), "CWE-79")

    def test_empty(self):
        self.assertEqual(extract_answer(""), "")


class TestNormalizeId(unittest.TestCase):
    def test_upper_and_strip_edges(self):
        self.assertEqual(normalize_id(" `ckv_aws_19`. "), "CKV_AWS_19")

    def test_keeps_inner_hyphen_and_underscore(self):
        self.assertEqual(normalize_id("ac-7"), "AC-7")
        self.assertEqual(normalize_id("cwe-917"), "CWE-917")


class TestParseIdSet(unittest.TestCase):
    def test_comma_and_whitespace_split(self):
        self.assertEqual(
            parse_id_set("<answer>AC-7, AC-8  AC-9</answer>"),
            {"AC-7", "AC-8", "AC-9"},
        )

    def test_dedupe_and_case(self):
        self.assertEqual(parse_id_set("<answer>ac-7, AC-7</answer>"), {"AC-7"})

    def test_allowed_filters_hallucinations(self):
        # ZZ-99 is not in the candidate set → dropped, cannot inflate recall.
        parsed = parse_id_set(
            "<answer>AC-7, ZZ-99</answer>", allowed=["AC-7", "AC-8"]
        )
        self.assertEqual(parsed, {"AC-7"})

    def test_empty_sentinel_maps_to_empty_set(self):
        self.assertEqual(parse_id_set("<answer>NONE</answer>"), set())
        self.assertEqual(parse_id_set("<answer>SAFE</answer>"), set())

    def test_already_extracted(self):
        self.assertEqual(
            parse_id_set("CKV_AWS_19", already_extracted=True), {"CKV_AWS_19"}
        )


class TestSetEm(unittest.TestCase):
    def test_exact(self):
        self.assertEqual(set_em({"A", "B"}, {"B", "A"}), 1.0)

    def test_mismatch(self):
        self.assertEqual(set_em({"A"}, {"A", "B"}), 0.0)

    def test_both_empty(self):
        self.assertEqual(set_em(set(), set()), 1.0)


class TestSetF1(unittest.TestCase):
    def test_both_empty_is_one(self):
        self.assertEqual(set_f1(set(), set()), 1.0)

    def test_empty_vs_nonempty_is_zero(self):
        self.assertEqual(set_f1(set(), {"A"}), 0.0)
        self.assertEqual(set_f1({"A"}, set()), 0.0)

    def test_partial_overlap(self):
        # pred={A,B}, gold={B,C}: tp=1, P=1/2, R=1/2, F1=0.5
        self.assertAlmostEqual(set_f1({"A", "B"}, {"B", "C"}), 0.5)

    def test_no_overlap(self):
        self.assertEqual(set_f1({"A"}, {"B"}), 0.0)

    def test_perfect(self):
        self.assertEqual(set_f1({"A", "B"}, {"A", "B"}), 1.0)


class TestBuildEvalDetail(unittest.TestCase):
    def test_lists_missed_and_extra(self):
        detail = build_eval_detail(
            question="q",
            predicted={"A", "X"},
            gold={"A", "B"},
            em=0.0,
            f1=0.5,
        )
        self.assertIn("Missed (in gold, not predicted): ['B']", detail)
        self.assertIn("Extra (predicted, not in gold): ['X']", detail)
        self.assertIn("Set F1: 0.5000", detail)


if __name__ == "__main__":
    unittest.main()
