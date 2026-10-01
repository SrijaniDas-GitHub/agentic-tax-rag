"""Tests for the numeric grounding guardrail and citation formatting.

* Normalization and the whitelist: years, labels and page references are not
  checked, and "$14,600" matches "14,600".
* Numbers are checked against the passages the claim cites, not the whole pack.
"""

from __future__ import annotations

from decimal import Decimal

from agents.citations import format_citation, locator, numbered
from agents.contracts import AnswerClaim, Passage
from guardrails.numeric_grounding import check_answer, extract_numbers, normalise


def us(chunk_id="us_p17_2024:98:0", text="| Single | $14,600 |", page=98, printed=96):
    return Passage(chunk_id=chunk_id, text=text, country="US", tax_year_label="2024",
                   doc_id="us_p17_2024", doc_title="IRS Publication 17 (2024)",
                   page=page, printed_page=printed, score=1.0)


def uk(chunk_id="uk_rates_2024:5:0", text="Higher rate 40% £37,701 to £125,140", page=5):
    return Passage(chunk_id=chunk_id, text=text, country="UK", tax_year_label="2024-25",
                   doc_id="uk_rates_2024", doc_title="gov.uk rates",
                   page=page, printed_page=None, score=1.0)


def claim(text, *chunk_ids):
    return AnswerClaim(text=text, chunk_ids=list(chunk_ids))


# ------------------------------------------------------------- normalisation --

def test_currency_separators_and_symbols_all_normalise_to_one_value():
    assert normalise("$14,600") == normalise("14,600") == normalise("14600") == Decimal(14600)
    assert normalise("£12,570") == Decimal(12570)


def test_a_dollar_figure_grounds_against_a_bare_table_figure():
    report = check_answer([claim("The deduction is $14,600.", "us_p17_2024:98:0")],
                          [us(text="| Single | 14,600 |")])
    assert report.ok and report.checked == 1


# ----------------------------------------------------------------- whitelist --

def test_years_labels_and_page_references_are_not_checked():
    """Years, tax-year labels, and page/section/table references are whitelisted."""
    text = "In 2024 (tax year 2024-25, see p.96 and §5, Table 10-1) the rate applied."
    tokens = extract_numbers(text)
    assert tokens, "expected the numbers to be found before being whitelisted"
    assert all(t.whitelisted for t in tokens), [t for t in tokens if not t.whitelisted]


def test_a_tax_year_label_is_not_split_into_two_numbers():
    """'2024-25' must not be split into 2024 and 25."""
    tokens = extract_numbers("for 2024-25")
    assert all(t.whitelisted for t in tokens)


def test_a_currency_amount_that_looks_like_a_year_is_still_checked():
    """Only bare four-digit years are whitelisted; `$2,024` is a figure."""
    tokens = [t for t in extract_numbers("The fee is $2,024.") if not t.whitelisted]
    assert [t.value for t in tokens] == [Decimal(2024)]


# --------------------------------------------------------------- percentages --

def test_a_percentage_is_its_own_token():
    """A plain 45 must not ground '45%', and vice versa."""
    passage = uk(text="The additional rate threshold is £125,140.")
    report = check_answer([claim("The additional rate is 45%.", "uk_rates_2024:5:0")], [passage])
    assert not report.ok

    report = check_answer([claim("The additional rate is 45%.", "uk_rates_2024:5:0")],
                          [uk(text="Additional rate 45% over £125,140")])
    assert report.ok


# ----------------------------------------------- the binding, and the collision --

def test_a_number_in_the_pack_but_not_in_the_cited_passage_fails():
    """`29,200` appears in several US passages with different meanings, so it must
    be found in the passage this claim cites.

    Limitation: this checks that the figure is in the cited passage, not that the
    passage is about the quantity the sentence names.
    """
    table = us(chunk_id="us_p17_2024:98:0", text="| Married filing jointly | $29,200 |")
    prose = us(chunk_id="us_p17_2024:96:0",
               text="Your standard deduction depends on your filing status and age.")

    right = check_answer([claim("It is $29,200.", "us_p17_2024:98:0")], [table, prose])
    wrong = check_answer([claim("It is $29,200.", "us_p17_2024:96:0")], [table, prose])

    assert right.ok
    assert wrong.ok is False           # the figure is in the pack; not in THIS citation
    assert wrong.violations[0]["kind"] == "ungrounded_number"


def test_an_uncited_figure_is_a_violation_not_an_exemption():
    report = check_answer([claim("The allowance is £12,570.")], [uk()])
    assert not report.ok
    assert "cites none" in report.violations[0]["detail"]


def test_a_citation_to_a_chunk_outside_the_pack_is_reported():
    report = check_answer([claim("The deduction is $14,600.", "invented:1:0")], [us()])
    kinds = {v["kind"] for v in report.violations}
    assert "unknown_citation" in kinds


def test_the_report_names_every_number_matched_or_not():
    """The report lists every number, grounded or not."""
    report = check_answer(
        [claim("In 2024 it is $14,600, up from $13,850.", "us_p17_2024:98:0")], [us()])
    rows = {r["text"]: r for r in report.numbers}
    assert rows["2024"]["whitelisted"] is True
    assert rows["$14,600"]["grounded"] is True
    assert rows["$14,600"]["matched_chunk_id"] == "us_p17_2024:98:0"
    assert rows["$13,850"]["grounded"] is False


# ----------------------------------------------------------------- citations --

def test_us_cites_the_printed_page_not_the_pdf_index():
    """`printed_page = page - 2` for this document."""
    assert locator(us(page=98, printed=96)) == "p.96"
    assert format_citation(us()) == "IRS Publication 17 (2024), p.96"


def test_uk_cites_a_section_because_a_web_page_has_no_page_5():
    assert locator(uk(page=5)) == "§5"
    assert format_citation(uk()) == "gov.uk rates, §5"


def test_markers_are_numbered_by_first_use_and_deduplicated():
    a, b = us(), uk()
    listed, marker = numbered([a, b, a])
    assert [c["n"] for c in listed] == [1, 2]
    assert marker[a.chunk_id] == 1 and marker[b.chunk_id] == 2


def test_two_chunks_of_one_section_share_a_marker():
    """`[1] ..., §1 [2] ..., §1` reads as two sources; it is one."""
    a, b = uk(chunk_id="uk_rates_2024:1:0", page=1), uk(chunk_id="uk_rates_2024:1:1", page=1)
    listed, marker = numbered([a, b, us()])
    assert [(c["n"], c["label"]) for c in listed] == [
        (1, "gov.uk rates, §1"), (2, "IRS Publication 17 (2024), p.96")]
    assert marker[a.chunk_id] == marker[b.chunk_id] == 1


# --------------------------------------------- why a number is ungrounded --
#
# Ungrounded numbers are classified as derived arithmetic (sum or difference of two
# cited figures) or invented. The classification is deterministic.

def test_the_750_case_is_reported_as_arithmetic_not_invention():
    """"increased by $750" is 14,600 - 13,850: true, but stated in no passage."""
    report = check_answer(
        [claim("The standard deduction increased by $750.",
               "us_p17_2024:98:0", "us_p17_2023:98:0")],
        [us(text="| Single | $14,600 |"),
         us(chunk_id="us_p17_2023:98:0", text="| Single | $13,850 |")],
    )
    assert not report.ok
    violation, = report.violations
    assert violation["kind"] == "ungrounded_number"
    assert violation["subkind"] == "derived_arithmetic"
    assert violation["derivation"] == "14600 - 13850"
    assert "did arithmetic" in violation["detail"]


def test_a_sum_is_caught_as_arithmetic_too():
    report = check_answer(
        [claim("Together they come to $28,450.", "us_p17_2024:98:0")],
        [us(text="| Single | $14,600 |\n| Other | $13,850 |")],
    )
    assert report.violations[0]["subkind"] == "derived_arithmetic"
    assert report.violations[0]["derivation"] == "13850 + 14600"


def test_a_figure_nothing_in_the_passage_produces_is_an_invention():
    """No pair of cited figures produces it."""
    report = check_answer(
        [claim("The deduction is $21,317.", "us_p17_2024:98:0")],
        [us(text="| Single | $14,600 |")],
    )
    violation, = report.violations
    assert violation["subkind"] == "invented_figure"
    assert violation["derivation"] is None


def test_an_uncited_claim_has_no_pool_so_it_cannot_be_arithmetic():
    """With no cited passages there is nothing to derive from."""
    report = check_answer([claim("It rose by $750.")], [us()])
    assert report.violations[0]["subkind"] == "invented_figure"


def test_the_derivation_is_stable_across_runs():
    """The same answer must produce the same report every time."""
    passages = [us(text="| a | $14,600 |\n| b | $13,850 |\n| c | $27,700 |\n| d | $1,000 |")]
    claims = [claim("The gap is $750.", "us_p17_2024:98:0")]
    first = check_answer(claims, passages).violations[0]["derivation"]
    for _ in range(5):
        assert check_answer(claims, passages).violations[0]["derivation"] == first
