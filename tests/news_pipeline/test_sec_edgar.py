import unittest

from news_pipeline.sources.sec_edgar import (
    classify_sec_form,
    collect_sec_edgar_articles,
)
from news_pipeline.tickers import TrackedTicker


class SecEdgarTests(unittest.TestCase):
    def test_sec_edgar_normalizes_recent_filings(self):
        ticker = TrackedTicker("NVDA", "NVIDIA", ("Nvidia",), "portfolio")

        def fetcher(url, user_agent, timeout):
            if url.endswith("company_tickers.json"):
                return {"0": {"ticker": "NVDA", "cik_str": 1045810}}
            return {
                "filings": {
                    "recent": {
                        "form": ["8-K", "4"],
                        "filingDate": ["2026-06-08", "2026-06-08"],
                        "accessionNumber": [
                            "0001045810-26-000001",
                            "0001045810-26-000002",
                        ],
                        "primaryDocument": ["nvda-8k.htm", "nvda-form4.xml"],
                        "primaryDocDescription": [
                            "Current report",
                            "Insider transaction",
                        ],
                    }
                }
            }

        articles, attempts = collect_sec_edgar_articles(
            (ticker,),
            run_date="2026-06-09",
            user_agent="UnitTest/1.0",
            timeout_seconds=1,
            rate_limit_seconds=0,
            json_fetcher=fetcher,
        )

        self.assertEqual(len(articles), 1)
        article = articles[0]
        self.assertEqual(article.metadata["source_family"], "regulatory_official")
        self.assertEqual(article.metadata["filing_form_type"], "8-K")
        self.assertEqual(article.metadata["accession_number"], "0001045810-26-000001")
        self.assertEqual(article.metadata["ticker"], "NVDA")
        self.assertEqual(article.metadata["ticker_match_confidence"], 1.0)
        self.assertIn("Archives/edgar/data/1045810/", article.canonical_url)
        self.assertEqual(attempts[0].status, "success")

    def test_supported_sec_forms_receive_event_classification(self):
        ticker = TrackedTicker("NVDA", "NVIDIA", ("Nvidia",), "portfolio")
        forms = [
            "8-K",
            "10-Q",
            "10-K",
            "6-K",
            "SC 13D",
            "SC 13G",
            "DEF 14A",
            "S-1",
            "424B5",
        ]

        def fetcher(url, user_agent, timeout):
            if url.endswith("company_tickers.json"):
                return {"0": {"ticker": "NVDA", "cik_str": 1045810}}
            return {
                "filings": {
                    "recent": {
                        "form": forms,
                        "filingDate": ["2026-06-08"] * len(forms),
                        "accessionNumber": [
                            f"0001045810-26-{index:06d}"
                            for index in range(1, len(forms) + 1)
                        ],
                        "primaryDocument": [
                            f"filing-{index}.htm"
                            for index in range(1, len(forms) + 1)
                        ],
                        "primaryDocDescription": [
                            f"{form} filing"
                            for form in forms
                        ],
                    }
                }
            }

        articles, _attempts = collect_sec_edgar_articles(
            (ticker,),
            run_date="2026-06-09",
            user_agent="UnitTest/1.0",
            timeout_seconds=1,
            rate_limit_seconds=0,
            max_filings_per_ticker=20,
            json_fetcher=fetcher,
        )

        classifications = {
            article.metadata["filing_form_type"]: article.metadata[
                "filing_event_type"
            ]
            for article in articles
        }
        self.assertEqual(classifications["8-K"], "material_event")
        self.assertEqual(classifications["10-Q"], "quarterly_report")
        self.assertEqual(classifications["10-K"], "annual_report")
        self.assertEqual(classifications["6-K"], "foreign_issuer_report")
        self.assertEqual(classifications["SC 13D"], "ownership_event")
        self.assertEqual(classifications["SC 13G"], "ownership_event")
        self.assertEqual(classifications["DEF 14A"], "proxy_event")
        self.assertEqual(classifications["S-1"], "registration_event")
        self.assertEqual(
            classifications["424B5"],
            "offering_or_prospectus",
        )
        self.assertTrue(
            all(article.metadata["sec_event_summary"] for article in articles)
        )
        self.assertTrue(
            all(article.metadata["sec_event_basis"] for article in articles)
        )

    def test_sec_form_aliases_are_classified(self):
        self.assertEqual(classify_sec_form("13D")[0], "ownership_event")
        self.assertEqual(classify_sec_form("SC13G/A")[0], "ownership_event")
        self.assertEqual(classify_sec_form("DEF14A")[0], "proxy_event")
        self.assertEqual(
            classify_sec_form("424B3")[0],
            "offering_or_prospectus",
        )


if __name__ == "__main__":
    unittest.main()
