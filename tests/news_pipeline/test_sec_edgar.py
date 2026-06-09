import unittest

from news_pipeline.sources.sec_edgar import collect_sec_edgar_articles
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


if __name__ == "__main__":
    unittest.main()
