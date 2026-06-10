import unittest

from news_pipeline.sources.query_planner import plan_ticker_queries
from news_pipeline.tickers import TrackedTicker


class QueryPlannerTests(unittest.TestCase):
    def test_company_context_queries_are_created(self):
        ticker = TrackedTicker(
            "NVDA",
            "NVIDIA",
            ("Nvidia", "NVIDIA Corporation"),
            "portfolio",
        )

        plans = plan_ticker_queries((ticker,))

        self.assertTrue(any('"NVIDIA"' in plan.query_text for plan in plans))
        self.assertTrue(any(plan.query_type == "earnings" for plan in plans))
        self.assertTrue(any(plan.query_type == "guidance" for plan in plans))
        self.assertTrue(any(plan.query_type == "product_ai" for plan in plans))
        self.assertTrue(all(plan.daily_budget_cost_estimate == 1 for plan in plans))

    def test_ambiguous_ticker_avoids_ticker_only_queries(self):
        ticker = TrackedTicker(
            "META",
            "Meta",
            ("Meta Platforms", "Facebook"),
            "watchlist",
        )

        plans = plan_ticker_queries((ticker,))

        self.assertFalse(any(plan.query_type == "ticker_only" for plan in plans))
        self.assertTrue(
            any(
                "Meta Platforms" in plan.query_text
                and plan.query_type in {"company_stock", "company_context"}
                for plan in plans
            )
        )

    def test_weak_coverage_tickers_use_preferred_company_names(self):
        tickers = (
            TrackedTicker("ASML", "ASML", ("ASML Holding",), "portfolio"),
            TrackedTicker("CORZ", "Core Scientific", (), "watchlist"),
            TrackedTicker("META", "Meta", ("Meta Platforms",), "watchlist"),
            TrackedTicker("VRT", "Vertiv", ("Vertiv Holdings",), "watchlist"),
        )

        plans = plan_ticker_queries(
            tickers,
            provider_targets=("gnews", "newsapi", "nyt"),
        )

        expected_names = {
            "ASML": "ASML Holding",
            "CORZ": "Core Scientific",
            "META": "Meta Platforms",
            "VRT": "Vertiv Holdings",
        }
        for ticker, company in expected_names.items():
            ticker_plans = [plan for plan in plans if plan.ticker == ticker]
            self.assertTrue(
                any(f'"{company}"' in plan.query_text for plan in ticker_plans)
            )

    def test_nyt_queries_are_company_context_queries(self):
        ticker = TrackedTicker(
            "PANW",
            "Palo Alto Networks",
            ("Palo Alto",),
            "watchlist",
        )

        plans = plan_ticker_queries((ticker,), provider_targets=("nyt",))

        self.assertTrue(all(plan.source_family == "context_news_api" for plan in plans))
        self.assertFalse(any(plan.query_type == "ticker_only" for plan in plans))
        combined = " ".join(plan.query_text for plan in plans)
        for term in (
            "AI",
            "chips",
            "semiconductors",
            "antitrust",
            "regulation",
            "export controls",
            "data centers",
            "energy infrastructure",
            "cloud",
            "cybersecurity",
        ):
            self.assertIn(term, combined)


if __name__ == "__main__":
    unittest.main()
