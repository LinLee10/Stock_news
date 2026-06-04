import unittest

from news_pipeline.tickers import (
    load_portfolio,
    load_tracked_tickers,
    load_watchlist,
    match_tickers,
    symbols,
    ticker_lookup,
)


class TickerConfigTests(unittest.TestCase):
    def test_portfolio_loads_expected_symbols(self):
        self.assertEqual(
            symbols("portfolio"),
            ("SNDK", "ASML", "MU", "AVGO", "NBIS", "NVDA", "PLTR"),
        )
        lookup = ticker_lookup("portfolio")
        self.assertEqual(lookup["SNDK"].company_name, "SanDisk")
        self.assertIn("NVIDIA Corporation", lookup["NVDA"].aliases)

    def test_watchlist_loads_expected_symbols(self):
        self.assertEqual(
            symbols("watchlist"),
            (
                "ADI",
                "VRT",
                "MRVL",
                "PANW",
                "CRWV",
                "APLD",
                "CORZ",
                "GEV",
                "META",
                "AMD",
                "TSM",
                "ARM",
                "RDDT",
                "ACHR",
            ),
        )
        lookup = ticker_lookup("watchlist")
        self.assertEqual(lookup["PANW"].company_name, "Palo Alto Networks")
        self.assertIn("TSMC", lookup["TSM"].aliases)

    def test_combined_config_has_no_duplicate_symbols(self):
        tracked = load_tracked_tickers()
        all_symbols = [ticker.symbol for ticker in tracked]

        self.assertEqual(len(all_symbols), len(set(all_symbols)))
        self.assertEqual(len(load_portfolio()), 7)
        self.assertEqual(len(load_watchlist()), 14)

    def test_match_tickers_uses_symbols_company_names_and_aliases(self):
        text = (
            "NVIDIA and Taiwan Semiconductor Manufacturing led chip stocks, "
            "while Palantir Technologies and CoreWeave gained attention."
        )
        matched = {ticker.symbol for ticker in match_tickers(text)}

        self.assertEqual(matched, {"NVDA", "TSM", "PLTR", "CRWV"})

    def test_group_limited_matching(self):
        text = "Palantir and CoreWeave were both mentioned."

        self.assertEqual({ticker.symbol for ticker in match_tickers(text, "portfolio")}, {"PLTR"})
        self.assertEqual({ticker.symbol for ticker in match_tickers(text, "watchlist")}, {"CRWV"})

    def test_ambiguous_short_tickers_require_market_or_company_context(self):
        self.assertEqual({ticker.symbol for ticker in match_tickers("The robotic arm moved the sample.")}, set())
        self.assertEqual({ticker.symbol for ticker in match_tickers("A meta-analysis reviewed trial data.")}, set())
        self.assertEqual({ticker.symbol for ticker in match_tickers("MU variant cases declined.")}, set())

        self.assertEqual({ticker.symbol for ticker in match_tickers("ARM stock fell with chip shares.")}, {"ARM"})
        self.assertEqual({ticker.symbol for ticker in match_tickers("Meta enters the enterprise AI race.")}, {"META"})
        self.assertEqual({ticker.symbol for ticker in match_tickers("Micron reported stronger memory revenue.")}, {"MU"})


if __name__ == "__main__":
    unittest.main()
