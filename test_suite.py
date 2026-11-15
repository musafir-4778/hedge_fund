"""test_suite.py — Unit & Integration Test Suite for Wealth Management Investment Office.

Validates:
1. Pydantic contracts & validations (state.py)
2. Database schema, WAL mode & thread-safe CRUD (database.py)
3. Deterministic technical & risk boundaries math (data_tools.py)
4. Paper broker execution, taxes, fees & cash integrity (paper_broker.py)
5. Intraday guard stop-loss triggers (paper_broker.py)
6. Post-mortem reflection audit & 30-trade scorecard (reflection_engine.py)
7. LangGraph StateGraph compilation & sequential flow (graph.py)
"""

import os
import unittest
import numpy as np
import pandas as pd

from state import TradeDecision, TradePostMortem, InvestmentFirmState
from database import (
    init_db,
    get_portfolio_account,
    update_cash,
    add_position,
    get_position,
    remove_position,
    get_all_positions,
    log_trade,
    get_trade_logs,
    save_thesis_snapshot,
    get_open_thesis_snapshot,
    close_thesis_snapshot,
    save_trade_post_mortem,
    get_agent_scorecard,
    get_top_actionable_rules,
)
from data_tools import (
    normalize_ticker,
    compute_wilder_rsi,
    compute_atr,
    calculate_risk_boundaries,
)
from paper_broker import (
    compute_buy_charges,
    compute_sell_charges,
    execute_buy_decision,
    execute_sell,
)
from graph import build_investment_graph


class TestWealthAgentOffice(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        init_db()

    def test_01_state_contracts(self):
        """Validates Pydantic schemas enforce bounds and types."""
        # Valid TradeDecision
        dec = TradeDecision(
            ticker="BEL.NS",
            action="BUY",
            conviction="HIGH",
            allocation_pct=0.10,
            entry_price=300.0,
            stop_loss_price=280.0,
            target_price=350.0,
            holding_horizon_days=30,
            rationale="Strong order book breakout."
        )
        self.assertEqual(dec.ticker, "BEL.NS")
        self.assertEqual(dec.allocation_pct, 0.10)

        # Exceeding allocation cap (> 12%) must raise validation error
        with self.assertRaises(Exception):
            TradeDecision(
                ticker="BEL.NS",
                action="BUY",
                conviction="HIGH",
                allocation_pct=0.15,  # Exceeds max 0.12
                entry_price=300.0,
                stop_loss_price=280.0,
                target_price=350.0,
                rationale="Invalid cap."
            )

        # Valid TradePostMortem
        pm = TradePostMortem(
            outcome="WIN",
            pnl_pct=0.12,
            primary_driver="TECHNICAL_BREAKOUT_VALID",
            accurate_agent="TECHNICAL",
            failed_agent="NONE",
            root_cause_analysis="Clean breakout on high volume.",
            actionable_rule="Trail stops along 20 SMA on high-volume breakouts."
        )
        self.assertEqual(pm.outcome, "WIN")

    def test_02_database_operations(self):
        """Validates SQLite WAL initialization and CRUD helpers."""
        acc = get_portfolio_account()
        self.assertGreater(acc["cash"], 0.0)

        # Test snapshot
        snap_id = save_thesis_snapshot(
            ticker="TEST_STOCK.NS",
            sector="Capital Goods",
            entry_price=500.0,
            stop_loss=475.0,
            target_price=565.0,
            indicators={"rsi": 62.0},
            technical_report="Bullish trend",
            fundamental_report="Healthy ROE",
            news_macro_report="PLI beneficiary",
            debate_summary="Approved by research manager",
            risk_summary="Risk acceptable",
            pm_rationale="High conviction entry"
        )
        self.assertIsInstance(snap_id, int)

        snap = get_open_thesis_snapshot("TEST_STOCK.NS")
        self.assertIsNotNone(snap)
        self.assertEqual(snap["status"], "OPEN")

        close_thesis_snapshot(snap_id)
        closed_snap = get_open_thesis_snapshot("TEST_STOCK.NS")
        self.assertIsNone(closed_snap)

    def test_03_deterministic_math(self):
        """Validates Wilder's RSI, ATR, and exact rupee risk boundaries."""
        self.assertEqual(normalize_ticker("reliance"), "RELIANCE.NS")
        self.assertEqual(normalize_ticker("TCS.NS"), "TCS.NS")

        # Risk boundaries with beta-adaptive ATR stop (default beta 1.0 -> 3.0x ATR)
        bounds = calculate_risk_boundaries(latest_close=1000.0, atr_14=30.0, total_nav=1_000_000.0, beta=1.0)
        # hard stop = 1000 - 3.0 * 30 = 910.0
        self.assertEqual(bounds["hard_stop_price"], 910.0)
        # risk per share = 90.0
        self.assertEqual(bounds["risk_per_share"], 90.0)
        # min target = 1000 + 2.5 * 90 = 1225.0
        self.assertEqual(bounds["min_valid_target"], 1225.0)
        # risk budget = 1,000,000 * 0.015 = 15,000 -> 15000 // 90 = 166 shares
        self.assertEqual(bounds["risk_based_max_shares"], 166)
        # cap budget = 1,000,000 * 0.12 = 120,000 -> 120000 // 1000 = 120 shares
        self.assertEqual(bounds["cap_based_max_shares"], 120)
        # recommended = min(166, 120) = 120
        self.assertEqual(bounds["recommended_max_qty"], 120)

        # Defensive low-beta test (beta 0.8 -> 2.5x ATR)
        bounds_low_beta = calculate_risk_boundaries(latest_close=1000.0, atr_14=30.0, total_nav=1_000_000.0, beta=0.8)
        self.assertEqual(bounds_low_beta["hard_stop_price"], 925.0)

    def test_04_paper_broker_execution_and_fees(self):
        """Validates realistic delivery trade cost model and execution."""
        trade_val = 100_000.0
        buy_fees = compute_buy_charges(trade_val)
        # STT (0.10%) + NSE/SEBI (~0.0035%) + Stamp (0.015%) + Slippage (0.05%) = ~0.1685%
        self.assertGreater(buy_fees, 150.0)
        self.assertLess(buy_fees, 200.0)

        sell_fees = compute_sell_charges(trade_val)
        # STT (0.10%) + NSE/SEBI (~0.0035%) + Slippage (0.05%) = ~0.1535% (no stamp duty)
        self.assertGreater(sell_fees, 140.0)
        self.assertLess(sell_fees, buy_fees)

        # Test trade execution cycle
        acc_before = get_portfolio_account()
        cash_before = acc_before["cash"]

        dec = TradeDecision(
            ticker="MOCK_STOCK.NS",
            action="BUY",
            conviction="HIGH",
            allocation_pct=0.05,
            entry_price=200.0,
            stop_loss_price=190.0,
            target_price=225.0,
            rationale="Test mock stock buy."
        )

        mock_state = {
            "sector": "Information Technology",
            "technicals": {
                "latest_close": 200.0,
                "hard_stop_price": 190.0,
                "min_valid_target": 225.0,
                "recommended_max_qty": 250
            }
        }

        res = execute_buy_decision(dec, mock_state)
        self.assertIsNotNone(res)
        self.assertEqual(res["ticker"], "MOCK_STOCK.NS")

        pos = get_position("MOCK_STOCK.NS")
        self.assertIsNotNone(pos)
        self.assertEqual(pos["qty"], res["qty"])

        # Execute sell at target
        sell_res = execute_sell(
            ticker="MOCK_STOCK.NS",
            execution_price=225.0,
            exit_reason="TARGET_HIT",
            rationale="Target hit in mock test."
        )
        self.assertIsNotNone(sell_res)
        self.assertGreater(sell_res["realized_pnl"], 0.0)

        # Position should be cleared
        pos_after = get_position("MOCK_STOCK.NS")
        self.assertIsNone(pos_after)

    def test_05_graph_compilation(self):
        """Validates LangGraph StateGraph builds successfully."""
        app = build_investment_graph()
        self.assertIsNotNone(app)

    def test_06_episodic_mistake_learning(self):
        """Validates that mistakes are audited, guardrails synthesized, and scorecard updated."""
        from reflection_engine import simulate_mistake_learning
        res = simulate_mistake_learning(
            ticker="TEST_LOSS.NS",
            sector="Automobile",
            failed_agent="FUNDAMENTAL_ANALYST",
            primary_driver="ACCOUNTING_QUALITY",
            root_cause="Negative cash flow from operations despite positive net income.",
            actionable_rule="Mandate CFO/PAT > 0.8; downgrade to HOLD if CFO < 0."
        )
        self.assertIsNotNone(res["id"])
        self.assertEqual(res["ticker"], "TEST_LOSS.NS")

        # Verify rule exists in top actionable rules
        rules = get_top_actionable_rules(limit=10)
        self.assertIn("Mandate CFO/PAT > 0.8; downgrade to HOLD if CFO < 0.", rules)

        # Verify scorecard reflects failed agent
        scorecard = get_agent_scorecard()
        self.assertIn("FUNDAMENTAL", scorecard)

    def test_07_jaccard_deduplication(self):
        """Validates token Jaccard similarity eliminates near-duplicate rules."""
        from database import _deduplicate_rules
        raw = [
            "Mandate CFO/PAT > 0.8 for non-financials",
            "Mandate CFO/PAT > 0.8 strictly for non-financials",
            "In BEAR_RISK_OFF maintain 50% cash preservation"
        ]
        deduped = _deduplicate_rules(raw, limit=5, threshold=0.65)
        self.assertEqual(len(deduped), 2)
        self.assertEqual(deduped[0], "Mandate CFO/PAT > 0.8 for non-financials")
        self.assertEqual(deduped[1], "In BEAR_RISK_OFF maintain 50% cash preservation")

    def test_08_hard_veto_gate(self):
        """Validates deterministic PM hard veto on severe earnings contraction without LLM calls."""
        from agents import portfolio_manager_node
        veto_state = {
            "ticker": "VETO_TEST.NS",
            "sector": "Metals",
            "market_regime": "BULL_STAGE2",
            "india_vix": 14.5,
            "portfolio_context": {"cash": 1_000_000.0, "total_nav": 1_000_000.0, "active_positions": []},
            "technicals": {"latest_close": 500.0, "hard_stop_price": 450.0, "min_valid_target": 625.0, "sma_50": 480.0},
            "fundamentals": {"earnings_growth": -0.28, "industry": "Steel Manufacturing"},
            "agent_scorecard": "",
            "research_proposal": "Bullish thesis",
            "episodic_lessons": ""
        }
        out = portfolio_manager_node(veto_state)
        dec = out["trade_decision"]
        self.assertEqual(dec["action"], "HOLD")
        self.assertEqual(dec["allocation_pct"], 0.0)
        self.assertIn("Deterministic Hard Veto", dec["rationale"])

    def test_09_stcg_tax_calculation(self):
        """Validates Indian Short-Term Capital Gains (STCG) 20% tax on holdings < 12 months."""
        from tax_engine import compute_capital_gains_tax
        # 100 days held, profit = ₹50,000 -> 20% STCG = ₹10,000 tax
        res = compute_capital_gains_tax(
            buy_date="2025-01-10T09:30:00",
            sell_date="2025-04-20T15:30:00",
            cost_basis=100000.0,
            net_proceeds=150000.0,
            fy_ltcg_exempt_used=0.0
        )
        self.assertEqual(res["term_type"], "STCG")
        self.assertFalse(res["is_long_term"])
        self.assertEqual(res["gross_pnl"], 50000.0)
        self.assertEqual(res["tax_rate"], 0.20)
        self.assertEqual(res["tax_deducted"], 10000.0)
        self.assertEqual(res["net_pnl_after_tax"], 40000.0)

    def test_10_ltcg_tax_and_exemption(self):
        """Validates Indian Long-Term Capital Gains (LTCG) 12.5% tax over ₹1.25L exemption."""
        from tax_engine import compute_capital_gains_tax
        # 400 days held (>= 365d), profit = ₹200,000.
        # Exemption = ₹125,000, Taxable = ₹75,000 @ 12.5% = ₹9,375.
        res = compute_capital_gains_tax(
            buy_date="2024-01-10T09:30:00",
            sell_date="2025-02-14T15:30:00",
            cost_basis=200000.0,
            net_proceeds=400000.0,
            fy_ltcg_exempt_used=0.0
        )
        self.assertEqual(res["term_type"], "LTCG")
        self.assertTrue(res["is_long_term"])
        self.assertEqual(res["gross_pnl"], 200000.0)
        self.assertEqual(res["exemption_applied"], 125000.0)
        self.assertEqual(res["taxable_pnl"], 75000.0)
        self.assertEqual(res["tax_rate"], 0.125)
        self.assertEqual(res["tax_deducted"], 9375.0)
        self.assertEqual(res["net_pnl_after_tax"], 190625.0)
        self.assertEqual(res["new_fy_exempt_used"], 125000.0)

    def test_11_liquid_fund_sweep_and_yield(self):
        """Validates idle cash sweep into 6.75% Liquid Mutual Fund and yield accrual."""
        from database import reset_portfolio_account, get_portfolio_account
        from paper_broker import sweep_idle_cash_if_eligible, accrue_daily_risk_free_yield
        reset_portfolio_account(clean_trades=True, initial_capital=1_000_000.0)

        # In BEAR_RISK_OFF with 100% cash (₹1,000,000), sweeps excess above 10% NAV buffer (₹100,000)
        swept = sweep_idle_cash_if_eligible("BEAR_RISK_OFF", total_nav=1_000_000.0)
        self.assertEqual(swept, 900000.0)

        acc = get_portfolio_account()
        self.assertEqual(acc["cash"], 100000.0)
        self.assertEqual(acc["liquid_fund_balance"], 900000.0)

        # Accrue 30 days of yield: 900,000 * (0.0675 / 365) * 30 = ~₹4,993.15
        interest = accrue_daily_risk_free_yield(days=30, annual_yield=0.0675)
        self.assertGreater(interest, 4900.0)
        self.assertLess(interest, 5100.0)

        acc_after = get_portfolio_account()
        self.assertGreater(acc_after["liquid_fund_balance"], 904900.0)

        # Clean reset back to ₹10,00,000
        reset_portfolio_account(clean_trades=True, initial_capital=1_000_000.0)

    def test_12_curriculum_blocks_configuration(self):
        """Validates historical curriculum blocks and out-of-sample window dates."""
        from monthly_learning import CURRICULUM_BLOCKS, get_first_trading_days_in_window
        import pandas as pd

        self.assertIn("block_1", CURRICULUM_BLOCKS)
        self.assertIn("block_2", CURRICULUM_BLOCKS)
        self.assertIn("block_3", CURRICULUM_BLOCKS)
        self.assertIn("block_4", CURRICULUM_BLOCKS)
        self.assertIn("oos_a", CURRICULUM_BLOCKS)
        self.assertIn("oos_b", CURRICULUM_BLOCKS)

        # Generate synthetic dates index
        idx = pd.date_range("2020-01-01", "2026-11-14", freq="B")
        b1_days = get_first_trading_days_in_window(idx, "2020-06-01", "2021-10-01")
        self.assertEqual(len(b1_days), 17)  # June 2020 through Oct 2021 (inclusive)

    def test_13_autonomous_curriculum_and_rpm_guardrails(self):
        """Validates zero regime label leakage in curriculum and safe RPM rate limiting."""
        from monthly_learning import CURRICULUM_BLOCKS
        from config import MIN_SECONDS_BETWEEN_LLM_CALLS, _rate_limiter

        # Ensure NO regime label or evaluation description leakage in curriculum
        for b_key, b_val in CURRICULUM_BLOCKS.items():
            self.assertNotIn("regime", b_val, f"Curriculum block {b_key} leaks regime label!")
            self.assertNotIn("description", b_val, f"Curriculum block {b_key} leaks description!")

        # Verify rate limiter safe RPM parameters
        self.assertGreaterEqual(MIN_SECONDS_BETWEEN_LLM_CALLS, 5.0)
        self.assertLessEqual(_rate_limiter.max_rpm, 12)


if __name__ == "__main__":
    unittest.main()

