import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from quant_production import InputValidator, TradeMetrics, PositionType

def test_symbol_validation():
    assert InputValidator.sanitize_symbol(" aApL ") == "AAPL"
    assert InputValidator.sanitize_symbol("brk.b") == "BRK.B"

def test_trade_pnl():
    t = TradeMetrics(
        symbol="TEST",
        entry_price=100,
        exit_price=105,
        position_size=10,
        position_type=PositionType.LONG
    )
    assert t.net_pnl == 50          # (105-100)*10 - 0 commission
    assert abs(t.return_pct - 5.0) < 0.01  # 5% return