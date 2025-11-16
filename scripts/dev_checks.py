import math

def linear_schedule(start_val, end_val, progress):
    return start_val + (end_val - start_val) * progress

def get_adaptive_risk(step: int, total_steps: int,
                      start_cfg: dict, target_cfg: dict,
                      current_drawdown: float = 0.0) -> dict:
    progress = min(1.0, max(0.0, step / max(1, total_steps)))
    pos_size = linear_schedule(start_cfg['position_size_pct'], target_cfg['position_size_pct'], progress)
    sl = linear_schedule(start_cfg['stop_loss_pct'], target_cfg['stop_loss_pct'], progress)
    tp = linear_schedule(start_cfg['take_profit_pct'], target_cfg['take_profit_pct'], progress)

    if current_drawdown >= 0.25:
        safety_mult = 0.4
    elif current_drawdown >= 0.15:
        safety_mult = 0.65
    else:
        safety_mult = 1.0

    adapted = {
        'position_size_pct': pos_size * safety_mult,
        'stop_loss_pct': max(0.01, sl),
        'take_profit_pct': tp
    }
    return adapted

def test_adaptive_risk_progression():
    print("--- Testing Adaptive Risk Progression ---")
    start = {'position_size_pct': 0.65, 'stop_loss_pct': 0.129, 'take_profit_pct': 0.104}
    target = {'position_size_pct': 0.15, 'stop_loss_pct': 0.05, 'take_profit_pct': 0.08}
    for s in [0, 10000, 25000, 50000, 100000]:
        r = get_adaptive_risk(s, 100000, start, target, current_drawdown=0.0)
        print(f"step={s:6d} -> pos={r['position_size_pct']:.3f} sl={r['stop_loss_pct']:.3f}")

def test_chunk_carry_logic():
    class MockEnv:
        def __init__(self):
            self.config = {
                "chunk_carry_over": {
                    'keep_unrealized_pct': 0.02,
                    'max_allowed_loss_pct': 0.10,
                    'min_steps_left_to_allow_carry': 6
                }
            }
            class MockLogger:
                def info(self, msg):
                    pass
            self.smart_logger = MockLogger()
            self.steps_per_chunk = 110

        def should_force_close_chunk(self, worker_id: int, asset: str, position: "Position", step_in_chunk: int) -> bool:
            cfg = self.config.get("chunk_carry_over", {})
            keep_unrealized_pct = cfg.get("keep_unrealized_pct", 0.02)
            max_allowed_loss_pct = cfg.get("max_allowed_loss_pct", 0.10)
            min_steps_left = cfg.get("min_steps_left_to_allow_carry", 5)
            data_length = self.steps_per_chunk
            unrealized_pct = 0.0
            try:
                unrealized_pct = position.unrealized_pnl_usd / max(1e-9, position.notional_usd)
            except Exception:
                unrealized_pct = 0.0
            if unrealized_pct >= keep_unrealized_pct:
                return False
            if unrealized_pct <= -max_allowed_loss_pct:
                return True
            if step_in_chunk >= (data_length - min_steps_left):
                return True
            return False

    class P:
        def __init__(self, unrealized_pnl_usd, notional_usd=100):
            self.unrealized_pnl_usd = unrealized_pnl_usd
            self.notional_usd = notional_usd

    env = MockEnv()
    print("\n--- Testing Chunk Carry Logic ---")
    print("\nTest Case 1: Near end of chunk (step 108/110) -> Should force close")
    for pnl in [5, 2, -1, -8, -15]:
        pos = P(unrealized_pnl_usd=pnl, notional_usd=100)
        result = env.should_force_close_chunk(0, "BTCUSDT", pos, step_in_chunk=108)
        print(f"pnl={pnl:3d} USD -> should_force_close: {result}")

    print("\nTest Case 2: Not near end of chunk (step 50/110)")
    for pnl in [5, 2, -1, -8, -15]:
        pos = P(unrealized_pnl_usd=pnl, notional_usd=100)
        result = env.should_force_close_chunk(0, "BTCUSDT", pos, step_in_chunk=50)
        print(f"pnl={pnl:3d} USD -> should_force_close: {result}")

if __name__ == "__main__":
    test_adaptive_risk_progression()
    test_chunk_carry_logic()
