# Implementation Plan: ADAN 2.0 Reconstruction

## Overview

This implementation plan breaks down the ADAN 2.0 reconstruction into discrete, manageable coding tasks organized by phase. Each task builds incrementally on previous tasks, with no orphaned code. All testing tasks are required for comprehensive coverage.

**Current Status**: Several core components have been implemented. This task list reflects the remaining work needed.

---

## Phase 1: Data Pipeline Integrity

- [x] 1.1 Implement SeedManager for deterministic initialization
  - ✅ Created `src/adan_trading_bot/utils/seed_manager.py`
  - ✅ Implemented `initialize(seed: int)` to set Python, NumPy, PyTorch seeds
  - ✅ Implemented `get_rng_states()` and `set_rng_states()` for state management
  - ✅ Integrated seed setting into train_parallel_agents.py
  - _Requirements: 7.1, 7.2_

- [ ] 1.2 Write property tests for SeedManager
  - **Property 30: Seed Initialization**
  - **Property 31: Environment Seeding**
  - **Property 32: Deterministic Training**
  - **Property 33: Deterministic Data Shuffling**
  - **Validates: Requirements 7.1, 7.2, 7.3, 7.4**

- [ ] 1.3 Create DataManager for strict temporal separation
  - Create `src/adan_trading_bot/data_processing/data_manager.py`
  - Implement `load_strict_split()` returning (train_df, test_df) with 2021-2023 train, 2024+ test
  - Implement `verify_temporal_separation()` confirming max(train) < min(test)
  - Implement `verify_required_features()` checking for ATR_20, ATR_50, ADX
  - _Requirements: 1.1, 1.2, 1.3, 1.4_

- [ ] 1.4 Write property tests for DataManager
  - **Property 1: Temporal Data Separation**
  - **Property 2: Required Features Present**
  - **Property 40: Data Leakage Detection**
  - **Validates: Requirements 1.1, 1.2, 1.3, 1.4, 9.2**

- [ ] 1.5 Create project cleanup script
  - Create `scripts/clean_project.sh`
  - Remove cached data directories
  - Remove checkpoint directories
  - Remove log directories
  - _Requirements: 1.5_

- [ ] 1.6 Write property test for project cleanup
  - **Property 3: Project Cleanup Completeness**
  - **Validates: Requirements 1.5**

- [ ] 1.7 Checkpoint: Verify data pipeline integrity
  - Ensure all tests pass, ask the user if questions arise.

---

## Phase 2: Realistic Trading Environment

- [x] 2.1 Implement RealisticTradingEnv with market friction models
  - ✅ Created `src/adan_trading_bot/environment/realistic_trading_env.py`
  - ✅ Created `src/adan_trading_bot/environment/market_friction.py` with AdaptiveSlippage, LatencySimulator, LiquidityModel
  - ✅ Integrated with MultiAssetChunkedEnv
  - _Requirements: 2.1, 2.2, 2.3, 2.4_

- [ ] 2.2 Write property tests for RealisticTradingEnv
  - **Property 4: Fee Application Consistency**
  - **Property 5: Slippage Scales with Order Size**
  - **Property 6: Latency Impact on Execution**
  - **Property 7: Liquidity Impact on Price**
  - **Validates: Requirements 2.1, 2.2, 2.3, 2.4**

- [x] 2.3 Implement TradeFrequencyController
  - ✅ Created `src/adan_trading_bot/environment/trade_frequency_controller.py`
  - ✅ Implemented `can_open_trade(asset, current_step)` checking minimum interval
  - ✅ Implemented `can_close_trade(asset, current_step)` checking daily limit
  - ✅ Implemented `record_trade(asset, current_step)` for constraint tracking
  - ✅ Implemented per-asset cooldown period enforcement
  - _Requirements: 2.5, 2.6, 2.7_

- [x] 2.4 Write unit tests for TradeFrequencyController
  - ✅ Created `tests/unit/test_trade_frequency.py` with 10 tests
  - ✅ Tests cover: initialization, natural vs force trades, daily limits, cooldowns, resets
  - **Validates: Requirements 2.5, 2.6, 2.7**

- [x] 2.5 Implement StableRewardCalculator
  - ✅ Created `src/adan_trading_bot/environment/stable_reward_calculator.py`
  - ✅ Implemented `calculate_reward(state)` orchestrating all components
  - ✅ Implemented `_normalize_pnl(pnl)` for PnL normalization
  - ✅ Implemented `_calculate_sharpe_contribution()` for Sharpe component
  - ✅ Implemented `_apply_drawdown_penalty(drawdown)` for drawdown penalty
  - ✅ Implemented `_apply_frequency_penalty(trade_count)` for frequency penalty
  - ✅ Implemented `_apply_consistency_bonus(returns)` for consistency bonus
  - ✅ Implemented reward clipping to [-1.0, 1.0]
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6_

- [ ] 2.6 Write property tests for StableRewardCalculator
  - **Property 11: Normalized PnL Component**
  - **Property 12: Sharpe Contribution Calculation**
  - **Property 13: Drawdown Penalty Application**
  - **Property 14: Trade Frequency Penalty Application**
  - **Property 15: Consistency Bonus Application**
  - **Property 16: Reward Clipping**
  - **Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6**

- [x] 2.7 Integrate TradeFrequencyController and StableRewardCalculator into RealisticTradingEnv
  - ✅ Modified `RealisticTradingEnv.step()` to use TradeFrequencyController
  - ✅ Modified `RealisticTradingEnv.step()` to use StableRewardCalculator
  - ✅ Constraints are enforced before reward calculation
  - _Requirements: 2.5, 2.6, 2.7, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6_

- [ ] 2.8 Verify StateBuilder provides raw portfolio data
  - Review `src/adan_trading_bot/data_processing/state_builder.py`
  - Ensure cash and equity values are raw (non-normalized)
  - Confirm normalization is left to VecNormalize wrapper
  - _Requirements: 4.4_

- [ ] 2.9 Write property test for StateBuilder
  - **Property 20: Raw Portfolio State**
  - **Validates: Requirements 4.4**

- [ ] 2.10 Checkpoint: Verify realistic trading environment
  - Ensure all tests pass, ask the user if questions arise.

---

## Phase 3: Observation Normalization

- [x] 3.1 Create training script with VecNormalize integration
  - ✅ Created `scripts/train_parallel_agents.py`
  - ✅ Implemented environment wrapping in VecNormalize(norm_obs=True, norm_reward=True)
  - ✅ Implemented automatic vecnormalize.pkl saving after training
  - ✅ Integrated SeedManager for deterministic initialization
  - ✅ Integrated RealisticTradingEnv with all components
  - _Requirements: 4.1, 4.2, 7.1, 7.2_

- [ ] 3.2 Write property tests for VecNormalize integration
  - **Property 17: VecNormalize Wrapping**
  - **Property 18: VecNormalize Persistence**
  - **Validates: Requirements 4.1, 4.2**

- [x] 3.3 Implement normalization verification script
  - ✅ Created `scripts/backtest_2024_oos.py` with VecNormalize loading
  - ✅ Created `scripts/diagnose_pnl_anomaly.py` with VecNormalize verification
  - ✅ Loads model with vecnormalize.pkl for coherent predictions
  - _Requirements: 12.1, 12.2, 12.3_

- [ ] 3.4 Write property tests for normalization verification
  - **Property 58: Incoherent Predictions Without Normalization**
  - **Property 59: Coherent Predictions With Normalization**
  - **Property 60: Normalization Dependency Confirmation**
  - **Validates: Requirements 12.1, 12.2, 12.3**

- [ ] 3.5 Checkpoint: Verify normalization setup
  - Ensure all tests pass, ask the user if questions arise.

---

## Phase 4: Multi-Expert Fusion Pipeline

- [x] 4.1 Implement AdanFusionPipeline
  - ✅ Created `scripts/fuse_workers_to_adan.py`
  - ✅ Implemented `fuse_workers()` for expert combination via weight averaging
  - ✅ Implemented vecnormalize.pkl copying for unified model
  - ✅ Produces single ADAN model file (adan_model_final.zip)
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

- [ ] 4.2 Write property tests for AdanFusionPipeline
  - **Property 21: Individual Expert Training**
  - **Property 22: Collaborative Training Phase**
  - **Property 23: Expert Fusion**
  - **Property 24: ADAN Fine-Tuning**
  - **Property 25: Single ADAN Model Output**
  - **Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5**

- [ ] 4.3 Implement IntelligentCheckpoint callback
  - Create `src/adan_trading_bot/common/intelligent_checkpoint.py`
  - Inherit from `stable_baselines3.common.callbacks.BaseCallback`
  - Implement `_on_step()` for checkpoint evaluation
  - Implement `_evaluate_checkpoint()` assessing performance, stability, overfitting, fusion progress
  - Implement `_save_checkpoint()` saving complete fusion state
  - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6_

- [ ] 4.4 Write property tests for IntelligentCheckpoint
  - **Property 34: Checkpoint Performance Assessment**
  - **Property 35: Checkpoint Stability Assessment**
  - **Property 36: Checkpoint Overfitting Assessment**
  - **Property 37: Checkpoint Fusion Progress Assessment**
  - **Property 38: Conditional Checkpoint Saving**
  - **Validates: Requirements 8.1, 8.2, 8.3, 8.4, 8.5, 8.6**

- [ ] 4.5 Integrate IntelligentCheckpoint into train_parallel_agents.py
  - Modify `scripts/train_parallel_agents.py`
  - Add IntelligentCheckpoint callback to training loop
  - Ensure checkpoint state includes fusion metadata
  - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6_

- [ ] 4.6 Checkpoint: Verify fusion pipeline
  - Ensure all tests pass, ask the user if questions arise.

---

## Phase 5: Temporal Validation

- [ ] 5.1 Implement TemporalCrossValidation
  - Create `src/adan_trading_bot/validation/temporal_cross_validation.py`
  - Implement `create_windows()` for overlapping train/validation windows
  - Implement `validate_model(model)` for window-based validation
  - Implement metric aggregation across all windows
  - _Requirements: 6.1, 6.2, 6.3, 6.4_

- [ ] 5.2 Write property tests for TemporalCrossValidation
  - **Property 26: Temporal Window Creation**
  - **Property 27: Window-Based Retraining**
  - **Property 28: Window-Based Validation**
  - **Property 29: Metric Aggregation**
  - **Validates: Requirements 6.1, 6.2, 6.3, 6.4**

- [ ] 5.3 Implement DetailedValidationReport
  - Create `src/adan_trading_bot/validation/detailed_validation_report.py`
  - Implement `generate_report()` for comprehensive report generation
  - Implement `add_metrics(phase, metrics)` for metric collection
  - Implement `export_html(filepath)` for HTML export
  - Include performance, risk, behavior, and reproducibility metrics
  - _Requirements: 5.2, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8_

- [ ] 5.4 Write property tests for DetailedValidationReport
  - **Property 39: Reproducibility Test Seed Consistency**
  - **Property 40: Data Leakage Detection**
  - **Property 41: Environment Determinism**
  - **Property 42: Stress Test Performance**
  - **Property 43: Fee and Slippage Impact**
  - **Property 44: Trade Frequency Constraint Enforcement**
  - **Property 45: Risk Management Limit Enforcement**
  - **Property 46: Multi-Regime Performance**
  - **Validates: Requirements 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8**

- [ ] 5.5 Checkpoint: Verify temporal validation
  - Ensure all tests pass, ask the user if questions arise.

---

## Phase 6: Out-of-Sample Validation

- [x] 6.1 Implement out-of-sample backtest script
  - ✅ Created `scripts/backtest_2024_oos.py`
  - ✅ Implements backtest using 2024+ data
  - ✅ Loads model with VecNormalize for proper inference
  - ✅ Runs episodes and calculates rewards
  - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7_

- [ ] 6.2 Write property tests for out-of-sample validation
  - **Property 47: Out-of-Sample Data Selection**
  - **Property 48: Sharpe Ratio Calculation**
  - **Property 49: Total Return Calculation**
  - **Property 50: Maximum Drawdown Calculation**
  - **Property 51: Sharpe Ratio Threshold**
  - **Property 52: Positive Return Requirement**
  - **Property 53: Drawdown Limit**
  - **Validates: Requirements 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7**

- [ ] 6.3 Implement model saturation detection script
  - Create `scripts/check_model_saturation.py`
  - Load trained model
  - Sample predictions across diverse inputs
  - Analyze output distribution
  - Detect saturation (majority outputs ±1.0)
  - Flag and reject saturated models
  - _Requirements: 11.1, 11.2, 11.3, 11.4_

- [ ] 6.4 Write property tests for saturation detection
  - **Property 54: Diverse Input Sampling**
  - **Property 55: Output Distribution Analysis**
  - **Property 56: Saturation Detection**
  - **Property 57: Saturated Model Rejection**
  - **Validates: Requirements 11.1, 11.2, 11.3, 11.4**

- [ ] 6.5 Checkpoint: Verify out-of-sample validation
  - Ensure all tests pass, ask the user if questions arise.

---

## Phase 7: Comprehensive Testing Suite

- [ ] 7.1 Create unit tests for remaining components
  - Create `tests/unit/test_seed_manager.py`
  - Create `tests/unit/test_data_manager.py`
  - Create `tests/unit/test_realistic_trading_env.py`
  - Create `tests/unit/test_stable_reward_calculator.py`
  - Create `tests/unit/test_adan_fusion_pipeline.py`
  - Create `tests/unit/test_temporal_cross_validation.py`
  - Create `tests/unit/test_intelligent_checkpoint.py`
  - Create `tests/unit/test_detailed_validation_report.py`
  - _Requirements: All_

- [ ] 7.2 Create property-based tests for all properties
  - Create `tests/property_based/test_data_properties.py`
  - Create `tests/property_based/test_environment_properties.py`
  - Create `tests/property_based/test_reward_properties.py`
  - Create `tests/property_based/test_normalization_properties.py`
  - Create `tests/property_based/test_fusion_properties.py`
  - Create `tests/property_based/test_validation_properties.py`
  - Create `tests/property_based/test_reproducibility_properties.py`
  - Create `tests/property_based/test_checkpoint_properties.py`
  - Create `tests/property_based/test_validation_suite_properties.py`
  - Create `tests/property_based/test_out_of_sample_properties.py`
  - Create `tests/property_based/test_saturation_properties.py`
  - Create `tests/property_based/test_normalization_verification_properties.py`
  - _Requirements: All_

- [ ] 7.3 Create integration tests for complete pipelines
  - Create `tests/integration/test_full_pipeline.py`
  - Create `tests/integration/test_fusion_pipeline.py`
  - Create `tests/integration/test_validation_pipeline.py`
  - _Requirements: All_

- [ ] 7.4 Final Checkpoint: Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

---

## Summary

This implementation plan covers all aspects of the ADAN 2.0 reconstruction:

- **Phase 1**: Data integrity with strict temporal separation and seeding
- **Phase 2**: Realistic trading environment with market friction models ✅ (Partially Complete)
- **Phase 3**: Observation normalization with VecNormalize integration
- **Phase 4**: Multi-expert fusion pipeline with intelligent checkpointing
- **Phase 5**: Temporal cross-validation with comprehensive reporting
- **Phase 6**: Out-of-sample validation with saturation detection
- **Phase 7**: Comprehensive testing suite with unit, property-based, and integration tests

### Completed Tasks Summary:
- ✅ SeedManager for reproducibility (Phase 1.1)
- ✅ RealisticTradingEnv with market friction models (Phase 2.1)
- ✅ TradeFrequencyController with unit tests (Phase 2.3, 2.4)
- ✅ StableRewardCalculator (Phase 2.5)
- ✅ Integration of controllers into RealisticTradingEnv (Phase 2.7)
- ✅ VecNormalize integration in training script (Phase 3.1)
- ✅ Normalization verification scripts (Phase 3.3)
- ✅ AdanFusionPipeline for multi-expert training (Phase 4.1)
- ✅ Out-of-sample backtest script (Phase 6.1)
- ✅ Deadlock regression test

### Remaining Priority Tasks:
1. DataManager for strict temporal separation (Phase 1.3)
2. TemporalCrossValidation for validation (Phase 5.1)
3. DetailedValidationReport (Phase 5.3)
4. IntelligentCheckpoint callback (Phase 4.3)
5. Property-based tests for all components (Phase 7.2)
6. Model saturation detection script (Phase 6.3)

All tasks build incrementally with clear dependencies. All testing tasks are now required for comprehensive coverage. Each phase includes a checkpoint to verify progress before moving to the next phase.
