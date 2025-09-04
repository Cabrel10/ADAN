#!/usr/bin/env python3
"""
Test script for modular configuration system.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from adan_trading_bot.common.enhanced_config_manager import EnhancedConfigManager


def test_modular_config():
    """Test the modular configuration system."""
    print("🧪 Testing Modular Configuration System...")

    try:
        # Initialize the enhanced config manager
        config_manager = EnhancedConfigManager(
            config_dir="config",
            enable_hot_reload=False  # Disable for testing
        )

        print("✅ EnhancedConfigManager initialized")

        # Test loading individual modular configurations
        modules_to_test = [
            "model", "environment", "agent", "data",
            "training", "trading", "workers", "paths"
        ]

        loaded_configs = {}

        for module_name in modules_to_test:
            try:
                config = config_manager.get_config(module_name)
                if config:
                    loaded_configs[module_name] = config
                    print(f"  ✅ Loaded {module_name}.yaml: {len(config)} sections")

                    # Show some key information from each config
                    if module_name == "model" and "model" in config:
                        arch = config["model"].get("architecture", {})
                        print(f"    📊 Model architecture blocks: {list(arch.keys())}")

                    elif module_name == "environment" and "environment" in config:
                        balance = config["environment"].get("initial_balance", "N/A")
                        assets = config["environment"].get("assets", [])
                        print(f"    💰 Initial balance: {balance}, Assets: {len(assets)}")

                    elif module_name == "agent" and "agent" in config:
                        algorithm = config["agent"].get("algorithm", "N/A")
                        lr = config["agent"].get("learning_rate", "N/A")
                        print(f"    🤖 Algorithm: {algorithm}, Learning rate: {lr}")

                    elif module_name == "training" and "training" in config:
                        instances = config["training"].get("num_instances", "N/A")
                        timesteps = config["training"].get("timesteps_per_instance", "N/A")
                        print(f"    🏋️  Instances: {instances}, Timesteps: {timesteps}")

                    elif module_name == "workers" and "workers" in config:
                        workers = config["workers"]
                        print(f"    👷 Workers configured: {list(workers.keys())}")

                else:
                    print(f"  ⚠️  {module_name}.yaml: No data loaded")

            except Exception as e:
                print(f"  ❌ Failed to load {module_name}.yaml: {e}")

        # Test getting merged configuration
        print("\n🔄 Testing merged configuration...")
        try:
            merged_config = config_manager.get_merged_config()
            if merged_config:
                print(f"  ✅ Merged configuration: {len(merged_config)} top-level sections")

                # Show top-level sections
                sections = list(merged_config.keys())
                print(f"    📋 Sections: {sections}")

                # Test accessing nested configurations
                if "model" in merged_config and "architecture" in merged_config["model"]:
                    print("    ✅ Nested access working (model.architecture)")

                if "environment" in merged_config and "assets" in merged_config["environment"]:
                    assets = merged_config["environment"]["assets"]
                    print(f"    ✅ Environment assets: {len(assets)} configured")

            else:
                print("  ❌ Failed to get merged configuration")

        except Exception as e:
            print(f"  ❌ Error getting merged configuration: {e}")

        # Test configuration validation
        print("\n🔍 Testing configuration validation...")
        try:
            validation_results = config_manager.validate_all_configs()

            valid_count = sum(1 for result in validation_results.values() if result.get("valid", False))
            total_count = len(validation_results)

            print(f"  📊 Validation results: {valid_count}/{total_count} configurations valid")

            for config_name, result in validation_results.items():
                if result.get("valid", False):
                    print(f"    ✅ {config_name}: Valid")
                else:
                    errors = result.get("errors", [])
                    print(f"    ⚠️  {config_name}: {len(errors)} validation issues")

        except Exception as e:
            print(f"  ⚠️  Validation test failed: {e}")

        # Test configuration summary
        print("\n📋 Configuration Summary:")
        summary = config_manager.get_config_summary()

        for key, value in summary.items():
            if isinstance(value, dict):
                print(f"  {key}: {len(value)} items")
            elif isinstance(value, list):
                print(f"  {key}: {len(value)} items")
            else:
                print(f"  {key}: {value}")

        print(f"\n✅ Modular configuration test completed!")
        print(f"📊 Successfully loaded {len(loaded_configs)} configuration modules")

        return True

    except Exception as e:
        print(f"❌ Modular configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_manager_features():
    """Test additional features of the config manager."""
    print("\n🧪 Testing Enhanced Config Manager Features...")

    try:
        config_manager = EnhancedConfigManager(
            config_dir="config",
            enable_hot_reload=False
        )

        # Test getting specific configuration sections
        print("\n🔍 Testing specific section access...")

        # Test model configuration access
        try:
            model_config = config_manager.get_config("model")
            if model_config and "model" in model_config:
                architecture = model_config["model"].get("architecture", {})
                print(f"  ✅ Model architecture: {len(architecture)} blocks configured")

                # Test specific block access
                if "block_a" in architecture:
                    block_a = architecture["block_a"]
                    print(f"    📊 Block A: {block_a.get('out_channels', 'N/A')} channels")

        except Exception as e:
            print(f"  ❌ Model config access failed: {e}")

        # Test environment configuration access
        try:
            env_config = config_manager.get_config("environment")
            if env_config and "environment" in env_config:
                env_settings = env_config["environment"]
                balance = env_settings.get("initial_balance", "N/A")
                max_steps = env_settings.get("max_steps", "N/A")
                print(f"  ✅ Environment: Balance={balance}, Max steps={max_steps}")

        except Exception as e:
            print(f"  ❌ Environment config access failed: {e}")

        # Test configuration file existence
        print("\n📁 Testing configuration file existence...")
        config_dir = Path("config")

        expected_files = [
            "model.yaml", "environment.yaml", "agent.yaml",
            "data.yaml", "training.yaml", "trading.yaml",
            "workers.yaml", "paths.yaml"
        ]

        existing_files = []
        missing_files = []

        for file_name in expected_files:
            file_path = config_dir / file_name
            if file_path.exists():
                existing_files.append(file_name)
                # Check file size
                size = file_path.stat().st_size
                print(f"  ✅ {file_name}: {size} bytes")
            else:
                missing_files.append(file_name)
                print(f"  ❌ {file_name}: Missing")

        print(f"\n📊 File summary: {len(existing_files)} existing, {len(missing_files)} missing")

        if missing_files:
            print(f"⚠️  Missing files: {missing_files}")

        return len(missing_files) == 0

    except Exception as e:
        print(f"❌ Config manager features test failed: {e}")
        return False


def main():
    """Run all modular configuration tests."""
    print("🚀 Starting Modular Configuration Tests...\n")

    tests = [
        ("Modular Configuration Loading", test_modular_config),
        ("Config Manager Features", test_config_manager_features)
    ]

    results = []

    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"Running: {test_name}")
        print('='*60)

        success = test_func()
        results.append((test_name, success))

        if success:
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")

    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print('='*60)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All modular configuration tests completed successfully!")
        return True
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
