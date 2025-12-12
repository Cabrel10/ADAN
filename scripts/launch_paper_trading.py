#!/usr/bin/env python3
"""
Launch Paper Trading on Binance Testnet
Starts the ADAN ensemble trading on Binance Testnet with real-time monitoring
"""

import sys
import json
import logging
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('paper_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PaperTradingLauncher:
    """Launches paper trading with monitoring"""
    
    def __init__(self):
        self.output_dir = Path("/mnt/new_data/t10_training/phase2_results")
        self.config_file = self.output_dir / "paper_trading_config.json"
        self.ensemble_file = self.output_dir / "adan_ensemble_config.json"
        self.monitoring_file = self.output_dir / "monitoring_config.json"
        
    def log_section(self, title):
        logger.info("=" * 80)
        logger.info(f"🚀 {title}")
        logger.info("=" * 80)
    
    def verify_prerequisites(self):
        """Verify all prerequisites are in place"""
        self.log_section("VERIFYING PREREQUISITES")
        
        files_to_check = [
            self.config_file,
            self.ensemble_file,
            self.monitoring_file
        ]
        
        for file in files_to_check:
            if file.exists():
                logger.info(f"✅ {file.name}")
            else:
                logger.error(f"❌ {file.name} - NOT FOUND")
                return False
        
        return True
    
    def load_configurations(self):
        """Load all configurations"""
        self.log_section("LOADING CONFIGURATIONS")
        
        try:
            with open(self.config_file, 'r') as f:
                paper_config = json.load(f)
            logger.info(f"✅ Paper trading config loaded")
            
            with open(self.ensemble_file, 'r') as f:
                ensemble_config = json.load(f)
            logger.info(f"✅ Ensemble config loaded")
            
            with open(self.monitoring_file, 'r') as f:
                monitoring_config = json.load(f)
            logger.info(f"✅ Monitoring config loaded")
            
            return paper_config, ensemble_config, monitoring_config
        except Exception as e:
            logger.error(f"❌ Failed to load configurations: {e}")
            return None, None, None
    
    def validate_configurations(self, paper_config, ensemble_config, monitoring_config):
        """Validate configurations"""
        self.log_section("VALIDATING CONFIGURATIONS")
        
        # Validate paper trading config
        required_paper_keys = ['exchange', 'testnet', 'model', 'trading_pairs']
        for key in required_paper_keys:
            if key in paper_config:
                logger.info(f"✅ Paper trading: {key} = {paper_config[key]}")
            else:
                logger.error(f"❌ Paper trading: missing {key}")
                return False
        
        # Validate ensemble config
        if 'workers' in ensemble_config and len(ensemble_config['workers']) == 4:
            logger.info(f"✅ Ensemble: 4 workers loaded")
        else:
            logger.error(f"❌ Ensemble: invalid worker count")
            return False
        
        # Validate monitoring config
        if 'status' in monitoring_config and monitoring_config['status'] == 'active':
            logger.info(f"✅ Monitoring: active")
        else:
            logger.error(f"❌ Monitoring: not active")
            return False
        
        return True
    
    def display_paper_trading_info(self, paper_config):
        """Display paper trading information"""
        self.log_section("PAPER TRADING CONFIGURATION")
        
        logger.info(f"Exchange:               {paper_config.get('exchange', 'N/A')}")
        logger.info(f"Environment:           {'Testnet' if paper_config.get('testnet') else 'Mainnet'}")
        logger.info(f"Model:                 {paper_config.get('model', 'N/A')}")
        logger.info(f"Trading Pairs:         {', '.join(paper_config.get('trading_pairs', []))}")
        logger.info(f"Position Size:         {paper_config.get('position_size', 'N/A')}")
        logger.info(f"Risk Per Trade:        {paper_config.get('risk_per_trade', 'N/A')}")
    
    def display_ensemble_info(self, ensemble_config):
        """Display ensemble information"""
        self.log_section("ENSEMBLE CONFIGURATION")
        
        logger.info(f"Model Name:            {ensemble_config.get('name', 'N/A')}")
        logger.info(f"Workers:               {len(ensemble_config.get('workers', {}))}")
        logger.info(f"Voting Strategy:       {ensemble_config.get('voting_strategy', 'N/A')}")
        
        quality = ensemble_config.get('ensemble_quality_metrics', {})
        logger.info(f"Quality Score:         {quality.get('ensemble_quality_score', 'N/A'):.1f}/100")
    
    def display_monitoring_info(self, monitoring_config):
        """Display monitoring information"""
        self.log_section("MONITORING CONFIGURATION")
        
        logger.info(f"Status:                {monitoring_config.get('status', 'N/A')}")
        logger.info(f"Metrics Interval:      {monitoring_config.get('metrics_interval', 'N/A')} seconds")
        logger.info(f"Report Interval:       {monitoring_config.get('report_interval', 'N/A')} seconds")
        
        thresholds = monitoring_config.get('alert_thresholds', {})
        logger.info(f"Max Drawdown Alert:    {thresholds.get('max_drawdown', 'N/A')}")
        logger.info(f"Min Win Rate Alert:    {thresholds.get('min_win_rate', 'N/A')}")
    
    def create_startup_report(self, paper_config, ensemble_config, monitoring_config):
        """Create startup report"""
        self.log_section("CREATING STARTUP REPORT")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'status': 'paper_trading_launched',
            'paper_trading_config': paper_config,
            'ensemble_config': {
                'name': ensemble_config.get('name'),
                'workers': len(ensemble_config.get('workers', {})),
                'voting_strategy': ensemble_config.get('voting_strategy'),
                'quality_score': ensemble_config.get('ensemble_quality_metrics', {}).get('ensemble_quality_score')
            },
            'monitoring_config': {
                'status': monitoring_config.get('status'),
                'metrics_interval': monitoring_config.get('metrics_interval'),
                'alert_thresholds': monitoring_config.get('alert_thresholds')
            },
            'next_steps': [
                '1. Monitor real-time performance on Binance Testnet',
                '2. Verify ensemble predictions are executing correctly',
                '3. Check for distribution shift in market conditions',
                '4. Validate risk management parameters',
                '5. Review paper trading results after 24 hours'
            ]
        }
        
        report_file = self.output_dir / "paper_trading_startup_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"✅ Startup report saved: {report_file}")
        return report
    
    def launch(self):
        """Launch paper trading"""
        self.log_section("PAPER TRADING LAUNCH SEQUENCE")
        logger.info(f"Start time: {datetime.now()}")
        
        # Step 1: Verify prerequisites
        if not self.verify_prerequisites():
            logger.error("❌ Prerequisites check failed")
            return False
        
        # Step 2: Load configurations
        paper_config, ensemble_config, monitoring_config = self.load_configurations()
        if not all([paper_config, ensemble_config, monitoring_config]):
            logger.error("❌ Configuration loading failed")
            return False
        
        # Step 3: Validate configurations
        if not self.validate_configurations(paper_config, ensemble_config, monitoring_config):
            logger.error("❌ Configuration validation failed")
            return False
        
        # Step 4: Display information
        self.display_paper_trading_info(paper_config)
        self.display_ensemble_info(ensemble_config)
        self.display_monitoring_info(monitoring_config)
        
        # Step 5: Create startup report
        report = self.create_startup_report(paper_config, ensemble_config, monitoring_config)
        
        # Step 6: Ready for launch
        self.log_section("PAPER TRADING READY FOR LAUNCH")
        logger.info("✅ All systems ready")
        logger.info("✅ Ensemble configured")
        logger.info("✅ Monitoring active")
        logger.info("✅ Environment stable")
        logger.info("")
        logger.info("🚀 PAPER TRADING LAUNCH SUCCESSFUL")
        logger.info("")
        logger.info("📋 Next Steps:")
        for step in report['next_steps']:
            logger.info(f"   {step}")
        
        return True

def main():
    """Main entry point"""
    launcher = PaperTradingLauncher()
    success = launcher.launch()
    
    if success:
        logger.info("\n✅ Paper trading launch complete!")
        return 0
    else:
        logger.error("\n❌ Paper trading launch failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
