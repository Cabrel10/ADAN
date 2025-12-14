#!/bin/bash
# Launch ADAN Dashboard with Paper Trading Monitor
# This script starts both the monitor and dashboard for real-time monitoring

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
MONITOR_REFRESH=300  # 5 minutes (300 seconds)
DASHBOARD_REFRESH=60 # 1 minute (60 seconds)
STATE_FILE="/mnt/new_data/t10_training/phase2_results/paper_trading_state.json"

echo -e "${CYAN}🎯 ADAN Dashboard + Monitor Launcher${NC}"
echo -e "${CYAN}=====================================${NC}\n"

# Check if state file directory exists
STATE_DIR=$(dirname "$STATE_FILE")
if [ ! -d "$STATE_DIR" ]; then
    echo -e "${YELLOW}⚠️  State directory does not exist: $STATE_DIR${NC}"
    echo -e "${YELLOW}Creating directory...${NC}"
    mkdir -p "$STATE_DIR"
fi

# Parse arguments
MONITOR_ONLY=false
DASHBOARD_ONLY=false
MOCK_DATA=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --monitor-only)
            MONITOR_ONLY=true
            shift
            ;;
        --dashboard-only)
            DASHBOARD_ONLY=true
            shift
            ;;
        --mock)
            MOCK_DATA=true
            shift
            ;;
        --monitor-refresh)
            MONITOR_REFRESH="$2"
            shift 2
            ;;
        --dashboard-refresh)
            DASHBOARD_REFRESH="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --monitor-only          Start only the paper trading monitor"
            echo "  --dashboard-only        Start only the dashboard"
            echo "  --mock                  Use mock data for dashboard (testing)"
            echo "  --monitor-refresh SECS  Monitor refresh interval (default: 300)"
            echo "  --dashboard-refresh SECS Dashboard refresh interval (default: 60)"
            echo "  --help                  Show this help message"
            echo ""
            echo "Examples:"
            echo "  # Start both monitor and dashboard"
            echo "  $0"
            echo ""
            echo "  # Start only the monitor"
            echo "  $0 --monitor-only"
            echo ""
            echo "  # Start dashboard with mock data"
            echo "  $0 --dashboard-only --mock"
            exit 0
            ;;
        *)
            echo -e "${RED}❌ Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Function to start monitor
start_monitor() {
    echo -e "${GREEN}✅ Starting Paper Trading Monitor...${NC}"
    echo -e "${CYAN}   Refresh interval: ${MONITOR_REFRESH}s${NC}"
    echo -e "${CYAN}   State file: $STATE_FILE${NC}\n"
    
    cd "$PROJECT_ROOT"
    python scripts/paper_trading_monitor.py &
    MONITOR_PID=$!
    echo -e "${GREEN}✅ Monitor started (PID: $MONITOR_PID)${NC}\n"
}

# Function to start dashboard
start_dashboard() {
    echo -e "${GREEN}✅ Starting ADAN Dashboard...${NC}"
    echo -e "${CYAN}   Refresh interval: ${DASHBOARD_REFRESH}s${NC}"
    
    if [ "$MOCK_DATA" = true ]; then
        echo -e "${YELLOW}   Data source: Mock Data (Testing)${NC}\n"
        cd "$PROJECT_ROOT"
        python scripts/adan_btc_dashboard.py --mock --refresh "$DASHBOARD_REFRESH"
    else
        echo -e "${CYAN}   Data source: Real Data (Live)${NC}\n"
        cd "$PROJECT_ROOT"
        python scripts/adan_btc_dashboard.py --refresh "$DASHBOARD_REFRESH"
    fi
}

# Main logic
if [ "$MONITOR_ONLY" = true ]; then
    start_monitor
    wait $MONITOR_PID
elif [ "$DASHBOARD_ONLY" = true ]; then
    start_dashboard
else
    # Start both
    echo -e "${CYAN}Starting both monitor and dashboard...${NC}\n"
    
    start_monitor
    
    # Wait a bit for monitor to initialize
    echo -e "${YELLOW}⏳ Waiting for monitor to initialize...${NC}"
    sleep 3
    
    # Start dashboard
    start_dashboard
    
    # If dashboard exits, kill monitor
    kill $MONITOR_PID 2>/dev/null || true
fi

echo -e "\n${YELLOW}👋 Shutdown complete${NC}"
