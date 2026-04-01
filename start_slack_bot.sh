#!/bin/bash
# Start the Retirement Simulator Slack bot
# Usage: ./start_slack_bot.sh

# Load tokens from .env file if it exists
ENV_FILE="$(dirname "$0")/.env"
if [ -f "$ENV_FILE" ]; then
    export $(grep -v '^#' "$ENV_FILE" | xargs)
fi

# Check required vars
if [ -z "$SLACK_BOT_TOKEN" ] || [ -z "$SLACK_APP_TOKEN" ] || [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "Missing required environment variables."
    echo "Create a .env file at: $ENV_FILE"
    echo ""
    echo "Required contents:"
    echo "  SLACK_BOT_TOKEN=xoxb-your-token"
    echo "  SLACK_APP_TOKEN=xapp-your-token"
    echo "  ANTHROPIC_API_KEY=sk-ant-your-key"
    echo ""
    echo "Optional:"
    echo "  RETIREMENT_EMAIL=your@email.com"
    echo "  SMTP_PASSWORD=your-gmail-app-password"
    exit 1
fi

cd "$(dirname "$0")"
echo "Starting Retirement Simulator Slack bot..."
mcproj/bin/python3 slack_retirement_bot.py
