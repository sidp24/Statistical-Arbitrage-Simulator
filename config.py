"""
Configuration settings for Statistical Arbitrage Simulator
All sensitive values loaded from environment variables
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Data settings
TICKERS = os.getenv('TICKERS', 'AAPL,MSFT,GOOGL,AMZN,META,JPM,V,UNH,NVDA,HD').split(',')
START_DATE = os.getenv('START_DATE', '2022-01-01')
END_DATE = os.getenv('END_DATE', '2024-01-01')

# Cointegration
P_VALUE_THRESHOLD = float(os.getenv('P_VALUE_THRESHOLD', '0.05'))

# Z-score calculation
ZSCORE_WINDOW = int(os.getenv('ZSCORE_WINDOW', '30'))

# Trading strategy
ENTRY_Z = float(os.getenv('ENTRY_Z', '1.0'))
EXIT_Z = float(os.getenv('EXIT_Z', '0.5'))
INITIAL_CAPITAL = float(os.getenv('INITIAL_CAPITAL', '10000'))

# File paths
PRICE_DATA_PATH = os.getenv('PRICE_DATA_PATH', 'data/price_data.csv')
ZSCORE_OUTPUT_PATH = os.getenv('ZSCORE_OUTPUT_PATH', 'data/zscore_signals.csv')

# API Keys (loaded from environment - NEVER commit actual keys)
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')

# Database settings
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///data/arbitrage.db')

# Authentication settings
AUTH_ENABLED = os.getenv('AUTH_ENABLED', 'false').lower() == 'true'
SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')

# Email/Notification settings
SMTP_HOST = os.getenv('SMTP_HOST', '')
SMTP_PORT = int(os.getenv('SMTP_PORT', '587'))
SMTP_USER = os.getenv('SMTP_USER', '')
SMTP_PASSWORD = os.getenv('SMTP_PASSWORD', '')
NOTIFICATION_EMAIL = os.getenv('NOTIFICATION_EMAIL', '')

# Application settings
DEBUG = os.getenv('DEBUG', 'false').lower() == 'true'
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
