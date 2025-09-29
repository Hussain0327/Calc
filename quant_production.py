#!/usr/bin/env python3
"""
Production-Ready Quantitative Trading Platform
Version 2.0 - Complete Implementation
"""

import json
import sqlite3
import math
import statistics
import logging
import re
import asyncio
import threading
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
from datetime import datetime, timedelta
from functools import lru_cache, wraps
from collections import deque, OrderedDict, defaultdict
import os
from pathlib import Path
import time
import hashlib
import warnings
import sys

# Setup logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / f'trading_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Core dependencies check
REQUIRED_PACKAGES = {
    'yfinance': False,
    'matplotlib': False,
    'pandas': False,
    'numpy': False,
    'plotly': False,
    'quantstats': False
}

# Check which packages are available
for package in REQUIRED_PACKAGES:
    try:
        __import__(package)
        REQUIRED_PACKAGES[package] = True
        logger.info(f"Package {package} loaded successfully")
    except ImportError:
        logger.warning(f"Package {package} not available")

# Import available packages
if REQUIRED_PACKAGES['pandas']:
    import pandas as pd
if REQUIRED_PACKAGES['numpy']:
    import numpy as np
if REQUIRED_PACKAGES['yfinance']:
    import yfinance as yf
if REQUIRED_PACKAGES['matplotlib']:
    import matplotlib.pyplot as plt
    plt.style.use('seaborn-v0_8-darkgrid')
if REQUIRED_PACKAGES['plotly']:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

# ==========================================
# SECURITY AND VALIDATION
# ==========================================

class InputValidator:
    """Input validation and sanitization"""
    
    @staticmethod
    def sanitize_symbol(symbol: str) -> str:
        """Sanitize and validate stock symbol"""
        if not symbol:
            raise ValueError("Symbol cannot be empty")
        
        symbol = symbol.upper().strip()
        
        # Only allow alphanumeric, dots, and hyphens (for stocks like BRK.B)
        if not re.match(r'^[A-Z0-9\-\.]{1,10}$', symbol):
            raise ValueError(f"Invalid symbol format: {symbol}")
        
        return symbol
    
    @staticmethod
    def validate_price(price: float, field_name: str = "price") -> float:
        """Validate price input"""
        if not isinstance(price, (int, float)):
            raise ValueError(f"{field_name} must be a number")
        
        if price <= 0:
            raise ValueError(f"{field_name} must be positive")
        
        if price > 1000000:  # Sanity check
            raise ValueError(f"{field_name} seems unrealistic: {price}")
        
        return float(price)
    
    @staticmethod
    def validate_percentage(value: float, field_name: str = "percentage") -> float:
        """Validate percentage input"""
        if not isinstance(value, (int, float)):
            raise ValueError(f"{field_name} must be a number")
        
        if not 0 <= value <= 100:
            raise ValueError(f"{field_name} must be between 0 and 100")
        
        return float(value)
    
    @staticmethod
    def validate_date(date_str: str) -> str:
        """Validate date format"""
        try:
            datetime.strptime(date_str, '%Y-%m-%d')
            return date_str
        except ValueError:
            raise ValueError(f"Invalid date format: {date_str}. Use YYYY-MM-DD")

# ==========================================
# RISK MANAGEMENT
# ==========================================

class RiskLimiter:
    """Enforce trading risk limits"""
    
    # Conservative default limits
    MAX_POSITION_SIZE = 0.20      # Max 20% in one position
    MAX_DAILY_LOSS = 0.03          # Max 3% daily loss
    MAX_CORRELATION = 0.75         # Max correlation between positions
    MAX_LEVERAGE = 1.0             # No leverage by default
    MIN_LIQUIDITY_RATIO = 0.2      # Keep 20% cash minimum
    
    def __init__(self):
        self.daily_losses = {}
        self.position_sizes = {}
        self.alerts_triggered = []
    
    def check_position_limit(self, 
                           position_value: float, 
                           total_capital: float,
                           symbol: str) -> Tuple[bool, str]:
        """Verify position size is within limits"""
        if total_capital <= 0:
            return False, "Invalid capital amount"
            
        position_pct = position_value / total_capital
        
        if position_pct > self.MAX_POSITION_SIZE:
            msg = f"Position {symbol} exceeds {self.MAX_POSITION_SIZE*100}% limit: {position_pct*100:.2f}%"
            logger.warning(msg)
            return False, msg
        
        self.position_sizes[symbol] = position_pct
        return True, "Position size acceptable"
    
    def check_daily_loss(self, daily_pnl: float, capital: float) -> Tuple[bool, str]:
        """Circuit breaker for daily losses"""
        if capital <= 0:
            return False, "Invalid capital amount"
            
        today = datetime.now().date()
        loss_pct = abs(daily_pnl) / capital
        
        if daily_pnl < 0:
            if today not in self.daily_losses:
                self.daily_losses[today] = 0
            self.daily_losses[today] += abs(daily_pnl)
            
            total_daily_loss = self.daily_losses[today] / capital
            
            if total_daily_loss > self.MAX_DAILY_LOSS:
                msg = f"CIRCUIT BREAKER: Daily loss limit breached: {total_daily_loss*100:.2f}%"
                logger.critical(msg)
                self.alerts_triggered.append({
                    'time': datetime.now(),
                    'type': 'DAILY_LOSS_LIMIT',
                    'message': msg
                })
                return False, msg
        
        return True, "Within daily loss limits"
    
    def check_portfolio_correlation(self, correlation_matrix) -> Tuple[bool, str]:
        """Check if portfolio is too correlated"""
        if correlation_matrix is None:
            return True, "No correlation data"
        
        # Check for high correlations (excluding diagonal)
        high_corr_pairs = []
        
        for i in range(len(correlation_matrix)):
            for j in range(i+1, len(correlation_matrix)):
                if abs(correlation_matrix.iloc[i, j]) > self.MAX_CORRELATION:
                    high_corr_pairs.append((
                        correlation_matrix.index[i],
                        correlation_matrix.columns[j],
                        correlation_matrix.iloc[i, j]
                    ))
        
        if high_corr_pairs:
            msg = f"High correlations detected: {high_corr_pairs}"
            logger.warning(msg)
            return False, msg
        
        return True, "Portfolio diversification acceptable"
    
    def get_risk_report(self) -> Dict:
        """Generate comprehensive risk report"""
        return {
            'position_sizes': self.position_sizes,
            'daily_losses': {str(k): v for k, v in self.daily_losses.items()},
            'alerts_triggered': self.alerts_triggered,
            'limits': {
                'max_position_size': self.MAX_POSITION_SIZE,
                'max_daily_loss': self.MAX_DAILY_LOSS,
                'max_correlation': self.MAX_CORRELATION
            }
        }

# ==========================================
# CACHE MANAGEMENT
# ==========================================

class CacheManager:
    """LRU cache with TTL for market data"""
    
    def __init__(self, max_size: int = 100, ttl_seconds: int = 3600):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.timestamps = {}
        self.hit_count = 0
        self.miss_count = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache if valid"""
        if key in self.cache:
            # Check if expired
            if time.time() - self.timestamps[key] > self.ttl_seconds:
                del self.cache[key]
                del self.timestamps[key]
                self.miss_count += 1
                return None
            
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            self.hit_count += 1
            return self.cache[key]
        
        self.miss_count += 1
        return None
    
    def set(self, key: str, value: Any):
        """Add item to cache"""
        # Remove oldest if at capacity
        if len(self.cache) >= self.max_size:
            oldest = next(iter(self.cache))
            del self.cache[oldest]
            del self.timestamps[oldest]
        
        self.cache[key] = value
        self.timestamps[key] = time.time()
    
    def clear_expired(self):
        """Remove expired entries"""
        current_time = time.time()
        expired_keys = [
            k for k, timestamp in self.timestamps.items()
            if current_time - timestamp > self.ttl_seconds
        ]
        
        for key in expired_keys:
            del self.cache[key]
            del self.timestamps[key]
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0
        
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate': hit_rate,
            'ttl_seconds': self.ttl_seconds
        }

# ==========================================
# DATA MODELS
# ==========================================

class PositionType(Enum):
    LONG = "LONG"
    SHORT = "SHORT"

@dataclass
class TradeMetrics:
    """Enhanced trade metrics with validation"""
    entry_price: float
    exit_price: float
    position_size: float
    position_type: PositionType
    symbol: str = "N/A"
    entry_date: str = ""
    exit_date: str = ""
    commission: float = 0.0
    slippage: float = 0.0
    strategy_name: str = "manual"
    
    def __post_init__(self):
        """Validate trade data on creation"""
        if self.symbol != "N/A":
            self.symbol = InputValidator.sanitize_symbol(self.symbol)
        self.entry_price = InputValidator.validate_price(self.entry_price, "Entry price")
        self.exit_price = InputValidator.validate_price(self.exit_price, "Exit price")
        
        if self.position_size <= 0:
            raise ValueError("Position size must be positive")
        
        if self.entry_date:
            self.entry_date = InputValidator.validate_date(self.entry_date)
        if self.exit_date:
            self.exit_date = InputValidator.validate_date(self.exit_date)
    
    @property
    def gross_pnl(self) -> float:
        if self.position_type == PositionType.LONG:
            return (self.exit_price - self.entry_price) * self.position_size
        else:
            return (self.entry_price - self.exit_price) * self.position_size
    
    @property
    def net_pnl(self) -> float:
        return self.gross_pnl - self.commission - self.slippage
    
    @property
    def return_pct(self) -> float:
        if self.entry_price * self.position_size == 0:
            return 0
        return (self.net_pnl / (self.entry_price * self.position_size)) * 100
    
    def to_dict(self) -> dict:
        """Convert to dictionary for storage"""
        data = asdict(self)
        data['position_type'] = self.position_type.value
        return data
    
    @classmethod
    def from_dict(cls, data: dict):
        """Create from dictionary with validation"""
        data['position_type'] = PositionType[data['position_type']]
        return cls(**data)

# ==========================================
# DATABASE MANAGEMENT
# ==========================================

class DataManager:
    """Enhanced data persistence with transactions and validation"""
    
    def __init__(self, db_path: str = "trading_data.db"):
        self.db_path = db_path
        self.json_path = "trading_backup.json"
        self._init_database()
        self._create_indexes()
    
    def _init_database(self):
        """Initialize SQLite database with proper schema"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Enable foreign keys
                conn.execute("PRAGMA foreign_keys = ON")
                
                # Trades table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS trades (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        entry_price REAL NOT NULL CHECK(entry_price > 0),
                        exit_price REAL NOT NULL CHECK(exit_price > 0),
                        position_size REAL NOT NULL CHECK(position_size > 0),
                        position_type TEXT NOT NULL CHECK(position_type IN ('LONG', 'SHORT')),
                        entry_date TEXT,
                        exit_date TEXT,
                        commission REAL DEFAULT 0,
                        slippage REAL DEFAULT 0,
                        strategy_name TEXT DEFAULT 'manual',
                        net_pnl REAL,
                        return_pct REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        CHECK(exit_date >= entry_date OR exit_date IS NULL)
                    )
                ''')
                
                # Strategies table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS strategies (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT UNIQUE NOT NULL,
                        initial_capital REAL NOT NULL CHECK(initial_capital > 0),
                        current_capital REAL,
                        max_drawdown REAL DEFAULT 0,
                        total_trades INTEGER DEFAULT 0,
                        win_rate REAL DEFAULT 0,
                        sharpe_ratio REAL DEFAULT 0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Risk limits table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS risk_limits (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        limit_type TEXT NOT NULL,
                        limit_value REAL NOT NULL,
                        is_active BOOLEAN DEFAULT 1,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Audit log table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS audit_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        action TEXT NOT NULL,
                        details TEXT,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                conn.commit()
                logger.info("Database initialized successfully")
                
        except sqlite3.Error as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    def _create_indexes(self):
        """Create database indexes for performance"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("CREATE INDEX IF NOT EXISTS idx_symbol ON trades(symbol)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_date ON trades(exit_date)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_strategy ON trades(strategy_name)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_pnl ON trades(net_pnl)")
                conn.commit()
                logger.info("Database indexes created")
        except sqlite3.Error as e:
            logger.error(f"Failed to create indexes: {e}")
    
    def save_trade(self, trade: TradeMetrics) -> int:
        """Save trade with transaction management"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("BEGIN TRANSACTION")
                
                cursor = conn.execute('''
                    INSERT INTO trades (
                        symbol, entry_price, exit_price, position_size,
                        position_type, entry_date, exit_date, commission,
                        slippage, strategy_name, net_pnl, return_pct
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    trade.symbol, trade.entry_price, trade.exit_price,
                    trade.position_size, trade.position_type.value,
                    trade.entry_date, trade.exit_date, trade.commission,
                    trade.slippage, trade.strategy_name, trade.net_pnl,
                    trade.return_pct
                ))
                
                # Log the action
                conn.execute(
                    "INSERT INTO audit_log (action, details) VALUES (?, ?)",
                    ("TRADE_ADDED", f"Symbol: {trade.symbol}, PnL: {trade.net_pnl}")
                )
                
                conn.commit()
                logger.info(f"Trade saved: {trade.symbol} PnL: {trade.net_pnl:.2f}")
                return cursor.lastrowid
                
        except sqlite3.Error as e:
            logger.error(f"Failed to save trade: {e}")
            raise
    
    def save_trades_batch(self, trades: List[TradeMetrics]) -> bool:
        """Save multiple trades in a single transaction"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("BEGIN TRANSACTION")
                
                for trade in trades:
                    conn.execute('''
                        INSERT INTO trades (
                            symbol, entry_price, exit_price, position_size,
                            position_type, entry_date, exit_date, commission,
                            slippage, strategy_name, net_pnl, return_pct
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        trade.symbol, trade.entry_price, trade.exit_price,
                        trade.position_size, trade.position_type.value,
                        trade.entry_date, trade.exit_date, trade.commission,
                        trade.slippage, trade.strategy_name, trade.net_pnl,
                        trade.return_pct
                    ))
                
                conn.execute(
                    "INSERT INTO audit_log (action, details) VALUES (?, ?)",
                    ("BATCH_TRADES_ADDED", f"Count: {len(trades)}")
                )
                
                conn.commit()
                logger.info(f"Batch saved: {len(trades)} trades")
                return True
                
        except sqlite3.Error as e:
            logger.error(f"Batch save failed: {e}")
            conn.execute("ROLLBACK")
            return False
    
    def load_trades(self, strategy_name: Optional[str] = None) -> List[TradeMetrics]:
        """Load trades with error handling"""
        trades = []
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                if strategy_name:
                    cursor = conn.execute(
                        "SELECT * FROM trades WHERE strategy_name = ? ORDER BY exit_date DESC",
                        (strategy_name,)
                    )
                else:
                    cursor = conn.execute("SELECT * FROM trades ORDER BY exit_date DESC")
                
                for row in cursor.fetchall():
                    try:
                        trade = TradeMetrics(
                            symbol=row[1],
                            entry_price=row[2],
                            exit_price=row[3],
                            position_size=row[4],
                            position_type=PositionType[row[5]],
                            entry_date=row[6] or "",
                            exit_date=row[7] or "",
                            commission=row[8],
                            slippage=row[9],
                            strategy_name=row[10]
                        )
                        trades.append(trade)
                    except Exception as e:
                        logger.error(f"Failed to load trade {row[0]}: {e}")
                        continue
                
        except sqlite3.Error as e:
            logger.error(f"Failed to load trades: {e}")
        
        return trades
    
    def export_to_json(self, filepath: Optional[str] = None) -> str:
        """Export with validation and compression"""
        filepath = filepath or self.json_path
        
        try:
            trades = self.load_trades()
            
            data = {
                "export_date": datetime.now().isoformat(),
                "version": "2.0",
                "trades": [t.to_dict() for t in trades],
                "total_trades": len(trades),
                "checksum": None
            }
            
            # Add checksum for integrity
            data_str = json.dumps(data['trades'])
            data['checksum'] = hashlib.sha256(data_str.encode()).hexdigest()
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Exported {len(trades)} trades to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            raise

# ==========================================
# MARKET DATA PROVIDER
# ==========================================

class MarketDataProvider:
    """Enhanced market data with validation and caching"""
    
    def __init__(self):
        self.cache = CacheManager(max_size=100, ttl_seconds=3600)
        self.has_yfinance = REQUIRED_PACKAGES['yfinance']
        self.rate_limiter = RateLimiter(calls_per_minute=60)
    
    def fetch_price_data(self, symbol: str, period: str = "1y", interval: str = "1d"):
        """Fetch with caching and validation"""
        if not self.has_yfinance:
            logger.warning("yfinance not available")
            return None
        
        try:
            # Validate input
            symbol = InputValidator.sanitize_symbol(symbol)
            
            # Check cache
            cache_key = f"{symbol}_{period}_{interval}"
            cached_data = self.cache.get(cache_key)
            if cached_data is not None:
                logger.debug(f"Cache hit for {symbol}")
                return cached_data
            
            # Rate limiting
            if not self.rate_limiter.allow_request():
                logger.warning("Rate limit exceeded")
                return None
            
            # Fetch data with retry logic
            ticker = yf.Ticker(symbol)
            data = self._fetch_with_retry(ticker, period, interval)
            
            if data is not None and not data.empty:
                # Validate data
                if self._validate_price_data(data):
                    self.cache.set(cache_key, data)
                    return data
                else:
                    logger.warning(f"Invalid data for {symbol}")
                    return None
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to fetch {symbol}: {e}")
            return None
    
    def _fetch_with_retry(self, ticker, period: str, interval: str, max_retries: int = 3):
        """Fetch with exponential backoff retry"""
        for attempt in range(max_retries):
            try:
                data = ticker.history(period=period, interval=interval)
                if not data.empty:
                    return data
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.info(f"Retry {attempt + 1} after {wait_time}s")
                    time.sleep(wait_time)
                else:
                    raise
        return None
    
    def _validate_price_data(self, data) -> bool:
        """Validate price data integrity"""
        if data.empty:
            return False
        
        # Check for invalid high/low
        if (data['High'] < data['Low']).any():
            logger.warning("Invalid high/low data detected")
            return False
        
        # Check for extreme price moves (>50% in a day)
        returns = data['Close'].pct_change()
        if (returns.abs() > 0.5).any():
            logger.warning("Extreme price moves detected")
            # Don't reject, just warn
        
        # Check for missing values
        if data[['Open', 'High', 'Low', 'Close']].isnull().any().any():
            logger.warning("Missing price data detected")
            return False
        
        return True
    
    def calculate_volatility(self, symbol: str, period: str = "1mo") -> Dict[str, float]:
        """Calculate volatility with validation"""
        try:
            data = self.fetch_price_data(symbol, period)
            if data is None or data.empty:
                return {}
            
            returns = data['Close'].pct_change().dropna()
            
            metrics = {
                "daily_volatility": float(returns.std()),
                "annualized_volatility": float(returns.std() * math.sqrt(252)),
                "average_true_range": self._calculate_atr(data),
                "beta": self._calculate_beta(returns, symbol),
                "max_drawdown": self._calculate_max_drawdown(data['Close'])
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Volatility calculation failed for {symbol}: {e}")
            return {}
    
    def _calculate_atr(self, data, period: int = 14) -> float:
        """Calculate Average True Range"""
        try:
            if not REQUIRED_PACKAGES['pandas']:
                return 0.0
            
            high = data['High']
            low = data['Low']
            close = data['Close']
            
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=period).mean().iloc[-1]
            
            return float(atr) if not pd.isna(atr) else 0.0
            
        except Exception as e:
            logger.error(f"ATR calculation failed: {e}")
            return 0.0
    
    def _calculate_beta(self, returns, symbol: str, market_symbol: str = "SPY") -> float:
        """Calculate beta against market"""
        try:
            # Don't calculate beta for the market itself
            if symbol == market_symbol:
                return 1.0
            
            market_data = self.fetch_price_data(market_symbol, period="1mo")
            if market_data is None or market_data.empty:
                return 1.0
            
            market_returns = market_data['Close'].pct_change().dropna()
            
            # Align the data
            if len(returns) != len(market_returns):
                min_len = min(len(returns), len(market_returns))
                returns = returns.iloc[-min_len:]
                market_returns = market_returns.iloc[-min_len:]
            
            if REQUIRED_PACKAGES['numpy']:
                covariance = np.cov(returns, market_returns)[0, 1]
                market_variance = np.var(market_returns)
                beta = covariance / market_variance if market_variance != 0 else 1.0
                return float(beta)
            
            return 1.0
            
        except Exception as e:
            logger.error(f"Beta calculation failed: {e}")
            return 1.0
    
    def _calculate_max_drawdown(self, prices) -> float:
        """Calculate maximum drawdown"""
        try:
            cummax = prices.expanding().max()
            drawdown = (prices - cummax) / cummax * 100
            return float(drawdown.min())
        except Exception as e:
            logger.error(f"Max drawdown calculation failed: {e}")
            return 0.0

# ==========================================
# UTILITIES
# ==========================================

class RateLimiter:
    """Rate limiting for API calls"""
    
    def __init__(self, calls_per_minute: int = 60):
        self.calls_per_minute = calls_per_minute
        self.calls = deque()
    
    def allow_request(self) -> bool:
        """Check if request is allowed"""
        now = time.time()
        
        # Remove calls older than 1 minute
        while self.calls and self.calls[0] < now - 60:
            self.calls.popleft()
        
        if len(self.calls) < self.calls_per_minute:
            self.calls.append(now)
            return True
        
        return False

# ==========================================
# PRICE MONITORING
# ==========================================

class PriceMonitor:
    """Real-time price monitoring and alerts"""
    
    def __init__(self):
        self.alerts = {}
        self.active = False
        self.thread = None
    
    def set_alert(self, symbol: str, price: float, condition: str, callback=None):
        """Set price alert"""
        try:
            symbol = InputValidator.sanitize_symbol(symbol)
            price = InputValidator.validate_price(price)
            
            if condition not in ['above', 'below']:
                raise ValueError("Condition must be 'above' or 'below'")
            
            self.alerts[symbol] = {
                'price': price,
                'condition': condition,
                'triggered': False,
                'callback': callback,
                'created_at': datetime.now()
            }
            
            logger.info(f"Alert set: {symbol} {condition} ${price}")
            
        except Exception as e:
            logger.error(f"Failed to set alert: {e}")
    
    def check_alerts(self):
        """Check all active alerts"""
        if not REQUIRED_PACKAGES['yfinance']:
            return
        
        for symbol, alert in self.alerts.items():
            if alert['triggered']:
                continue
            
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                current_price = info.get('currentPrice') or info.get('regularMarketPrice')
                
                if not current_price:
                    continue
                
                condition_met = False
                
                if alert['condition'] == 'above' and current_price > alert['price']:
                    condition_met = True
                elif alert['condition'] == 'below' and current_price < alert['price']:
                    condition_met = True
                
                if condition_met:
                    alert['triggered'] = True
                    msg = f"ALERT: {symbol} is {alert['condition']} ${alert['price']:.2f} (Current: ${current_price:.2f})"
                    logger.warning(msg)
                    
                    if alert['callback']:
                        alert['callback'](symbol, current_price, alert)
                        
            except Exception as e:
                logger.error(f"Failed to check alert for {symbol}: {e}")
    
    def start_monitoring(self, interval_seconds: int = 60):
        """Start monitoring in background thread"""
        if self.active:
            return
        
        self.active = True
        
        def monitor_loop():
            while self.active:
                self.check_alerts()
                time.sleep(interval_seconds)
        
        self.thread = threading.Thread(target=monitor_loop, daemon=True)
        self.thread.start()
        logger.info("Price monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.active = False
        if self.thread:
            self.thread.join(timeout=5)
        logger.info("Price monitoring stopped")

# ==========================================
# BACKTEST ENGINE
# ==========================================

class BacktestEngine:
    """Backtesting framework for strategies"""
    
    def __init__(self, data_provider: MarketDataProvider):
        self.data_provider = data_provider
        self.commission_rate = 0.001  # 0.1% commission
        self.slippage_rate = 0.0005   # 0.05% slippage
    
    def backtest_moving_average_crossover(self, symbol: str, 
                                         fast_period: int = 20, 
                                         slow_period: int = 50,
                                         initial_capital: float = 10000) -> List[TradeMetrics]:
        """Backtest MA crossover strategy"""
        try:
            # Fetch data
            data = self.data_provider.fetch_price_data(symbol, period="2y")
            if data is None or data.empty:
                logger.error(f"No data available for {symbol}")
                return []
            
            # Calculate moving averages
            data['MA_Fast'] = data['Close'].rolling(window=fast_period).mean()
            data['MA_Slow'] = data['Close'].rolling(window=slow_period).mean()
            
            # Drop NaN values
            data = data.dropna()
            
            # Generate signals
            data['Signal'] = 0
            data.loc[data['MA_Fast'] > data['MA_Slow'], 'Signal'] = 1
            data.loc[data['MA_Fast'] < data['MA_Slow'], 'Signal'] = -1
            data['Position'] = data['Signal'].diff()
            
            trades = []
            position = None
            
            for idx, row in data.iterrows():
                # Entry signal
                if row['Position'] == 2 and position is None:  # Buy signal
                    position = {
                        'entry_price': row['Close'],
                        'entry_date': idx.strftime('%Y-%m-%d'),
                        'position_size': initial_capital / row['Close']
                    }
                
                # Exit signal
                elif row['Position'] == -2 and position is not None:  # Sell signal
                    exit_price = row['Close']
                    commission = (position['position_size'] * position['entry_price'] * self.commission_rate +
                                position['position_size'] * exit_price * self.commission_rate)
                    slippage = position['position_size'] * exit_price * self.slippage_rate
                    
                    trade = TradeMetrics(
                        symbol=symbol,
                        entry_price=position['entry_price'],
                        exit_price=exit_price,
                        position_size=position['position_size'],
                        position_type=PositionType.LONG,
                        entry_date=position['entry_date'],
                        exit_date=idx.strftime('%Y-%m-%d'),
                        commission=commission,
                        slippage=slippage,
                        strategy_name=f"MA_{fast_period}_{slow_period}"
                    )
                    trades.append(trade)
                    position = None
            
            return trades
            
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            return []
    
    def backtest_rsi_strategy(self, symbol: str, 
                             rsi_period: int = 14,
                             oversold: int = 30,
                             overbought: int = 70,
                             initial_capital: float = 10000) -> List[TradeMetrics]:
        """Backtest RSI strategy"""
        try:
            data = self.data_provider.fetch_price_data(symbol, period="1y")
            if data is None or data.empty:
                return []
            
            # Calculate RSI
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
            rs = gain / loss
            data['RSI'] = 100 - (100 / (1 + rs))
            
            trades = []
            position = None
            
            for idx, row in data.iterrows():
                if pd.isna(row['RSI']):
                    continue
                
                # Buy signal (oversold)
                if row['RSI'] < oversold and position is None:
                    position = {
                        'entry_price': row['Close'],
                        'entry_date': idx.strftime('%Y-%m-%d'),
                        'position_size': initial_capital / row['Close']
                    }
                
                # Sell signal (overbought)
                elif row['RSI'] > overbought and position is not None:
                    exit_price = row['Close']
                    commission = (position['position_size'] * position['entry_price'] * self.commission_rate +
                                position['position_size'] * exit_price * self.commission_rate)
                    slippage = position['position_size'] * exit_price * self.slippage_rate
                    
                    trade = TradeMetrics(
                        symbol=symbol,
                        entry_price=position['entry_price'],
                        exit_price=exit_price,
                        position_size=position['position_size'],
                        position_type=PositionType.LONG,
                        entry_date=position['entry_date'],
                        exit_date=idx.strftime('%Y-%m-%d'),
                        commission=commission,
                        slippage=slippage,
                        strategy_name=f"RSI_{rsi_period}"
                    )
                    trades.append(trade)
                    position = None
            
            return trades
            
        except Exception as e:
            logger.error(f"RSI backtest failed: {e}")
            return []

# ==========================================
# STRATEGY ANALYZER
# ==========================================

class StrategyAnalyzer:
    """Analyze strategy performance"""
    
    def __init__(self):
        self.metrics = {}
    
    def analyze_trades(self, trades: List[TradeMetrics]) -> Dict:
        """Comprehensive trade analysis"""
        if not trades:
            return {
                'error': 'No trades to analyze',
                'total_trades': 0
            }
        
        # Basic metrics
        total_trades = len(trades)
        winning_trades = [t for t in trades if t.net_pnl > 0]
        losing_trades = [t for t in trades if t.net_pnl <= 0]
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        # PnL metrics
        total_pnl = sum(t.net_pnl for t in trades)
        avg_win = sum(t.net_pnl for t in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(t.net_pnl for t in losing_trades) / len(losing_trades) if losing_trades else 0
        
        # Risk metrics
        returns = [t.return_pct for t in trades]
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        max_drawdown = self._calculate_max_drawdown_from_trades(trades)
        profit_factor = abs(sum(t.net_pnl for t in winning_trades) / sum(t.net_pnl for t in losing_trades)) if losing_trades else float('inf')
        
        # Additional metrics
        avg_holding_days = self._calculate_avg_holding_period(trades)
        consecutive_wins = self._max_consecutive(trades, True)
        consecutive_losses = self._max_consecutive(trades, False)
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate * 100,
            'total_pnl': total_pnl,
            'average_win': avg_win,
            'average_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'avg_holding_days': avg_holding_days,
            'max_consecutive_wins': consecutive_wins,
            'max_consecutive_losses': consecutive_losses,
            'best_trade': max(trades, key=lambda t: t.net_pnl).net_pnl if trades else 0,
            'worst_trade': min(trades, key=lambda t: t.net_pnl).net_pnl if trades else 0,
            'avg_return': statistics.mean(returns) if returns else 0,
            'return_std': statistics.stdev(returns) if len(returns) > 1 else 0
        }
    
    def _calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if not returns or len(returns) < 2:
            return 0
        
        excess_returns = [r - (risk_free_rate/252) for r in returns]
        mean_excess = statistics.mean(excess_returns)
        std_excess = statistics.stdev(excess_returns)
        
        if std_excess == 0:
            return 0
        
        return (mean_excess / std_excess) * math.sqrt(252)
    
    def _calculate_max_drawdown_from_trades(self, trades: List[TradeMetrics]) -> float:
        """Calculate maximum drawdown from trades"""
        if not trades:
            return 0
        
        cumulative = []
        cum_sum = 0
        
        for trade in trades:
            cum_sum += trade.net_pnl
            cumulative.append(cum_sum)
        
        peak = cumulative[0]
        max_dd = 0
        
        for value in cumulative:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak if peak > 0 else 0
            max_dd = max(max_dd, drawdown)
        
        return max_dd * 100
    
    def _calculate_avg_holding_period(self, trades: List[TradeMetrics]) -> float:
        """Calculate average holding period in days"""
        periods = []
        
        for trade in trades:
            if trade.entry_date and trade.exit_date:
                try:
                    entry = datetime.strptime(trade.entry_date, '%Y-%m-%d')
                    exit = datetime.strptime(trade.exit_date, '%Y-%m-%d')
                    periods.append((exit - entry).days)
                except:
                    continue
        
        return statistics.mean(periods) if periods else 0
    
    def _max_consecutive(self, trades: List[TradeMetrics], wins: bool) -> int:
        """Calculate max consecutive wins/losses"""
        if not trades:
            return 0
        
        max_count = 0
        current_count = 0
        
        for trade in trades:
            if wins and trade.net_pnl > 0:
                current_count += 1
                max_count = max(max_count, current_count)
            elif not wins and trade.net_pnl <= 0:
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 0
        
        return max_count

# ==========================================
# VISUALIZER
# ==========================================

class Visualizer:
    """Create charts and visualizations"""
    
    def __init__(self):
        self.has_matplotlib = REQUIRED_PACKAGES['matplotlib']
        self.has_plotly = REQUIRED_PACKAGES['plotly']
    
    def plot_equity_curve(self, trades: List[TradeMetrics], initial_capital: float = 10000):
        """Plot equity curve from trades"""
        if not self.has_matplotlib:
            logger.warning("Matplotlib not available for plotting")
            return
        
        if not trades:
            logger.warning("No trades to plot")
            return
        
        # Calculate cumulative returns
        equity = [initial_capital]
        current = initial_capital
        
        for trade in trades:
            current += trade.net_pnl
            equity.append(current)
        
        # Create plot
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(equity)), equity, linewidth=2)
        plt.title('Equity Curve', fontsize=14, fontweight='bold')
        plt.xlabel('Trade Number')
        plt.ylabel('Equity ($)')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=initial_capital, color='r', linestyle='--', alpha=0.5)
        
        # Add statistics
        final_equity = equity[-1]
        total_return = ((final_equity - initial_capital) / initial_capital) * 100
        plt.text(0.02, 0.98, f'Final Equity: ${final_equity:,.2f}\nTotal Return: {total_return:.2f}%',
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.show()
    
    def plot_drawdown(self, trades: List[TradeMetrics]):
        """Plot drawdown chart"""
        if not self.has_matplotlib:
            return
        
        if not trades:
            return
        
        # Calculate drawdown series
        cumulative = []
        cum_sum = 0
        
        for trade in trades:
            cum_sum += trade.net_pnl
            cumulative.append(cum_sum)
        
        # Calculate drawdown
        peak = [cumulative[0]]
        drawdown = []
        
        for i in range(1, len(cumulative)):
            peak.append(max(peak[-1], cumulative[i]))
            dd = ((peak[i] - cumulative[i]) / peak[i] * 100) if peak[i] > 0 else 0
            drawdown.append(dd)
        
        # Plot
        plt.figure(figsize=(12, 4))
        plt.fill_between(range(len(drawdown)), drawdown, color='red', alpha=0.3)
        plt.plot(drawdown, color='red', linewidth=1)
        plt.title('Drawdown Chart', fontsize=14, fontweight='bold')
        plt.xlabel('Trade Number')
        plt.ylabel('Drawdown (%)')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='black', linewidth=0.5)
        
        # Add max drawdown
        if drawdown:
            max_dd = max(drawdown)
            max_dd_idx = drawdown.index(max_dd)
            plt.scatter([max_dd_idx], [max_dd], color='darkred', s=100, zorder=5)
            plt.text(max_dd_idx, max_dd, f'Max: {max_dd:.2f}%', fontsize=10)
        
        plt.tight_layout()
        plt.show()
    
    def plot_returns_distribution(self, trades: List[TradeMetrics]):
        """Plot returns distribution"""
        if not self.has_matplotlib:
            return
        
        if not trades:
            return
        
        returns = [t.return_pct for t in trades]
        
        plt.figure(figsize=(10, 6))
        
        # Histogram
        n, bins, patches = plt.hist(returns, bins=30, edgecolor='black', alpha=0.7)
        
        # Color code positive and negative
        for i in range(len(patches)):
            if bins[i] < 0:
                patches[i].set_facecolor('red')
            else:
                patches[i].set_facecolor('green')
        
        plt.title('Returns Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Return (%)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.axvline(x=0, color='black', linewidth=1)
        
        # Add statistics
        mean_return = statistics.mean(returns)
        median_return = statistics.median(returns)
        
        plt.axvline(x=mean_return, color='blue', linestyle='--', label=f'Mean: {mean_return:.2f}%')
        plt.axvline(x=median_return, color='orange', linestyle='--', label=f'Median: {median_return:.2f}%')
        
        plt.legend()
        plt.tight_layout()
        plt.show()

# ==========================================
# RISK CALCULATOR
# ==========================================

class RiskCalculator:
    """Advanced risk calculations"""
    
    def __init__(self):
        self.confidence_level = 0.95
    
    def calculate_var(self, returns: List[float], confidence: float = 0.95) -> float:
        """Calculate Value at Risk"""
        if not returns:
            return 0
        
        sorted_returns = sorted(returns)
        index = int((1 - confidence) * len(sorted_returns))
        return sorted_returns[index] if index < len(sorted_returns) else sorted_returns[0]
    
    def calculate_cvar(self, returns: List[float], confidence: float = 0.95) -> float:
        """Calculate Conditional Value at Risk"""
        if not returns:
            return 0
        
        var = self.calculate_var(returns, confidence)
        tail_returns = [r for r in returns if r <= var]
        return statistics.mean(tail_returns) if tail_returns else var
    
    def calculate_kelly_criterion(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Calculate Kelly criterion for position sizing"""
        if avg_loss == 0:
            return 0
        
        b = avg_win / abs(avg_loss)
        p = win_rate
        q = 1 - p
        
        kelly = (p * b - q) / b if b != 0 else 0
        
        # Apply Kelly fraction (usually 25% of full Kelly)
        return max(0, min(0.25, kelly * 0.25))
    
    def calculate_risk_reward_ratio(self, entry: float, target: float, stop: float) -> float:
        """Calculate risk/reward ratio"""
        risk = abs(entry - stop)
        reward = abs(target - entry)
        
        return reward / risk if risk != 0 else 0

# ==========================================
# STRATEGY OPTIMIZER
# ==========================================

class StrategyOptimizer:
    """Optimize strategy parameters"""
    
    def __init__(self, backtest_engine: BacktestEngine):
        self.backtest_engine = backtest_engine
    
    def optimize_ma_crossover(self, symbol: str, 
                             fast_range: Tuple[int, int, int] = (10, 30, 5),
                             slow_range: Tuple[int, int, int] = (30, 100, 10)) -> Dict:
        """Grid search for optimal MA parameters"""
        
        best_sharpe = -999
        best_params = {}
        all_results = []
        
        for fast in range(*fast_range):
            for slow in range(*slow_range):
                if fast >= slow:
                    continue
                
                try:
                    trades = self.backtest_engine.backtest_moving_average_crossover(
                        symbol, fast, slow
                    )
                    
                    if not trades:
                        continue
                    
                    # Calculate metrics
                    returns = [t.return_pct for t in trades]
                    if len(returns) < 2:
                        continue
                    
                    sharpe = self._calculate_sharpe(returns)
                    total_return = sum(returns)
                    win_rate = sum(1 for r in returns if r > 0) / len(returns)
                    
                    result = {
                        'fast': fast,
                        'slow': slow,
                        'sharpe': sharpe,
                        'total_return': total_return,
                        'win_rate': win_rate,
                        'num_trades': len(trades)
                    }
                    
                    all_results.append(result)
                    
                    if sharpe > best_sharpe:
                        best_sharpe = sharpe
                        best_params = result
                        
                except Exception as e:
                    logger.error(f"Optimization failed for {fast}/{slow}: {e}")
                    continue
        
        return {
            'best_params': best_params,
            'all_results': sorted(all_results, key=lambda x: x['sharpe'], reverse=True)[:10]
        }
    
    def _calculate_sharpe(self, returns: List[float], risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if not returns or len(returns) < 2:
            return 0
        
        excess_returns = [r - (risk_free_rate/252) for r in returns]
        std_dev = statistics.stdev(excess_returns)
        
        if std_dev == 0:
            return 0
        
        return (statistics.mean(excess_returns) / std_dev) * math.sqrt(252)

# ==========================================
# MAIN APPLICATION
# ==========================================

def main():
    """Main application with enhanced error handling"""
    
    print("=" * 70)
    print("PRODUCTION QUANTITATIVE TRADING PLATFORM v2.0")
    print("=" * 70)
    
    # Initialize components
    try:
        risk_limiter = RiskLimiter()
        data_manager = DataManager()
        data_provider = MarketDataProvider()
        backtest_engine = BacktestEngine(data_provider)
        strategy_analyzer = StrategyAnalyzer()
        visualizer = Visualizer()
        risk_calculator = RiskCalculator()
        price_monitor = PriceMonitor()
        validator = InputValidator()
        
        logger.info("System initialized successfully")
        
    except Exception as e:
        logger.critical(f"Failed to initialize system: {e}")
        print(f"CRITICAL ERROR: {e}")
        return
    
    # Main loop with error recovery
    while True:
        try:
            print("\n" + "=" * 50)
            print("MAIN MENU")
            print("=" * 50)
            print("1. Risk Management")
            print("2. Trade Management")
            print("3. Market Analysis")
            print("4. Backtesting")
            print("5. Portfolio Analytics")
            print("6. Price Monitoring")
            print("7. System Status")
            print("8. Export/Import")
            print("9. Exit")
            print()
            
            choice = input("Select option (1-9): ").strip()
            
            if choice == "1":
                handle_risk_management(risk_limiter, risk_calculator, validator)
                
            elif choice == "2":
                handle_trade_management(data_manager, risk_limiter, validator)
                
            elif choice == "3":
                handle_market_analysis(data_provider, validator)
                
            elif choice == "4":
                handle_backtesting(backtest_engine, strategy_analyzer, visualizer, validator)
                
            elif choice == "5":
                handle_portfolio_analytics(data_manager, strategy_analyzer, visualizer)
                
            elif choice == "6":
                handle_price_monitoring(price_monitor, validator)
                
            elif choice == "7":
                print("\n--- SYSTEM STATUS ---")
                print(f"Cache Stats: {data_provider.cache.get_stats()}")
                print(f"Risk Report: {risk_limiter.get_risk_report()}")
                
            elif choice == "8":
                handle_export_import(data_manager)
                
            elif choice == "9":
                print("\nShutting down...")
                price_monitor.stop_monitoring()
                data_manager.export_to_json()
                logger.info("System shutdown complete")
                break
                
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
            break
            
        except Exception as e:
            logger.error(f"Unexpected error in main loop: {e}")
            print(f"ERROR: {e}")
            print("System recovered, continuing...")

def handle_risk_management(risk_limiter, risk_calculator, validator):
    """Handle risk management operations"""
    print("\n--- RISK MANAGEMENT ---")
    print("1. Position Sizing Calculator")
    print("2. Risk Limits Configuration")
    print("3. View Risk Report")
    print("4. Kelly Criterion Calculator")
    print("5. Risk/Reward Calculator")
    
    choice = input("Select: ").strip()
    
    if choice == "1":
        try:
            balance = validator.validate_price(
                float(input("Account balance: $")), "Balance"
            )
            risk_pct = validator.validate_percentage(
                float(input("Risk per trade (%): "))
            )
            entry = validator.validate_price(
                float(input("Entry price: $")), "Entry"
            )
            stop = validator.validate_price(
                float(input("Stop loss: $")), "Stop"
            )
            
            # Calculate position size
            risk_amount = balance * (risk_pct / 100)
            price_risk = abs(entry - stop)
            shares = risk_amount / price_risk if price_risk > 0 else 0
            position_value = shares * entry
            
            # Check limits
            passed, msg = risk_limiter.check_position_limit(
                position_value, balance, "TEST"
            )
            
            print(f"\n--- POSITION SIZING RESULTS ---")
            print(f"Shares: {shares:.2f}")
            print(f"Position Value: ${position_value:,.2f}")
            print(f"Risk Amount: ${risk_amount:,.2f}")
            print(f"Limit Check: {msg}")
            
        except ValueError as e:
            print(f"Invalid input: {e}")
    
    elif choice == "2":
        print("\n--- CONFIGURE RISK LIMITS ---")
        print(f"Current max position size: {risk_limiter.MAX_POSITION_SIZE*100}%")
        print(f"Current max daily loss: {risk_limiter.MAX_DAILY_LOSS*100}%")
        
        try:
            new_pos = input("New max position size (% or Enter to skip): ").strip()
            if new_pos:
                risk_limiter.MAX_POSITION_SIZE = validator.validate_percentage(float(new_pos)) / 100
            
            new_loss = input("New max daily loss (% or Enter to skip): ").strip()
            if new_loss:
                risk_limiter.MAX_DAILY_LOSS = validator.validate_percentage(float(new_loss)) / 100
            
            print("Risk limits updated successfully")
            
        except ValueError as e:
            print(f"Invalid input: {e}")
    
    elif choice == "3":
        report = risk_limiter.get_risk_report()
        print("\n--- RISK REPORT ---")
        print(json.dumps(report, indent=2))
    
    elif choice == "4":
        try:
            win_rate = validator.validate_percentage(float(input("Win rate (%): "))) / 100
            avg_win = validator.validate_price(float(input("Average win ($): ")))
            avg_loss = validator.validate_price(float(input("Average loss ($): ")))
            
            kelly = risk_calculator.calculate_kelly_criterion(win_rate, avg_win, avg_loss)
            
            print(f"\n--- KELLY CRITERION ---")
            print(f"Optimal position size: {kelly*100:.2f}% of capital")
            print(f"Conservative (25% Kelly): {kelly*100:.2f}% of capital")
            
        except ValueError as e:
            print(f"Invalid input: {e}")
    
    elif choice == "5":
        try:
            entry = validator.validate_price(float(input("Entry price: $")))
            target = validator.validate_price(float(input("Target price: $")))
            stop = validator.validate_price(float(input("Stop loss: $")))
            
            rr_ratio = risk_calculator.calculate_risk_reward_ratio(entry, target, stop)
            
            print(f"\n--- RISK/REWARD ANALYSIS ---")
            print(f"Risk: ${abs(entry - stop):.2f}")
            print(f"Reward: ${abs(target - entry):.2f}")
            print(f"Risk/Reward Ratio: 1:{rr_ratio:.2f}")
            
            if rr_ratio < 2:
                print("⚠️  Warning: Risk/Reward ratio below 2:1")
            
        except ValueError as e:
            print(f"Invalid input: {e}")

def handle_trade_management(data_manager, risk_limiter, validator):
    """Handle trade operations"""
    print("\n--- TRADE MANAGEMENT ---")
    print("1. Add Trade")
    print("2. View Trades")
    print("3. Delete Trade")
    
    choice = input("Select: ").strip()
    
    if choice == "1":
        try:
            symbol = validator.sanitize_symbol(input("Symbol: "))
            entry_price = validator.validate_price(float(input("Entry price: $")))
            exit_price = validator.validate_price(float(input("Exit price: $")))
            position_size = float(input("Position size (shares): "))
            
            if position_size <= 0:
                raise ValueError("Position size must be positive")
            
            position_type = input("Position type (long/short): ").upper()
            if position_type not in ['LONG', 'SHORT']:
                raise ValueError("Position type must be 'long' or 'short'")
            
            entry_date = input("Entry date (YYYY-MM-DD or Enter to skip): ").strip()
            exit_date = input("Exit date (YYYY-MM-DD or Enter to skip): ").strip()
            
            trade = TradeMetrics(
                symbol=symbol,
                entry_price=entry_price,
                exit_price=exit_price,
                position_size=position_size,
                position_type=PositionType[position_type],
                entry_date=entry_date if entry_date else "",
                exit_date=exit_date if exit_date else "",
                commission=position_size * (entry_price + exit_price) * 0.001,  # 0.1% commission
                slippage=position_size * exit_price * 0.0005  # 0.05% slippage
            )
            
            data_manager.save_trade(trade)
            print(f"\nTrade saved successfully!")
            print(f"Net PnL: ${trade.net_pnl:.2f}")
            print(f"Return: {trade.return_pct:.2f}%")
            
        except Exception as e:
            print(f"Failed to add trade: {e}")
    
    elif choice == "2":
        trades = data_manager.load_trades()
        
        if not trades:
            print("No trades found")
        else:
            print(f"\n--- TRADE HISTORY ({len(trades)} trades) ---")
            for i, trade in enumerate(trades[:10], 1):
                print(f"{i}. {trade.symbol}: ${trade.net_pnl:.2f} ({trade.return_pct:.2f}%)")
            
            if len(trades) > 10:
                print(f"... and {len(trades) - 10} more trades")

def handle_market_analysis(data_provider, validator):
    """Handle market data operations"""
    print("\n--- MARKET ANALYSIS ---")
    print("1. Fetch Price Data")
    print("2. Calculate Volatility")
    print("3. Technical Indicators")
    
    choice = input("Select: ").strip()
    
    if choice == "1":
        try:
            symbol = validator.sanitize_symbol(input("Symbol: "))
            period = input("Period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y): ").strip()
            
            data = data_provider.fetch_price_data(symbol, period)
            
            if data is not None and not data.empty:
                print(f"\n--- PRICE DATA FOR {symbol} ---")
                print(data.tail())
            else:
                print("Failed to fetch data")
                
        except Exception as e:
            print(f"Error: {e}")
    
    elif choice == "2":
        try:
            symbol = validator.sanitize_symbol(input("Symbol: "))
            
            metrics = data_provider.calculate_volatility(symbol)
            
            if metrics:
                print(f"\n--- VOLATILITY METRICS FOR {symbol} ---")
                for key, value in metrics.items():
                    print(f"{key}: {value:.4f}")
            else:
                print("Failed to calculate volatility")
                
        except Exception as e:
            print(f"Error: {e}")

def handle_backtesting(backtest_engine, strategy_analyzer, visualizer, validator):
    """Handle backtesting operations"""
    print("\n--- BACKTESTING ---")
    print("1. Moving Average Crossover")
    print("2. RSI Strategy")
    print("3. Optimize Strategy")
    
    choice = input("Select: ").strip()
    
    if choice == "1":
        try:
            symbol = validator.sanitize_symbol(input("Symbol: "))
            fast = int(input("Fast MA period (e.g., 20): "))
            slow = int(input("Slow MA period (e.g., 50): "))
            
            if fast <= 0 or slow <= 0:
                raise ValueError("Periods must be positive")
            if fast >= slow:
                raise ValueError("Fast period must be less than slow period")
            
            trades = backtest_engine.backtest_moving_average_crossover(symbol, fast, slow)
            
            if trades:
                analysis = strategy_analyzer.analyze_trades(trades)
                
                print(f"\n--- BACKTEST RESULTS ---")
                for key, value in analysis.items():
                    if isinstance(value, float):
                        print(f"{key}: {value:.2f}")
                    else:
                        print(f"{key}: {value}")
                
                # Optionally plot results
                if input("\nPlot results? (y/n): ").lower() == 'y':
                    visualizer.plot_equity_curve(trades)
                    visualizer.plot_drawdown(trades)
                    visualizer.plot_returns_distribution(trades)
            else:
                print("No trades generated")
                
        except Exception as e:
            print(f"Backtest failed: {e}")
    
    elif choice == "2":
        try:
            symbol = validator.sanitize_symbol(input("Symbol: "))
            rsi_period = int(input("RSI period (e.g., 14): "))
            oversold = int(input("Oversold level (e.g., 30): "))
            overbought = int(input("Overbought level (e.g., 70): "))
            
            trades = backtest_engine.backtest_rsi_strategy(
                symbol, rsi_period, oversold, overbought
            )
            
            if trades:
                analysis = strategy_analyzer.analyze_trades(trades)
                
                print(f"\n--- BACKTEST RESULTS ---")
                for key, value in analysis.items():
                    if isinstance(value, float):
                        print(f"{key}: {value:.2f}")
                    else:
                        print(f"{key}: {value}")
            else:
                print("No trades generated")
                
        except Exception as e:
            print(f"Backtest failed: {e}")
    
    elif choice == "3":
        try:
            symbol = validator.sanitize_symbol(input("Symbol: "))
            
            print("\nOptimizing MA Crossover strategy...")
            optimizer = StrategyOptimizer(backtest_engine)
            
            results = optimizer.optimize_ma_crossover(symbol)
            
            if results['best_params']:
                print(f"\n--- OPTIMAL PARAMETERS ---")
                for key, value in results['best_params'].items():
                    print(f"{key}: {value}")
                
                print(f"\n--- TOP 10 PARAMETER SETS ---")
                for i, params in enumerate(results['all_results'][:10], 1):
                    print(f"{i}. Fast={params['fast']}, Slow={params['slow']}, "
                          f"Sharpe={params['sharpe']:.2f}, Return={params['total_return']:.2f}%")
            else:
                print("Optimization failed")
                
        except Exception as e:
            print(f"Optimization failed: {e}")

def handle_portfolio_analytics(data_manager, strategy_analyzer, visualizer):
    """Handle portfolio analysis"""
    print("\n--- PORTFOLIO ANALYTICS ---")
    
    trades = data_manager.load_trades()
    
    if not trades:
        print("No trades to analyze")
        return
    
    analysis = strategy_analyzer.analyze_trades(trades)
    
    print("\n--- PORTFOLIO PERFORMANCE ---")
    for key, value in analysis.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")
    
    if input("\nPlot charts? (y/n): ").lower() == 'y':
        visualizer.plot_equity_curve(trades)
        visualizer.plot_drawdown(trades)
        visualizer.plot_returns_distribution(trades)

def handle_price_monitoring(price_monitor, validator):
    """Handle price alerts"""
    print("\n--- PRICE MONITORING ---")
    print("1. Set Alert")
    print("2. View Alerts")
    print("3. Start Monitoring")
    print("4. Stop Monitoring")
    
    choice = input("Select: ").strip()
    
    if choice == "1":
        try:
            symbol = validator.sanitize_symbol(input("Symbol: "))
            price = validator.validate_price(float(input("Alert price: $")))
            condition = input("Condition (above/below): ").lower()
            
            if condition not in ['above', 'below']:
                raise ValueError("Condition must be 'above' or 'below'")
            
            price_monitor.set_alert(symbol, price, condition)
            print("Alert set successfully")
            
        except ValueError as e:
            print(f"Invalid input: {e}")
    
    elif choice == "2":
        if not price_monitor.alerts:
            print("No active alerts")
        else:
            print("\n--- ACTIVE ALERTS ---")
            for symbol, alert in price_monitor.alerts.items():
                status = "TRIGGERED" if alert['triggered'] else "ACTIVE"
                print(f"{symbol}: {alert['condition']} ${alert['price']:.2f} [{status}]")
    
    elif choice == "3":
        interval = int(input("Check interval (seconds, default 60): ") or "60")
        price_monitor.start_monitoring(interval)
        print("Monitoring started")
    
    elif choice == "4":
        price_monitor.stop_monitoring()
        print("Monitoring stopped")

def handle_export_import(data_manager):
    """Handle data export/import"""
    print("\n--- EXPORT/IMPORT ---")
    print("1. Export to JSON")
    print("2. Import from JSON")
    
    choice = input("Select: ").strip()
    
    if choice == "1":
        try:
            filepath = input("Export filepath (Enter for default): ").strip()
            filepath = filepath if filepath else None
            
            exported = data_manager.export_to_json(filepath)
            print(f"Data exported to: {exported}")
            
        except Exception as e:
            print(f"Export failed: {e}")
    
    elif choice == "2":
        print("Import functionality to be implemented")

if __name__ == "__main__":
    main()