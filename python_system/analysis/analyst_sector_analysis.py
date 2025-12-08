"""
Analyst Views & Sector News Analysis
====================================

Integrates fundamental analysis from:
- Analyst ratings and price targets
- Earnings estimates and revisions
- Sector rotation analysis
- News sentiment by sector
- Institutional ownership changes

Author: Institutional Trading System
Date: 2025-11-20
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from collections import Counter

logger = logging.getLogger(__name__)

@dataclass
class AnalystConsensus:
    """Analyst consensus data"""
    avg_price_target: float
    upside_potential: float  # % from current price
    num_analysts: int
    buy_ratings: int
    hold_ratings: int
    sell_ratings: int
    consensus_rating: str  # 'strong_buy', 'buy', 'hold', 'sell', 'strong_sell'
    confidence: float  # 0-100
    recent_upgrades: int
    recent_downgrades: int


@dataclass
class SectorAnalysis:
    """Sector analysis results"""
    sector: str
    relative_strength: float  # vs S&P 500
    momentum_score: float  # 0-100
    sentiment_score: float  # -100 to +100
    rotation_signal: str  # 'rotating_in', 'neutral', 'rotating_out'
    top_stocks: List[str]
    confidence: float


class AnalystAnalyzer:
    """
    Analyzes analyst ratings, price targets, and revisions.
    
    Analyst upgrades/downgrades often precede price moves.
    Consensus price targets provide valuation benchmarks.
    """
    
    def __init__(self):
        """Initialize analyst analyzer"""
        self.logger = logging.getLogger(__name__)
    
    def analyze_analyst_ratings(
        self,
        recommendations: List[Dict],
        current_price: float
    ) -> AnalystConsensus:
        """
        Analyze analyst recommendations and price targets.
        
        Args:
            recommendations: List of analyst recommendations from Finnhub
            current_price: Current stock price
            
        Returns:
            AnalystConsensus
        """
        if not recommendations:
            self.logger.warning("No analyst recommendations available")
            return AnalystConsensus(
                avg_price_target=current_price,
                upside_potential=0,
                num_analysts=0,
                buy_ratings=0,
                hold_ratings=0,
                sell_ratings=0,
                consensus_rating='unknown',
                confidence=0,
                recent_upgrades=0,
                recent_downgrades=0
            )
        
        # Extract ratings
        buy_count = 0
        hold_count = 0
        sell_count = 0
        price_targets = []
        
        # Track recent changes (last 30 days)
        recent_upgrades = 0
        recent_downgrades = 0
        cutoff_date = datetime.now() - timedelta(days=30)
        
        for rec in recommendations:
            # Rating (1=Strong Buy, 2=Buy, 3=Hold, 4=Sell, 5=Strong Sell)
            rating = rec.get('grade', 3)
            
            if rating <= 2:
                buy_count += 1
            elif rating == 3:
                hold_count += 1
            else:
                sell_count += 1
            
            # Price target
            target = rec.get('priceTarget', 0)
            if target > 0:
                price_targets.append(target)
            
            # Track upgrades/downgrades
            rec_date = datetime.fromtimestamp(rec.get('period', 0))
            if rec_date > cutoff_date:
                prev_rating = rec.get('fromGrade', rating)
                if rating < prev_rating:  # Lower number = better rating
                    recent_upgrades += 1
                elif rating > prev_rating:
                    recent_downgrades += 1
        
        # Calculate consensus
        total_analysts = buy_count + hold_count + sell_count
        
        if total_analysts == 0:
            consensus_rating = 'unknown'
            confidence = 0
        else:
            buy_pct = buy_count / total_analysts
            sell_pct = sell_count / total_analysts
            
            if buy_pct >= 0.7:
                consensus_rating = 'strong_buy'
                confidence = 90
            elif buy_pct >= 0.5:
                consensus_rating = 'buy'
                confidence = 75
            elif sell_pct >= 0.5:
                consensus_rating = 'sell'
                confidence = 75
            elif sell_pct >= 0.7:
                consensus_rating = 'strong_sell'
                confidence = 90
            else:
                consensus_rating = 'hold'
                confidence = 60
        
        # Average price target
        avg_target = np.mean(price_targets) if price_targets else current_price
        upside = (avg_target / current_price - 1) * 100 if current_price > 0 else 0
        
        # Adjust confidence based on recent activity
        if recent_upgrades > recent_downgrades:
            confidence = min(100, confidence + 10)
        elif recent_downgrades > recent_upgrades:
            confidence = max(0, confidence - 10)
        
        self.logger.info(f"Analyst Consensus: {consensus_rating}")
        self.logger.info(f"  Buy: {buy_count}, Hold: {hold_count}, Sell: {sell_count}")
        self.logger.info(f"  Avg Price Target: ${avg_target:.2f} ({upside:+.1f}% upside)")
        self.logger.info(f"  Recent: {recent_upgrades} upgrades, {recent_downgrades} downgrades")
        
        return AnalystConsensus(
            avg_price_target=avg_target,
            upside_potential=upside,
            num_analysts=total_analysts,
            buy_ratings=buy_count,
            hold_ratings=hold_count,
            sell_ratings=sell_count,
            consensus_rating=consensus_rating,
            confidence=confidence,
            recent_upgrades=recent_upgrades,
            recent_downgrades=recent_downgrades
        )
    
    def analyze_earnings_estimates(
        self,
        earnings_data: Dict
    ) -> Dict[str, any]:
        """
        Analyze earnings estimates and revisions.
        
        Upward earnings revisions are bullish.
        Downward revisions are bearish.
        
        Args:
            earnings_data: Earnings estimates from Finnhub
            
        Returns:
            Dict with earnings analysis
        """
        if not earnings_data:
            return {'status': 'no_data'}
        
        # Extract current quarter estimates
        current_quarter = earnings_data.get('earningsEstimate', [])
        
        if not current_quarter:
            return {'status': 'no_estimates'}
        
        # Get consensus estimate
        consensus = current_quarter[0].get('epsAvg', 0)
        high_estimate = current_quarter[0].get('epsHigh', consensus)
        low_estimate = current_quarter[0].get('epsLow', consensus)
        
        # Number of analysts
        num_analysts = current_quarter[0].get('numberAnalysts', 0)
        
        # Estimate spread (uncertainty)
        spread = (high_estimate - low_estimate) / consensus if consensus != 0 else 0
        
        # Low spread = high confidence
        confidence = max(0, 100 - spread * 100)
        
        return {
            'status': 'available',
            'consensus_eps': consensus,
            'high_estimate': high_estimate,
            'low_estimate': low_estimate,
            'num_analysts': num_analysts,
            'estimate_spread': spread,
            'confidence': confidence
        }


class SectorAnalyzer:
    """
    Analyzes sector rotation and relative strength.
    
    Sector rotation is key to market timing.
    Strong sectors outperform in bull markets.
    Defensive sectors lead in bear markets.
    """
    
    # Sector classifications
    SECTORS = {
        'Technology': ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'AMD', 'INTC'],
        'Healthcare': ['JNJ', 'UNH', 'PFE', 'ABBV', 'TMO', 'MRK', 'LLY'],
        'Financials': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK'],
        'Consumer Discretionary': ['AMZN', 'TSLA', 'HD', 'NKE', 'MCD', 'SBUX', 'TGT'],
        'Consumer Staples': ['PG', 'KO', 'PEP', 'WMT', 'COST', 'CL', 'MDLZ'],
        'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX'],
        'Industrials': ['CAT', 'BA', 'HON', 'UNP', 'UPS', 'LMT', 'GE'],
        'Materials': ['LIN', 'APD', 'SHW', 'ECL', 'NEM', 'FCX', 'DOW'],
        'Utilities': ['NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'SRE'],
        'Real Estate': ['AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'WELL', 'DLR'],
        'Communication Services': ['GOOGL', 'META', 'DIS', 'NFLX', 'CMCSA', 'T', 'VZ']
    }
    
    def __init__(self):
        """Initialize sector analyzer"""
        self.logger = logging.getLogger(__name__)
    
    def get_sector_for_stock(self, symbol: str) -> Optional[str]:
        """Get sector for a stock symbol"""
        for sector, stocks in self.SECTORS.items():
            if symbol in stocks:
                return sector
        return None
    
    def analyze_sector_rotation(
        self,
        sector_performance: Dict[str, float],
        market_performance: float
    ) -> Dict[str, SectorAnalysis]:
        """
        Analyze sector rotation patterns.
        
        Args:
            sector_performance: Dict of sector -> % return
            market_performance: Overall market % return
            
        Returns:
            Dict of sector analyses
        """
        results = {}
        
        for sector, performance in sector_performance.items():
            # Relative strength vs market
            relative_strength = performance - market_performance
            
            # Determine rotation signal
            if relative_strength > 2:
                rotation_signal = 'rotating_in'
                confidence = min(90, 70 + relative_strength * 5)
            elif relative_strength < -2:
                rotation_signal = 'rotating_out'
                confidence = min(90, 70 + abs(relative_strength) * 5)
            else:
                rotation_signal = 'neutral'
                confidence = 50
            
            # Momentum score (0-100)
            momentum_score = max(0, min(100, 50 + relative_strength * 10))
            
            results[sector] = SectorAnalysis(
                sector=sector,
                relative_strength=relative_strength,
                momentum_score=momentum_score,
                sentiment_score=0,  # To be filled by news analysis
                rotation_signal=rotation_signal,
                top_stocks=[],  # To be filled
                confidence=confidence
            )
        
        return results
    
    def identify_sector_leaders(
        self,
        stock_performances: Dict[str, float],
        top_n: int = 3
    ) -> Dict[str, List[str]]:
        """
        Identify top performing stocks in each sector.
        
        Args:
            stock_performances: Dict of symbol -> % return
            top_n: Number of top stocks per sector
            
        Returns:
            Dict of sector -> list of top stock symbols
        """
        sector_leaders = {sector: [] for sector in self.SECTORS.keys()}
        
        # Group stocks by sector
        sector_stocks = {sector: [] for sector in self.SECTORS.keys()}
        
        for symbol, performance in stock_performances.items():
            sector = self.get_sector_for_stock(symbol)
            if sector:
                sector_stocks[sector].append((symbol, performance))
        
        # Get top performers in each sector
        for sector, stocks in sector_stocks.items():
            # Sort by performance
            sorted_stocks = sorted(stocks, key=lambda x: x[1], reverse=True)
            sector_leaders[sector] = [s[0] for s in sorted_stocks[:top_n]]
        
        return sector_leaders


class NewsSentimentAnalyzer:
    """
    Analyzes news sentiment for stocks and sectors.
    
    News sentiment is a leading indicator.
    Positive news often precedes price increases.
    """
    
    def __init__(self):
        """Initialize news sentiment analyzer"""
        self.logger = logging.getLogger(__name__)
    
    def analyze_news_sentiment(
        self,
        news_articles: List[Dict]
    ) -> Dict[str, any]:
        """
        Analyze sentiment from news articles.
        
        Args:
            news_articles: List of news articles from Finnhub/AlphaVantage
            
        Returns:
            Dict with sentiment analysis
        """
        if not news_articles:
            return {
                'sentiment_score': 0,
                'sentiment': 'neutral',
                'confidence': 0,
                'article_count': 0
            }
        
        # Aggregate sentiment scores
        sentiment_scores = []
        
        for article in news_articles:
            # Get sentiment score (-1 to +1)
            score = article.get('sentiment', 0)
            sentiment_scores.append(score)
        
        # Calculate average sentiment
        avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0
        
        # Convert to -100 to +100 scale
        sentiment_score = avg_sentiment * 100
        
        # Classify sentiment
        if sentiment_score > 20:
            sentiment = 'positive'
            confidence = min(90, 60 + abs(sentiment_score) / 2)
        elif sentiment_score < -20:
            sentiment = 'negative'
            confidence = min(90, 60 + abs(sentiment_score) / 2)
        else:
            sentiment = 'neutral'
            confidence = 50
        
        return {
            'sentiment_score': sentiment_score,
            'sentiment': sentiment,
            'confidence': confidence,
            'article_count': len(news_articles),
            'recent_headlines': [a.get('headline', '') for a in news_articles[:5]]
        }
    
    def detect_catalyst_events(
        self,
        news_articles: List[Dict]
    ) -> List[str]:
        """
        Detect potential catalyst events from news.
        
        Args:
            news_articles: List of news articles
            
        Returns:
            List of detected catalyst types
        """
        catalysts = []
        
        # Keywords for different catalyst types
        catalyst_keywords = {
            'earnings': ['earnings', 'eps', 'revenue', 'profit', 'quarterly results'],
            'merger': ['merger', 'acquisition', 'buyout', 'takeover'],
            'product': ['launch', 'new product', 'release', 'unveil'],
            'regulatory': ['fda', 'approval', 'regulation', 'compliance'],
            'partnership': ['partnership', 'deal', 'agreement', 'contract'],
            'guidance': ['guidance', 'outlook', 'forecast', 'expects']
        }
        
        # Search headlines for catalyst keywords
        for article in news_articles:
            headline = article.get('headline', '').lower()
            
            for catalyst_type, keywords in catalyst_keywords.items():
                if any(keyword in headline for keyword in keywords):
                    if catalyst_type not in catalysts:
                        catalysts.append(catalyst_type)
        
        return catalysts


# Global instances
analyst_analyzer = AnalystAnalyzer()
sector_analyzer = SectorAnalyzer()
news_sentiment_analyzer = NewsSentimentAnalyzer()
