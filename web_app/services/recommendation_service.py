#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Recommendation Service
Handles betting recommendations using model predictions and market odds.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class RecommendationService:
    """Service for generating betting recommendations"""
    
    def __init__(self):
        self.logger = logger
    
    def generate_recommendations(self, race_data: Dict, settings: Dict) -> Dict:
        """
        Generate betting recommendations for a race
        
        Args:
            race_data: Race information including runners and odds
            settings: Betting parameters (bet_type, market, model_weight, etc.)
            
        Returns:
            Dictionary containing recommendations
        """
        try:
            runners = race_data.get('runners', [])
            if not runners:
                return {'recommendations': [], 'total_picks': 0, 'message': 'No runners found'}
            
            # Extract settings
            bet_type = settings.get('bet_type', 'place')
            market = settings.get('market', 'fixed')
            model_weight = float(settings.get('model_weight', 0.2))
            min_edge = float(settings.get('min_edge', 0.0))
            min_kelly = float(settings.get('min_kelly', 0.0))
            bankroll = float(settings.get('bankroll', 100.0))
            kelly_fraction = float(settings.get('kelly_fraction', 0.25))
            max_picks = int(settings.get('max_picks', 0))
            
            # Process runners data
            recommendations = []
            
            for runner in runners:
                rec = self._process_runner(runner, bet_type, market, model_weight, 
                                         min_edge, min_kelly, bankroll, kelly_fraction)
                if rec:
                    recommendations.append(rec)
            
            # Sort by edge percentage (descending)
            recommendations.sort(key=lambda x: x.get('edge_pct', 0), reverse=True)
            
            # Limit picks if specified
            if max_picks > 0:
                recommendations = recommendations[:max_picks]
            
            return {
                'recommendations': recommendations,
                'total_picks': len(recommendations),
                'settings': settings,
                'race_info': race_data.get('race_info', {})
            }
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            return {'error': str(e), 'recommendations': [], 'total_picks': 0}
    
    def _process_runner(self, runner: Dict, bet_type: str, market: str, 
                       model_weight: float, min_edge: float, min_kelly: float,
                       bankroll: float, kelly_fraction: float) -> Optional[Dict]:
        """Process a single runner for recommendations"""
        try:
            # Get odds based on market type
            if market == 'fixed':
                if bet_type == 'win':
                    odds = runner.get('odds', {}).get('win_fixed')
                else:  # place
                    odds = runner.get('odds', {}).get('place_fixed')
            else:  # tote
                if bet_type == 'win':
                    odds = runner.get('odds', {}).get('win_tote')
                else:  # place
                    odds = runner.get('odds', {}).get('place_tote')
            
            if not odds or odds <= 1.0:
                return None
            
            # Get model prediction
            model_pred = runner.get('model_prediction', {})
            model_prob = model_pred.get('win_prob', 0.0)
            
            # Adjust model probability for place betting
            if bet_type == 'place':
                # Simple place probability estimation (model win prob * positions_paid)
                # This is a simplified approach - you might want to use a more sophisticated method
                positions_paid = 3  # Default, could be extracted from race data
                model_prob = min(model_prob * positions_paid, 0.95)  # Cap at 95%
            
            # Get market probability
            market_prob = runner.get('implied_win_prob', 0.0)
            if bet_type == 'place' and market_prob:
                # Adjust market probability for place betting
                market_prob = min(market_prob * positions_paid, 0.95)
            
            # Blend model and market probabilities
            if model_weight > 0 and market_prob > 0:
                blend_prob = model_weight * model_prob + (1 - model_weight) * market_prob
            else:
                blend_prob = model_prob
            
            # Calculate edge
            edge_pct = (blend_prob - market_prob) * 100 if market_prob > 0 else 0
            
            # Calculate Kelly percentage
            kelly_pct = self._calculate_kelly(blend_prob, odds)
            
            # Check if runner meets criteria
            if edge_pct < min_edge or kelly_pct < min_kelly:
                return None
            
            # Calculate stake
            stake = bankroll * (kelly_pct / 100.0) * kelly_fraction
            
            # Calculate expected value
            ev = (blend_prob * (odds - 1)) - ((1 - blend_prob) * 1)
            
            # Calculate fair odds
            fair_odds = 1.0 / blend_prob if blend_prob > 0 else 0
            
            return {
                'number': runner.get('number'),
                'runner': runner.get('name'),
                'market_label': f"{market.title()} {bet_type.title()}",
                'odds': odds,
                'market_pct': market_prob * 100,
                'model_pct': model_pred.get('win_percentage', 0.0),
                'blend_pct': blend_prob * 100,
                'edge_pct': edge_pct,
                'ev': ev,
                'fair': fair_odds,
                'kelly_pct': kelly_pct,
                'stake': stake,
                'jockey': runner.get('jockey', ''),
                'barrier': runner.get('barrier', '')
            }
            
        except Exception as e:
            self.logger.error(f"Error processing runner {runner.get('name', 'Unknown')}: {e}")
            return None
    
    def _calculate_kelly(self, probability: float, odds: float) -> float:
        """Calculate Kelly percentage for optimal bet sizing"""
        try:
            if probability <= 0 or odds <= 1.0:
                return 0.0
            
            # Kelly formula: f = (bp - q) / b
            # where b = odds - 1, p = probability of winning, q = 1 - p
            b = odds - 1.0
            p = probability
            q = 1.0 - p
            
            kelly = (b * p - q) / b
            return max(0.0, kelly * 100)  # Return as percentage
            
        except Exception as e:
            self.logger.error(f"Error calculating Kelly: {e}")
            return 0.0
    
    def _deoverround_proportional(self, probabilities: np.ndarray) -> np.ndarray:
        """Remove overround using proportional method"""
        total = np.sum(probabilities)
        if total > 0 and np.isfinite(total):
            return probabilities / total
        return probabilities
    
    def _deoverround_power(self, probabilities: np.ndarray, alpha: float = 0.9) -> np.ndarray:
        """Remove overround using power method"""
        try:
            clipped = np.clip(probabilities, 1e-12, 1.0)
            powered = np.power(clipped, alpha)
            total = np.sum(powered)
            if total > 0:
                return powered / total
            return probabilities
        except Exception:
            return probabilities
    
    def _logit(self, x: np.ndarray) -> np.ndarray:
        """Logit transformation"""
        x = np.clip(x, 1e-12, 1 - 1e-12)
        return np.log(x / (1 - x))
    
    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid transformation"""
        return 1.0 / (1.0 + np.exp(-z))
    
    def blend_probabilities(self, model_probs: np.ndarray, market_probs: np.ndarray, 
                          weight: float, method: str = 'linear') -> np.ndarray:
        """
        Blend model and market probabilities
        
        Args:
            model_probs: Model probabilities
            market_probs: Market probabilities
            weight: Weight for model (0-1)
            method: 'linear' or 'logit'
            
        Returns:
            Blended probabilities
        """
        try:
            if method == 'linear':
                return weight * model_probs + (1 - weight) * market_probs
            else:  # logit
                model_logits = self._logit(model_probs)
                market_logits = self._logit(market_probs)
                blended_logits = weight * model_logits + (1 - weight) * market_logits
                return self._sigmoid(blended_logits)
        except Exception as e:
            self.logger.error(f"Error blending probabilities: {e}")
            return model_probs
