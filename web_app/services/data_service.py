#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data Service
Handles data fetching from TAB APIs and data processing.
"""

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timezone
import re

logger = logging.getLogger(__name__)

class DataService:
    """Service for fetching and processing race data from TAB APIs"""
    
    def __init__(self):
        self.logger = logger
        
        # API endpoints
        self.SCHED_BASE = "https://json.tab.co.nz/schedule"
        self.ODDS_BASE = "https://json.tab.co.nz/odds"
        self.AFF_BASE = "https://api.tab.co.nz/affiliates/v1/racing"
        
        # Headers
        self.HEADERS = {
            "From": "r.cosgrove@hotmail.com",
            "X-Partner": "Personal use",
            "X-Partner-ID": "Personal use",
            "Accept": "application/json",
            "User-Agent": "BetGPT-WebApp/1.0",
        }
        
        self.TIMEOUT = 10
    
    def make_session(self) -> requests.Session:
        """Create a requests session with retry logic"""
        session = requests.Session()
        retry = Retry(
            total=3,
            backoff_factor=0.6,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
            raise_on_status=False,
        )
        session.mount("https://", HTTPAdapter(max_retries=retry))
        return session
    
    def fetch_meeting_context(self, date_str: str, meet_no: int) -> Dict:
        """
        Fetch meeting context including venue, races, and race metadata
        
        Args:
            date_str: Date in YYYY-MM-DD format
            meet_no: Meeting number
            
        Returns:
            Dictionary with meeting context
        """
        try:
            session = self.make_session()
            url = f"{self.SCHED_BASE}/{date_str}/{meet_no}/1"
            response = session.get(url, headers=self.HEADERS, timeout=self.TIMEOUT)
            response.raise_for_status()
            
            data = response.json() if isinstance(response.json(), dict) else {}
            meetings = data.get("meetings", [])
            
            if not meetings:
                return {
                    "date": date_str,
                    "meet": meet_no,
                    "venue": "",
                    "country": "",
                    "races": [],
                    "race_meta_by_no": {}
                }
            
            meeting = meetings[0]
            venue = meeting.get("venue") or meeting.get("name") or ""
            country = (meeting.get("country") or "").upper()
            
            races = []
            race_meta_by_no = {}
            
            for race in meeting.get("races", []):
                try:
                    race_no = int(race.get("number"))
                    races.append(race_no)
                    race_meta_by_no[race_no] = {
                        "id": race.get("id"),
                        "track": self._pretty_track(race.get("track")),
                        "weather": self._pretty_weather(race.get("weather")),
                        "name": race.get("name") or "",
                    }
                except (ValueError, TypeError):
                    continue
            
            races.sort()
            
            return {
                "date": date_str,
                "meet": meet_no,
                "venue": venue,
                "country": country,
                "races": races,
                "race_meta_by_no": race_meta_by_no,
            }
            
        except Exception as e:
            self.logger.error(f"Error fetching meeting context: {e}")
            return {
                "date": date_str,
                "meet": meet_no,
                "venue": "",
                "country": "",
                "races": [],
                "race_meta_by_no": {}
            }
    
    def fetch_race_data(self, date_str: str, meet_no: int, race_no: int) -> Dict:
        """
        Fetch complete race data including odds and event details
        
        Args:
            date_str: Date in YYYY-MM-DD format
            meet_no: Meeting number
            race_no: Race number
            
        Returns:
            Dictionary with complete race data
        """
        try:
            session = self.make_session()
            
            # 1. Fetch odds data
            race_node = self._fetch_tab_race_node(session, date_str, meet_no, race_no)
            if not race_node:
                raise ValueError("Race node not found")
            
            prices_by_num = self._extract_prices_from_tab_race(race_node)
            race_id = str(race_node.get("id", "")).strip()
            
            if not race_id:
                raise ValueError("Race ID not found")
            
            # 2. Fetch full event data
            event = self._fetch_aff_event(session, race_id)
            if not event or not event.get("data"):
                raise ValueError("Event data not found")
            
            # 3. Merge odds into event data
            self._merge_prices_into_event(event, prices_by_num)
            
            return event
            
        except Exception as e:
            self.logger.error(f"Error fetching race data: {e}")
            raise
    
    def _fetch_tab_race_node(self, session: requests.Session, date_str: str, 
                           meet_no: int, race_no: int) -> Dict:
        """Fetch race node from TAB odds API"""
        try:
            url = f"{self.ODDS_BASE}/{date_str}/{meet_no}/{race_no}"
            response = session.get(url, timeout=self.TIMEOUT)
            response.raise_for_status()
            
            data = response.json()
            if not isinstance(data, dict):
                return {}
            
            meetings = data.get("meetings", [])
            for meeting in meetings:
                for race in meeting.get("races", []):
                    if self._safe_int(race.get("number")) == race_no:
                        return race
                if meeting.get("races"):
                    return meeting["races"][0]
            
            if data.get("races"):
                return data["races"][0]
            
            return {}
            
        except Exception as e:
            self.logger.error(f"Error fetching TAB race node: {e}")
            return {}
    
    def _extract_prices_from_tab_race(self, race_node: Dict) -> Dict:
        """Extract prices from TAB race node"""
        prices = {}
        for entry in race_node.get("entries", []):
            num = self._safe_int(entry.get("number") or entry.get("runner") or entry.get("runner_number"))
            if not num:
                continue
            
            price_record = prices.setdefault(num, {})
            if entry.get("ffwin") is not None:
                price_record["win_fixed"] = entry.get("ffwin")
            if entry.get("ffplc") is not None:
                price_record["place_fixed"] = entry.get("ffplc")
            if entry.get("win") is not None:
                price_record["win_tote"] = entry.get("win")
            if entry.get("plc") is not None:
                price_record["place_tote"] = entry.get("plc")
        
        return prices
    
    def _fetch_aff_event(self, session: requests.Session, race_id: str) -> Dict:
        """Fetch event data from affiliates API"""
        try:
            url = f"{self.AFF_BASE}/events/{race_id}"
            response = session.get(url, headers=self.HEADERS, 
                                 params={"enc": "json"}, timeout=self.TIMEOUT)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.error(f"Error fetching affiliates event: {e}")
            return {}
    
    def _merge_prices_into_event(self, event: Dict, prices_by_number: Dict) -> None:
        """Merge price data into event runners"""
        data = event.get("data", {})
        runners = data.get("runners", [])
        
        for runner in runners:
            num = self._safe_int(runner.get("runner_number") or runner.get("number") or runner.get("barrier"))
            if num and num in prices_by_number:
                runner.setdefault("prices", {}).update(prices_by_number[num])
    
    def _safe_int(self, value, default=None):
        """Safely convert value to integer"""
        try:
            return int(value)
        except (ValueError, TypeError):
            return default
    
    def _safe_float(self, value, default=None):
        """Safely convert value to float"""
        try:
            f = float(value)
            return f if np.isfinite(f) else default
        except (ValueError, TypeError):
            return default
    
    def _pretty_track(self, track_str: str) -> str:
        """Format track condition string"""
        if not track_str:
            return "-"
        
        # e.g. "SOFT5" -> "Soft 5"
        match = re.match(r"([A-Z]+)\s*(\d+)?", str(track_str).strip().upper())
        if not match:
            return str(track_str).title()
        
        word = match.group(1).title()
        num = (" " + match.group(2)) if match.group(2) else ""
        return word + num
    
    def _pretty_weather(self, weather_str: str) -> str:
        """Format weather string"""
        if not weather_str:
            return "-"
        return str(weather_str).replace("_", " ").title()
    
    def _parse_start_to_utc(self, race_data: Dict) -> Optional[datetime]:
        """Parse start time to UTC datetime"""
        # Try different timestamp fields
        for key in ("advertised_start", "start_time", "tote_start_time"):
            timestamp = race_data.get(key)
            if isinstance(timestamp, (int, float)) or (isinstance(timestamp, str) and timestamp.isdigit()):
                try:
                    return datetime.fromtimestamp(int(timestamp), tz=timezone.utc)
                except (ValueError, OSError):
                    continue
        
        # Try string formats
        start_str = (race_data.get("advertised_start_string") or 
                    race_data.get("start_time") or 
                    race_data.get("tote_start_time") or "")
        
        if isinstance(start_str, str) and start_str:
            try:
                if start_str.endswith("Z"):
                    start_str = start_str[:-1] + "+00:00"
                start_str = start_str.replace(" ", "T", 1)
                dt = datetime.fromisoformat(start_str)
                return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                pass
        
        return None
    
    def get_available_dates(self) -> List[str]:
        """Get list of available race dates (placeholder implementation)"""
        # This would typically fetch from an API or database
        # For now, return a few recent dates
        today = datetime.now().date()
        dates = []
        for i in range(7):  # Last 7 days
            date_obj = today - pd.Timedelta(days=i)
            dates.append(date_obj.strftime("%Y-%m-%d"))
        return dates
    
    def get_meetings_for_date(self, date_str: str) -> List[Dict]:
        """Get all meetings for a specific date (placeholder implementation)"""
        # This would fetch from the schedule API
        # For now, return empty list
        return []
    
    def fetch_race_results(self, date_str: str, meet_no: int, race_no: int) -> Optional[Dict]:
        """Fetch race results from TAB API"""
        try:
            import requests
            
            # Use the TAB results API endpoint
            results_url = f"https://json.tab.co.nz/results/{date_str}/{meet_no}/{race_no}"
            
            response = requests.get(results_url, headers=self.HEADERS, timeout=self.TIMEOUT)
            
            if response.status_code != 200:
                self.logger.warning(f"Results API returned status {response.status_code} for {date_str} M{meet_no} R{race_no}")
                return None
            
            data = response.json()
            
            # Extract race results from the JSON structure
            meetings = data.get('meetings', [])
            if meetings:
                meeting = meetings[0]
                races = meeting.get('races', [])
                if races:
                    race = races[0]
                    
                    # Check if race is completed
                    status = race.get('status', '')
                    if status.lower() == 'complete':
                        # Extract placings (top 3)
                        placings = race.get('placings', [])
                        also_ran = race.get('also_ran', [])
                        
                        # Process placings
                        results = []
                        for placing in placings:
                            results.append({
                                'position': placing.get('rank', 0),
                                'number': placing.get('number', ''),
                                'name': placing.get('name', ''),
                                'jockey': placing.get('jockey', ''),
                                'margin': placing.get('margin', ''),
                                'distance': placing.get('distance', ''),
                                'favouritism': placing.get('favouritism', ''),
                                'win_odds': placing.get('win_odds', ''),
                                'place_odds': placing.get('place_odds', '')
                            })
                        
                        # Process also ran (positions 4+)
                        for runner in also_ran:
                            finish_pos = runner.get('finish_position', 0)
                            if finish_pos > 0:  # Only include runners that actually finished
                                results.append({
                                    'position': finish_pos,
                                    'number': runner.get('number', ''),
                                    'name': runner.get('name', ''),
                                    'jockey': runner.get('jockey', ''),
                                    'margin': '',
                                    'distance': runner.get('distance', ''),
                                    'favouritism': '',
                                    'win_odds': '',
                                    'place_odds': ''
                                })
                        
                        # Sort by position
                        results.sort(key=lambda x: x['position'])
                        
                        return {
                            'race_info': {
                                'date': date_str,
                                'meet_no': meet_no,
                                'race_no': race_no,
                                'name': race.get('name', ''),
                                'status': status,
                                'distance': race.get('distance', ''),
                                'class': race.get('class', ''),
                                'stake': race.get('stake', 0)
                            },
                            'results': results,
                            'scratchings': race.get('scratchings', [])
                        }
                    else:
                        self.logger.info(f"Race {date_str} M{meet_no} R{race_no} status: {status} (not complete)")
                        return None
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error fetching race results: {str(e)}")
            return None
