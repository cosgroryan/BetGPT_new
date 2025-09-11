#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Lightweight TAB data fetcher for web app
Extracted from NEW_racing_GUI.py without GUI dependencies
"""

import json
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# API endpoints
SCHED_BASE = "https://json.tab.co.nz/schedule"
ODDS_BASE = "https://json.tab.co.nz/odds"
AFF_BASE = "https://api.tab.co.nz/affiliates/v1/racing"

# Headers and timeout
HEADERS = {
    "Accept": "application/json",
    "User-Agent": "BetGPT-WebApp/1.0"
}
TIMEOUT = 15

def make_session() -> requests.Session:
    """Create a requests session with retry strategy"""
    session = requests.Session()
    
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    return session

def _safe_int(value, default=None):
    """Safely convert value to integer"""
    try:
        return int(value)
    except (ValueError, TypeError):
        return default

def _safe_float(value, default=None):
    """Safely convert value to float"""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def _pretty_track(s: str) -> str:
    """Format track condition nicely"""
    if not s: 
        return "-"
    # e.g. "SOFT5" -> "Soft 5"
    m = re.match(r"([A-Z]+)\s*(\d+)?", str(s).strip().upper())
    if not m: 
        return str(s).title()
    word = m.group(1).title()
    num = (" " + m.group(2)) if m.group(2) else ""
    return word + num

def _pretty_weather(s: str) -> str:
    """Format weather nicely"""
    return "-" if not s else str(s).replace("_", " ").title()

def fetch_meeting_context(session: requests.Session, date_str: str, meetno: int) -> Dict:
    """
    Pull the meeting card + races so we can show a header and build race buttons.
    Uses: https://json.tab.co.nz/schedule/{date}/{meet}/1
    Returns:
      {
        "date": ..., "meet": meetno,
        "venue": "...", "country": "...",
        "races": [1,2,...],
        "race_meta_by_no": { race_no: {"track": "...", "weather": "...", "id": "..."} }
      }
    """
    url = f"{SCHED_BASE}/{date_str}/{meetno}/1"
    r = session.get(url, headers=HEADERS, timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json() if isinstance(r.json(), dict) else {}

    mtgs = (data.get("meetings") or [])
    if not mtgs:
        return {"date": date_str, "meet": meetno, "venue": "", "country": "", "races": [], "race_meta_by_no": {}}

    m = mtgs[0]
    venue = m.get("venue") or m.get("name") or ""
    country = (m.get("country") or "").upper()
    races = []
    race_meta_by_no = {}
    
    for rc in (m.get("races") or []):
        try:
            rn = int(rc.get("number"))
            races.append(rn)
            race_meta_by_no[rn] = {
                "id": rc.get("id"),
                "track": _pretty_track(rc.get("track")),
                "weather": _pretty_weather(rc.get("weather")),
                "name": rc.get("name") or "",
            }
        except Exception:
            pass

    races.sort()
    return {
        "date": date_str,
        "meet": meetno,
        "venue": venue,
        "country": country,
        "races": races,
        "race_meta_by_no": race_meta_by_no,
    }

def fetch_tab_race_node(session: requests.Session, date_str: str, meetno: int, raceno: int) -> Dict:
    """json.tab.co.nz/odds/{date}/{meet}/{race} -> race node with entries and often id."""
    url = f"{ODDS_BASE}/{date_str}/{meetno}/{raceno}"
    r = session.get(url, timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, dict):
        return {}
    
    mtgs = data.get("meetings") or []
    for mtg in mtgs:
        for race in mtg.get("races", []) or []:
            if _safe_int(race.get("number")) == raceno:
                return race
        if mtg.get("races"):
            return mtg["races"][0]
    if data.get("races"):
        return data["races"][0]
    return {}

def extract_prices_from_tab_race(race_node: Dict) -> Dict[int, Dict[str, float]]:
    """Extract prices from race node: number -> {win_fixed, place_fixed, win_tote, place_tote}"""
    out = {}
    for e in race_node.get("entries") or []:
        num = _safe_int(e.get("number") or e.get("runner") or e.get("runner_number"))
        if not num: 
            continue
        rec = out.setdefault(num, {})
        if e.get("ffwin") is not None:  
            rec["win_fixed"] = e.get("ffwin")
        if e.get("ffplc") is not None:  
            rec["place_fixed"] = e.get("ffplc")
        if e.get("win") is not None:   
            rec["win_tote"] = e.get("win")
        if e.get("plc") is not None:   
            rec["place_tote"] = e.get("plc")
    return out

def fetch_aff_event(session: requests.Session, race_id: str) -> Dict:
    """Fetch event data from affiliates API"""
    url = f"{AFF_BASE}/events/{race_id}"
    r = session.get(url, headers=HEADERS, params={"enc": "json"}, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()

def merge_prices_into_event(payload: Dict, prices_by_number: Dict[int, Dict[str, float]]) -> None:
    """Attach .prices to each runner, by runner_number."""
    data = payload.get("data") or {}
    runners = data.get("runners") or []
    for r in runners:
        num = _safe_int(r.get("runner_number") or r.get("number") or r.get("barrier"))
        if num and num in prices_by_number:
            r.setdefault("prices", {}).update(prices_by_number[num])

def fetch_race_data(session: requests.Session, date_str: str, meetno: int, raceno: int) -> Optional[Dict]:
    """
    Main function to fetch complete race data
    Returns the event data with merged prices
    """
    try:
        # Get meeting context first
        meeting_context = fetch_meeting_context(session, date_str, meetno)
        
        # Get race node with odds
        race_node = fetch_tab_race_node(session, date_str, meetno, raceno)
        if not race_node:
            return None
            
        # Extract prices
        prices_by_number = extract_prices_from_tab_race(race_node)
        
        # Get race ID for affiliates API
        race_id = race_node.get("id")
        if not race_id:
            # Try to get from meeting context
            race_meta = meeting_context.get("race_meta_by_no", {}).get(raceno, {})
            race_id = race_meta.get("id")
        
        if race_id:
            # Fetch detailed event data
            event_data = fetch_aff_event(session, race_id)
            # Merge prices into event data
            merge_prices_into_event(event_data, prices_by_number)
            return event_data
        else:
            # Fallback: create minimal event data from race node
            return {
                "data": {
                    "race": {
                        "id": race_node.get("id"),
                        "number": raceno,
                        "name": race_node.get("name", ""),
                        "distance": race_node.get("distance", ""),
                        "track_condition": race_node.get("track", ""),
                        "weather": race_node.get("weather", ""),
                        "start_time": race_node.get("start_time", ""),
                        "positions_paid": race_node.get("positions_paid", 3),
                        "meeting_country": meeting_context.get("country", ""),
                        "display_meeting_name": meeting_context.get("venue", "")
                    },
                    "runners": []
                }
            }
            
    except Exception as e:
        print(f"Error fetching race data: {e}")
        return None

def fetch_race_results(session: requests.Session, date_str: str, meetno: int, raceno: int) -> Optional[Dict]:
    """
    Fetch race results from TAB API and merge with final field odds
    """
    try:
        # 1. Fetch results data
        results_url = f"https://json.tab.co.nz/results/{date_str}/{meetno}/{raceno}"
        response = session.get(results_url, headers=HEADERS, timeout=TIMEOUT)
        
        if response.status_code != 200:
            return None
            
        data = response.json()
        
        # 2. Fetch final field odds from odds API
        odds_url = f"{ODDS_BASE}/{date_str}/{meetno}/{raceno}"
        odds_response = session.get(odds_url, headers=HEADERS, timeout=TIMEOUT)
        final_odds = {}
        
        if odds_response.status_code == 200:
            odds_data = odds_response.json()
            # Extract final odds from the odds API response
            meetings = odds_data.get('meetings', [])
            for meeting in meetings:
                races = meeting.get('races', [])
                for race in races:
                    if race.get('number') == raceno:
                        entries = race.get('entries', [])
                        for entry in entries:
                            runner_num = entry.get('number')
                            if runner_num:
                                final_odds[runner_num] = {
                                    'win_fixed': entry.get('ffwin'),
                                    'place_fixed': entry.get('ffplc'),
                                    'win_tote': entry.get('win'),
                                    'place_tote': entry.get('plc')
                                }
        
        # 3. Extract race results from the JSON structure
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
                        runner_num = placing.get('number', '')
                        odds = final_odds.get(runner_num, {})
                        
                        results.append({
                            'position': placing.get('rank', 0),
                            'number': runner_num,
                            'name': placing.get('name', ''),
                            'jockey': placing.get('jockey', ''),
                            'margin': placing.get('margin', ''),
                            'distance': placing.get('distance', ''),
                            'favouritism': placing.get('favouritism', ''),
                            'win_odds': odds.get('win_fixed', ''),
                            'place_odds': odds.get('place_fixed', '')
                        })
                    
                    # Process also ran (positions 4+)
                    for runner in also_ran:
                        finish_pos = runner.get('finish_position', 0)
                        if finish_pos > 0:  # Only include runners that actually finished
                            runner_num = runner.get('number', '')
                            odds = final_odds.get(runner_num, {})
                            
                            results.append({
                                'position': finish_pos,
                                'number': runner_num,
                                'name': runner.get('name', ''),
                                'jockey': runner.get('jockey', ''),
                                'margin': '',
                                'distance': runner.get('distance', ''),
                                'favouritism': '',
                                'win_odds': odds.get('win_fixed', ''),
                                'place_odds': odds.get('place_fixed', '')
                            })
                    
                    # Sort by position
                    results.sort(key=lambda x: x['position'])
                    
                    return {
                        'race_info': {
                            'date': date_str,
                            'meet_no': meetno,
                            'race_no': raceno,
                            'name': race.get('name', ''),
                            'status': status,
                            'distance': race.get('distance', ''),
                            'class': race.get('class', ''),
                            'stake': race.get('stake', 0)
                        },
                        'results': results,
                        'scratchings': race.get('scratchings', [])
                    }
        
        return None
        
    except Exception as e:
        print(f"Error fetching race results: {e}")
        return None
