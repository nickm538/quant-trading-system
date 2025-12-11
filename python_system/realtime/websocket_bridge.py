#!/usr/bin/env python3.11
"""
WebSocket Bridge for Node.js <-> Python Trading Service Communication
Handles IPC via stdin/stdout using JSON messages
"""

import sys
import json
import logging
import os
from datetime import datetime

# Add parent directory to Python path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from python_system.realtime.trading_service import get_trading_service

# Configure logging to stderr (stdout is for IPC)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)


class WebSocketBridge:
    """Bridge between Node.js WebSocket server and Python trading service"""
    
    def __init__(self):
        self.trading_service = get_trading_service()
        
        # Register callbacks
        self.trading_service.squeeze_monitor.add_squeeze_fire_callback(self._on_squeeze_fire)
        self.trading_service.squeeze_monitor.add_squeeze_update_callback(self._on_squeeze_update)
        self.trading_service.position_tracker.add_exit_callback(self._on_exit_signal)
        self.trading_service.alert_system.add_callback(self._on_alert)
    
    def _send_message(self, message_type: str, data: dict):
        """Send message to Node.js via stdout"""
        try:
            message = {
                'type': message_type,
                'data': data
            }
            print(json.dumps(message), flush=True)
        except Exception as e:
            logger.error(f"Error sending message: {e}")
    
    def _on_squeeze_update(self, squeeze_state: dict):
        """Handle squeeze update from monitor"""
        # Convert datetime to ISO string
        if isinstance(squeeze_state.get('timestamp'), datetime):
            squeeze_state['timestamp'] = squeeze_state['timestamp'].isoformat()
        
        self._send_message('squeeze_update', squeeze_state)
    
    def _on_squeeze_fire(self, squeeze_state: dict):
        """Handle squeeze fire from monitor"""
        # Convert datetime to ISO string
        if isinstance(squeeze_state.get('timestamp'), datetime):
            squeeze_state['timestamp'] = squeeze_state['timestamp'].isoformat()
        
        self._send_message('squeeze_fire', squeeze_state)
    
    def _on_exit_signal(self, position_id: str, position: dict, signal, reason: str):
        """Handle exit signal from position tracker"""
        # Convert datetime to ISO string
        if isinstance(position.get('entry_date'), datetime):
            position['entry_date'] = position['entry_date'].isoformat()
        if isinstance(position.get('expiration'), datetime):
            position['expiration'] = position['expiration'].isoformat()
        
        self._send_message('exit_signal', {
            'position_id': position_id,
            'position': position,
            'signal': signal.value,
            'reason': reason
        })
    
    def _on_alert(self, alert):
        """Handle alert from alert system"""
        self._send_message('alert', alert.to_dict())
    
    def handle_message(self, message: dict):
        """Handle incoming message from Node.js"""
        action = message.get('action')
        
        try:
            if action == 'add_watchlist':
                symbol = message.get('symbol')
                self.trading_service.add_to_watchlist(symbol)
                logger.info(f"Added {symbol} to watchlist")
            
            elif action == 'remove_watchlist':
                symbol = message.get('symbol')
                self.trading_service.remove_from_watchlist(symbol)
                logger.info(f"Removed {symbol} from watchlist")
            
            elif action == 'add_position':
                position_id = message.get('position_id')
                position_data = message.get('position_data')
                
                # Convert ISO strings back to datetime
                if 'entry_date' in position_data and isinstance(position_data['entry_date'], str):
                    position_data['entry_date'] = datetime.fromisoformat(position_data['entry_date'])
                if 'expiration' in position_data and isinstance(position_data['expiration'], str):
                    position_data['expiration'] = datetime.fromisoformat(position_data['expiration'])
                
                self.trading_service.add_position(position_id, position_data)
                logger.info(f"Added position {position_id}")
            
            elif action == 'remove_position':
                position_id = message.get('position_id')
                self.trading_service.remove_position(position_id)
                logger.info(f"Removed position {position_id}")
            
            elif action == 'acknowledge_alert':
                alert_id = message.get('alert_id')
                self.trading_service.acknowledge_alert(alert_id)
                logger.info(f"Acknowledged alert {alert_id}")
            
            elif action == 'get_status':
                status = self.trading_service.get_status()
                reply_to = message.get('reply_to')
                self._send_message('status', {
                    'status': status,
                    'reply_to': reply_to
                })
            
            else:
                logger.warning(f"Unknown action: {action}")
        
        except Exception as e:
            logger.error(f"Error handling message: {e}")
    
    def run(self):
        """Main loop - read from stdin and handle messages"""
        logger.info("WebSocket bridge starting...")
        
        # Start trading service
        self.trading_service.start()
        logger.info("Trading service started")
        
        # Read from stdin
        try:
            for line in sys.stdin:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    message = json.loads(line)
                    self.handle_message(message)
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON: {e}")
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
        
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        
        finally:
            logger.info("Stopping trading service...")
            self.trading_service.stop()
            logger.info("WebSocket bridge stopped")


if __name__ == '__main__':
    bridge = WebSocketBridge()
    bridge.run()
