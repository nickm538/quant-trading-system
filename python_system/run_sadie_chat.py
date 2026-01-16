#!/usr/bin/env python3
"""
Runner script for Sadie AI Chatbot
Handles JSON input/output for the Node.js backend
"""

import sys
import json
import os

# Add the python_system directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sadie_ai_engine import SadieAIEngine

def main():
    """Main entry point for the Sadie AI chatbot."""
    try:
        # Parse command line arguments
        if len(sys.argv) < 2:
            print(json.dumps({
                "success": False,
                "message": "Usage: python run_sadie_chat.py <action> [args]",
                "data": {}
            }))
            return
        
        action = sys.argv[1]
        
        # Initialize the engine
        engine = SadieAIEngine()
        
        if action == "chat":
            # Chat action - expects message as argument
            if len(sys.argv) < 3:
                print(json.dumps({
                    "success": False,
                    "message": "Chat action requires a message argument",
                    "data": {}
                }))
                return
            
            message = " ".join(sys.argv[2:])
            result = engine.chat(message)
            print(json.dumps(result, default=str))
        
        elif action == "chat_with_image":
            # Chat with image action - expects message and image paths
            if len(sys.argv) < 4:
                print(json.dumps({
                    "success": False,
                    "message": "chat_with_image action requires message and image paths",
                    "data": {}
                }))
                return
            
            message = sys.argv[2]
            image_paths = sys.argv[3].split(',')
            
            result = engine.chat_with_image(message, image_paths)
            print(json.dumps(result, default=str))
            
        elif action == "analyze":
            # Quick analysis action - expects symbol as argument
            if len(sys.argv) < 3:
                print(json.dumps({
                    "success": False,
                    "message": "Analyze action requires a symbol argument",
                    "data": {}
                }))
                return
            
            symbol = sys.argv[2].upper()
            result = engine.get_quick_analysis(symbol)
            print(json.dumps({
                "success": True,
                "message": f"Analysis complete for {symbol}",
                "data": result
            }, default=str))
            
        elif action == "clear":
            # Clear conversation history
            result = engine.clear_history()
            print(json.dumps(result))
            
        else:
            print(json.dumps({
                "success": False,
                "message": f"Unknown action: {action}. Valid actions: chat, chat_with_image, analyze, clear",
                "data": {}
            }))
            
    except Exception as e:
        print(json.dumps({
            "success": False,
            "message": f"Error: {str(e)}",
            "data": {"error_type": type(e).__name__}
        }))

if __name__ == "__main__":
    main()
