#!/usr/bin/env python3

"""
Enhanced mathematical visualization script.
Supports multiple visualization types.

Usage: 
    python math_viz.py <expression> [--type TYPE] [--range START END]

Examples:
    python math_viz.py "sin(x) * cos(x)"                    # Basic 2D plot
    python math_viz.py "sin(x) * cos(x)" --type derivative  # Plot with derivative
    python math_viz.py "x^3 - 2*x^2 + 1" --type critical    # Plot with critical points
    python math_viz.py "sin(x)" --range -5 5                # Custom range
"""

import os
import sys
import json
import argparse
from visualization.agent.advanced_viz_agent import AdvancedVisualizationAgent

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Visualize mathematical expressions')
    parser.add_argument('expression', type=str, help='Mathematical expression to visualize')
    parser.add_argument('--type', type=str, default='2d', 
                        choices=['2d', 'derivative', 'critical', 'integral', 'taylor'],
                        help='Visualization type')
    parser.add_argument('--range', type=float, nargs=2, default=[-10, 10], metavar=('START', 'END'),
                        help='X-axis range (default: -10 to 10)')
    
    args = parser.parse_args()
    
    # Map command line args to visualization types
    viz_type_map = {
        '2d': 'function_2d',
        'derivative': 'derivative',
        'critical': 'critical_points',
        'integral': 'integral',
        'taylor': 'taylor_series'
    }
    
    # Initialize the advanced visualization agent
    agent = AdvancedVisualizationAgent({
        "storage_dir": "output_visualizations",
        "use_database": False,
        "default_format": "png"
    })
    
    # Create output directory
    os.makedirs("output_visualizations", exist_ok=True)
    
    # Create visualization request
    message = {
        "header": {
            "message_type": "visualization_request"
        },
        "body": {
            "visualization_type": viz_type_map[args.type],
            "parameters": {
                "expression": args.expression,
                "variable": "x",
                "x_range": (args.range[0], args.range[1]),
                "title": f"{args.type.capitalize()} visualization of {args.expression}",
                "save": True
            }
        }
    }
    
    # Process message
    print(f"Creating {args.type} visualization for: {args.expression}")
    print(f"X-range: {args.range[0]} to {args.range[1]}")
    
    result = agent.process_message(message)
    
    if result.get('success'):
        print(f"Visualization saved to: {result.get('file_path')}")
        
        # Open the file using the default application
        if 'file_path' in result and os.path.exists(result['file_path']):
            if sys.platform == 'darwin':  # macOS
                os.system(f"open {result['file_path']}")
            elif sys.platform == 'win32':  # Windows
                os.system(f"start {result['file_path']}")
            else:  # Linux or other
                os.system(f"xdg-open {result['file_path']}")
    else:
        print(f"Error: {result.get('error')}")

if __name__ == "__main__":
    main()