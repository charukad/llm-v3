#!/usr/bin/env python3
"""
Script to register the SuperVisualizationAgent with the system.
This enhances the visualization capabilities of the natural language interface.
"""

from visualization.agent.super_viz_agent import SuperVisualizationAgent

def register_super_viz_agent():
    """
    Create and register the SuperVisualizationAgent.
    Returns the agent instance.
    """
    print("Creating SuperVisualizationAgent...")
    
    # Create the agent with default configuration
    super_viz_agent = SuperVisualizationAgent({
        "storage_dir": "visualizations",
        "use_database": True,
        "default_format": "png",
        "default_dpi": 150,
        "max_width": 1500,
        "max_height": 1000
    })
    
    # Get agent capabilities
    capabilities = super_viz_agent.get_capabilities()
    
    print(f"SuperVisualizationAgent created successfully!")
    print(f"Agent type: {capabilities.get('agent_type')}")
    print(f"Supported visualization types: {len(capabilities.get('supported_types', []))}")
    print(f"Advanced features: {len(capabilities.get('advanced_features', []))}")
    
    return super_viz_agent

if __name__ == "__main__":
    # Register the agent
    agent = register_super_viz_agent()
    
    # Print detailed capabilities
    print("\nDetailed Capabilities:")
    
    capabilities = agent.get_capabilities()
    print(f"Supported visualization types:")
    for viz_type in sorted(capabilities.get('supported_types', [])):
        print(f"  - {viz_type}")
    
    print(f"\nAdvanced features:")
    for feature in sorted(capabilities.get('advanced_features', [])):
        print(f"  - {feature}")
    
    print("\nSuperVisualizationAgent is ready to use with the NLP visualization API endpoint!")
    print("Use the endpoint '/nlp-visualization' with natural language prompts to generate visualizations.") 