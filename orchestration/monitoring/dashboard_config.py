"""
Performance monitoring dashboard configuration.
Defines Grafana dashboard panels and metrics for system monitoring.
"""
import json
from typing import Dict, Any, List

# Grafana dashboard configuration
DASHBOARD_CONFIG = {
    "title": "Mathematical Multimodal LLM System Dashboard",
    "uid": "math-llm-system",
    "time": {
        "from": "now-1h",
        "to": "now"
    },
    "refresh": "10s",
    "schemaVersion": 27,
    "version": 1,
    "editable": True,
    "panels": [
        # System Overview Row
        {
            "type": "row",
            "title": "System Overview",
            "collapsed": False,
            "id": 1
        },
        
        # CPU Usage Panel
        {
            "type": "graph",
            "title": "CPU Usage",
            "id": 2,
            "gridPos": {
                "h": 8,
                "w": 12,
                "x": 0,
                "y": 1
            },
            "targets": [
                {
                    "expr": "system_cpu_usage_percent",
                    "legendFormat": "System CPU",
                    "refId": "A"
                },
                {
                    "expr": "process_cpu_usage_percent",
                    "legendFormat": "Process CPU",
                    "refId": "B"
                }
            ],
            "yaxes": [
                {
                    "format": "percent",
                    "min": 0,
                    "max": 100
                },
                {
                    "format": "short",
                    "show": False
                }
            ],
            "thresholds": [
                {
                    "value": 70,
                    "op": "gt",
                    "fill": True,
                    "line": True,
                    "colorMode": "warning"
                },
                {
                    "value": 90,
                    "op": "gt",
                    "fill": True,
                    "line": True,
                    "colorMode": "critical"
                }
            ]
        },
        
        # Memory Usage Panel
        {
            "type": "graph",
            "title": "Memory Usage",
            "id": 3,
            "gridPos": {
                "h": 8,
                "w": 12,
                "x": 12,
                "y": 1
            },
            "targets": [
                {
                    "expr": "system_memory_usage_percent",
                    "legendFormat": "System Memory",
                    "refId": "A"
                },
                {
                    "expr": "process_memory_usage_percent",
                    "legendFormat": "Process Memory",
                    "refId": "B"
                }
            ],
            "yaxes": [
                {
                    "format": "percent",
                    "min": 0,
                    "max": 100
                },
                {
                    "format": "short",
                    "show": False
                }
            ],
            "thresholds": [
                {
                    "value": 80,
                    "op": "gt",
                    "fill": True,
                    "line": True,
                    "colorMode": "warning"
                },
                {
                    "value": 95,
                    "op": "gt",
                    "fill": True,
                    "line": True,
                    "colorMode": "critical"
                }
            ]
        },
        
        # GPU Usage Panel (if available)
        {
            "type": "graph",
            "title": "GPU Usage",
            "id": 4,
            "gridPos": {
                "h": 8,
                "w": 12,
                "x": 0,
                "y": 9
            },
            "targets": [
                {
                    "expr": "gpu_memory_usage_percent{device='0'}",
                    "legendFormat": "GPU 0 Memory",
                    "refId": "A"
                },
                {
                    "expr": "gpu_utilization_percent{device='0'}",
                    "legendFormat": "GPU 0 Utilization",
                    "refId": "B"
                }
            ],
            "yaxes": [
                {
                    "format": "percent",
                    "min": 0,
                    "max": 100
                },
                {
                    "format": "short",
                    "show": False
                }
            ]
        },
        
        # Active Tasks Panel
        {
            "type": "graph",
            "title": "Active Tasks",
            "id": 5,
            "gridPos": {
                "h": 8,
                "w": 12,
                "x": 12,
                "y": 9
            },
            "targets": [
                {
                    "expr": "active_tasks_count",
                    "legendFormat": "Active Tasks",
                    "refId": "A"
                },
                {
                    "expr": "queued_tasks_count",
                    "legendFormat": "Queued Tasks",
                    "refId": "B"
                }
            ],
            "yaxes": [
                {
                    "format": "short",
                    "min": 0
                },
                {
                    "format": "short",
                    "show": False
                }
            ]
        },
        
        # API Performance Row
        {
            "type": "row",
            "title": "API Performance",
            "collapsed": False,
            "id": 6
        },
        
        # Request Rate Panel
        {
            "type": "graph",
            "title": "API Request Rate",
            "id": 7,
            "gridPos": {
                "h": 8,
                "w": 12,
                "x": 0,
                "y": 18
            },
            "targets": [
                {
                    "expr": "rate(api_request_count[1m])",
                    "legendFormat": "Requests per Second",
                    "refId": "A"
                }
            ],
            "yaxes": [
                {
                    "format": "reqps",
                    "min": 0
                },
                {
                    "format": "short",
                    "show": False
                }
            ]
        },
        
        # Response Time Panel
        {
            "type": "graph",
            "title": "API Response Time",
            "id": 8,
            "gridPos": {
                "h": 8,
                "w": 12,
                "x": 12,
                "y": 18
            },
            "targets": [
                {
                    "expr": "api_response_time_seconds{quantile='0.5'}",
                    "legendFormat": "Median",
                    "refId": "A"
                },
                {
                    "expr": "api_response_time_seconds{quantile='0.95'}",
                    "legendFormat": "95th Percentile",
                    "refId": "B"
                },
                {
                    "expr": "api_response_time_seconds{quantile='0.99'}",
                    "legendFormat": "99th Percentile",
                    "refId": "C"
                }
            ],
            "yaxes": [
                {
                    "format": "s",
                    "min": 0
                },
                {
                    "format": "short",
                    "show": False
                }
            ]
        },
        
        # Endpoint Performance Panel
        {
            "type": "graph",
            "title": "Endpoint Response Time",
            "id": 9,
            "gridPos": {
                "h": 8,
                "w": 24,
                "x": 0,
                "y": 26
            },
            "targets": [
                {
                    "expr": "api_endpoint_response_time_seconds{quantile='0.95'}",
                    "legendFormat": "{{endpoint}}",
                    "refId": "A"
                }
            ],
            "yaxes": [
                {
                    "format": "s",
                    "min": 0
                },
                {
                    "format": "short",
                    "show": False
                }
            ]
        },
        
        # Error Rate Panel
        {
            "type": "graph",
            "title": "API Error Rate",
            "id": 10,
            "gridPos": {
                "h": 8,
                "w": 12,
                "x": 0,
                "y": 34
            },
            "targets": [
                {
                    "expr": "rate(api_error_count[1m])",
                    "legendFormat": "Errors per Second",
                    "refId": "A"
                }
            ],
            "yaxes": [
                {
                    "format": "short",
                    "min": 0
                },
                {
                    "format": "short",
                    "show": False
                }
            ],
            "thresholds": [
                {
                    "value": 1,
                    "op": "gt",
                    "fill": True,
                    "line": True,
                    "colorMode": "warning"
                },
                {
                    "value": 5,
                    "op": "gt",
                    "fill": True,
                    "line": True,
                    "colorMode": "critical"
                }
            ]
        },
        
        # Success Rate Panel
        {
            "type": "gauge",
            "title": "API Success Rate",
            "id": 11,
            "gridPos": {
                "h": 8,
                "w": 12,
                "x": 12,
                "y": 34
            },
            "targets": [
                {
                    "expr": "api_success_rate_percent",
                    "refId": "A"
                }
            ],
            "options": {
                "fieldOptions": {
                    "calcs": ["last"],
                    "defaults": {
                        "min": 0,
                        "max": 100,
                        "unit": "percent"
                    },
                    "thresholds": [
                        {
                            "value": 90,
                            "color": "red"
                        },
                        {
                            "value": 95,
                            "color": "yellow"
                        },
                        {
                            "value": 99,
                            "color": "green"
                        }
                    ]
                }
            }
        },
        
        # Cache and Database Row
        {
            "type": "row",
            "title": "Cache and Database Performance",
            "collapsed": False,
            "id": 12
        },
        
        # Cache Hit Rate Panel
        {
            "type": "graph",
            "title": "Cache Hit Rate",
            "id": 13,
            "gridPos": {
                "h": 8,
                "w": 12,
                "x": 0,
                "y": 43
            },
            "targets": [
                {
                    "expr": "cache_hit_rate_percent",
                    "legendFormat": "Hit Rate",
                    "refId": "A"
                }
            ],
            "yaxes": [
                {
                    "format": "percent",
                    "min": 0,
                    "max": 100
                },
                {
                    "format": "short",
                    "show": False
                }
            ]
        },
        
        # Database Query Time Panel
        {
            "type": "graph",
            "title": "Database Query Time",
            "id": 14,
            "gridPos": {
                "h": 8,
                "w": 12,
                "x": 12,
                "y": 43
            },
            "targets": [
                {
                    "expr": "database_query_time_seconds{quantile='0.5'}",
                    "legendFormat": "Median",
                    "refId": "A"
                },
                {
                    "expr": "database_query_time_seconds{quantile='0.95'}",
                    "legendFormat": "95th Percentile",
                    "refId": "B"
                }
            ],
            "yaxes": [
                {
                    "format": "s",
                    "min": 0
                },
                {
                    "format": "short",
                    "show": False
                }
            ]
        },
        
        # Cache Operations Panel
        {
            "type": "graph",
            "title": "Cache Operations",
            "id": 15,
            "gridPos": {
                "h": 8,
                "w": 12,
                "x": 0,
                "y": 51
            },
            "targets": [
                {
                    "expr": "rate(cache_hit_count[1m])",
                    "legendFormat": "Hits",
                    "refId": "A"
                },
                {
                    "expr": "rate(cache_miss_count[1m])",
                    "legendFormat": "Misses",
                    "refId": "B"
                },
                {
                    "expr": "rate(cache_store_count[1m])",
                    "legendFormat": "Stores",
                    "refId": "C"
                }
            ],
            "yaxes": [
                {
                    "format": "ops",
                    "min": 0
                },
                {
                    "format": "short",
                    "show": False
                }
            ]
        },
        
        # Database Operations Panel
        {
            "type": "graph",
            "title": "Database Operations",
            "id": 16,
            "gridPos": {
                "h": 8,
                "w": 12,
                "x": 12,
                "y": 51
            },
            "targets": [
                {
                    "expr": "rate(database_read_count[1m])",
                    "legendFormat": "Reads",
                    "refId": "A"
                },
                {
                    "expr": "rate(database_write_count[1m])",
                    "legendFormat": "Writes",
                    "refId": "B"
                },
                {
                    "expr": "rate(database_query_count[1m])",
                    "legendFormat": "Queries",
                    "refId": "C"
                }
            ],
            "yaxes": [
                {
                    "format": "ops",
                    "min": 0
                },
                {
                    "format": "short",
                    "show": False
                }
            ]
        },
        
        # Component Performance Row
        {
            "type": "row",
            "title": "Component Performance",
            "collapsed": False,
            "id": 17
        },
        
        # Model Inference Performance Panel
        {
            "type": "graph",
            "title": "Model Inference Performance",
            "id": 18,
            "gridPos": {
                "h": 8,
                "w": 12,
                "x": 0,
                "y": 60
            },
            "targets": [
                {
                    "expr": "model_inference_time_seconds{quantile='0.5'}",
                    "legendFormat": "Median",
                    "refId": "A"
                },
                {
                    "expr": "model_inference_time_seconds{quantile='0.95'}",
                    "legendFormat": "95th Percentile",
                    "refId": "B"
                }
            ],
            "yaxes": [
                {
                    "format": "s",
                    "min": 0
                },
                {
                    "format": "short",
                    "show": False
                }
            ]
        },
        
        # Symbolic Computation Performance Panel
        {
            "type": "graph",
            "title": "Symbolic Computation Performance",
            "id": 19,
            "gridPos": {
                "h": 8,
                "w": 12,
                "x": 12,
                "y": 60
            },
            "targets": [
                {
                    "expr": "symbolic_computation_time_seconds{operation='solve',quantile='0.5'}",
                    "legendFormat": "Solve (Median)",
                    "refId": "A"
                },
                {
                    "expr": "symbolic_computation_time_seconds{operation='differentiate',quantile='0.5'}",
                    "legendFormat": "Differentiate (Median)",
                    "refId": "B"
                },
                {
                    "expr": "symbolic_computation_time_seconds{operation='integrate',quantile='0.5'}",
                    "legendFormat": "Integrate (Median)",
                    "refId": "C"
                }
            ],
            "yaxes": [
                {
                    "format": "s",
                    "min": 0
                },
                {
                    "format": "short",
                    "show": False
                }
            ]
        },
        
        # Handwriting Recognition Performance Panel
        {
            "type": "graph",
            "title": "Handwriting Recognition Performance",
            "id": 20,
            "gridPos": {
                "h": 8,
                "w": 12,
                "x": 0,
                "y": 68
            },
            "targets": [
                {
                    "expr": "ocr_processing_time_seconds{quantile='0.5'}",
                    "legendFormat": "OCR Processing (Median)",
                    "refId": "A"
                },
                {
                    "expr": "ocr_processing_time_seconds{quantile='0.95'}",
                    "legendFormat": "OCR Processing (95th)",
                    "refId": "B"
                },
                {
                    "expr": "latex_generation_time_seconds{quantile='0.5'}",
                    "legendFormat": "LaTeX Generation (Median)",
                    "refId": "C"
                }
            ],
            "yaxes": [
                {
                    "format": "s",
                    "min": 0
                },
                {
                    "format": "short",
                    "show": False
                }
            ]
        },
        
        # Visualization Performance Panel
        {
            "type": "graph",
            "title": "Visualization Performance",
            "id": 21,
            "gridPos": {
                "h": 8,
                "w": 12,
                "x": 12,
                "y": 68
            },
            "targets": [
                {
                    "expr": "visualization_generation_time_seconds{type='2d',quantile='0.5'}",
                    "legendFormat": "2D Plot (Median)",
                    "refId": "A"
                },
                {
                    "expr": "visualization_generation_time_seconds{type='3d',quantile='0.5'}",
                    "legendFormat": "3D Plot (Median)",
                    "refId": "B"
                },
                {
                    "expr": "visualization_generation_time_seconds{type='statistical',quantile='0.5'}",
                    "legendFormat": "Statistical Plot (Median)",
                    "refId": "C"
                }
            ],
            "yaxes": [
                {
                    "format": "s",
                    "min": 0
                },
                {
                    "format": "short",
                    "show": False
                }
            ]
        },
        
        # Search Component Performance Panel
        {
            "type": "graph",
            "title": "Search Component Performance",
            "id": 22,
            "gridPos": {
                "h": 8,
                "w": 12,
                "x": 0,
                "y": 76
            },
            "targets": [
                {
                    "expr": "search_query_time_seconds{quantile='0.5'}",
                    "legendFormat": "Search Query (Median)",
                    "refId": "A"
                },
                {
                    "expr": "search_processing_time_seconds{quantile='0.5'}",
                    "legendFormat": "Result Processing (Median)",
                    "refId": "B"
                }
            ],
            "yaxes": [
                {
                    "format": "s",
                    "min": 0
                },
                {
                    "format": "short",
                    "show": False
                }
            ]
        },
        
        # Message Bus Performance Panel
        {
            "type": "graph",
            "title": "Message Bus Performance",
            "id": 23,
            "gridPos": {
                "h": 8,
                "w": 12,
                "x": 12,
                "y": 76
            },
            "targets": [
                {
                    "expr": "message_delivery_time_seconds{quantile='0.5'}",
                    "legendFormat": "Message Delivery (Median)",
                    "refId": "A"
                },
                {
                    "expr": "message_processing_time_seconds{quantile='0.5'}",
                    "legendFormat": "Message Processing (Median)",
                    "refId": "B"
                }
            ],
            "yaxes": [
                {
                    "format": "s",
                    "min": 0
                },
                {
                    "format": "short",
                    "show": False
                }
            ]
        },
        
        # Workflow Performance Row
        {
            "type": "row",
            "title": "Workflow Performance",
            "collapsed": False,
            "id": 24
        },
        
        # End-to-End Workflow Performance Panel
        {
            "type": "graph",
            "title": "End-to-End Workflow Performance",
            "id": 25,
            "gridPos": {
                "h": 8,
                "w": 24,
                "x": 0,
                "y": 85
            },
            "targets": [
                {
                    "expr": "workflow_time_seconds{workflow='math_problem_solving',quantile='0.5'}",
                    "legendFormat": "Math Problem (Median)",
                    "refId": "A"
                },
                {
                    "expr": "workflow_time_seconds{workflow='handwriting_recognition',quantile='0.5'}",
                    "legendFormat": "Handwriting Recognition (Median)",
                    "refId": "B"
                },
                {
                    "expr": "workflow_time_seconds{workflow='visualization',quantile='0.5'}",
                    "legendFormat": "Visualization (Median)",
                    "refId": "C"
                },
                {
                    "expr": "workflow_time_seconds{workflow='search_integration',quantile='0.5'}",
                    "legendFormat": "Search Integration (Median)",
                    "refId": "D"
                }
            ],
            "yaxes": [
                {
                    "format": "s",
                    "min": 0
                },
                {
                    "format": "short",
                    "show": False
                }
            ]
        },
        
        # Workflow Step Breakdown Panel
        {
            "type": "graph",
            "title": "Math Problem Workflow Step Breakdown",
            "id": 26,
            "gridPos": {
                "h": 8,
                "w": 24,
                "x": 0,
                "y": 93
            },
            "targets": [
                {
                    "expr": "workflow_step_time_seconds{workflow='math_problem_solving',step='query_understanding',quantile='0.5'}",
                    "legendFormat": "Query Understanding",
                    "refId": "A"
                },
                {
                    "expr": "workflow_step_time_seconds{workflow='math_problem_solving',step='symbolic_computation',quantile='0.5'}",
                    "legendFormat": "Symbolic Computation",
                    "refId": "B"
                },
                {
                    "expr": "workflow_step_time_seconds{workflow='math_problem_solving',step='solution_generation',quantile='0.5'}",
                    "legendFormat": "Solution Generation",
                    "refId": "C"
                },
                {
                    "expr": "workflow_step_time_seconds{workflow='math_problem_solving',step='visualization',quantile='0.5'}",
                    "legendFormat": "Visualization",
                    "refId": "D"
                },
                {
                    "expr": "workflow_step_time_seconds{workflow='math_problem_solving',step='response_formatting',quantile='0.5'}",
                    "legendFormat": "Response Formatting",
                    "refId": "E"
                }
            ],
            "yaxes": [
                {
                    "format": "s",
                    "min": 0
                },
                {
                    "format": "short",
                    "show": False
                }
            ]
        },
        
        # Optimization Metrics Row
        {
            "type": "row",
            "title": "Optimization Metrics",
            "collapsed": False,
            "id": 27
        },
        
        # Request Batching Performance Panel
        {
            "type": "graph",
            "title": "Request Batching Performance",
            "id": 28,
            "gridPos": {
                "h": 8,
                "w": 12,
                "x": 0,
                "y": 102
            },
            "targets": [
                {
                    "expr": "batch_size_average",
                    "legendFormat": "Avg Batch Size",
                    "refId": "A"
                },
                {
                    "expr": "batch_processing_time_seconds{quantile='0.5'}",
                    "legendFormat": "Batch Processing Time (Median)",
                    "refId": "B"
                }
            ],
            "yaxes": [
                {
                    "format": "short",
                    "min": 0
                },
                {
                    "format": "short",
                    "show": False
                }
            ]
        },
        
        # Parallel Processing Performance Panel
        {
            "type": "graph",
            "title": "Parallel Processing Performance",
            "id": 29,
            "gridPos": {
                "h": 8,
                "w": 12,
                "x": 12,
                "y": 102
            },
            "targets": [
                {
                    "expr": "parallel_tasks_count",
                    "legendFormat": "Parallel Tasks",
                    "refId": "A"
                },
                {
                    "expr": "parallel_processing_speedup",
                    "legendFormat": "Speedup Factor",
                    "refId": "B"
                }
            ],
            "yaxes": [
                {
                    "format": "short",
                    "min": 0
                },
                {
                    "format": "short",
                    "show": False
                }
            ]
        },
        
        # Resource Optimization Panel
        {
            "type": "graph",
            "title": "Resource Optimization",
            "id": 30,
            "gridPos": {
                "h": 8,
                "w": 24,
                "x": 0,
                "y": 110
            },
            "targets": [
                {
                    "expr": "optimized_requests_percent",
                    "legendFormat": "Optimized Requests %",
                    "refId": "A"
                },
                {
                    "expr": "resource_efficiency_score",
                    "legendFormat": "Resource Efficiency Score",
                    "refId": "B"
                },
                {
                    "expr": "optimization_impact_score",
                    "legendFormat": "Optimization Impact Score",
                    "refId": "C"
                }
            ],
            "yaxes": [
                {
                    "format": "percent",
                    "min": 0,
                    "max": 100
                },
                {
                    "format": "short",
                    "show": False
                }
            ]
        }
    ]
}

# Function to get dashboard configuration
def get_dashboard_config() -> Dict[str, Any]:
    """Get the dashboard configuration."""
    return DASHBOARD_CONFIG

# Function to save dashboard configuration to file
def save_dashboard_config(filename: str) -> bool:
    """
    Save dashboard configuration to file.
    
    Args:
        filename: Path to save the configuration
        
    Returns:
        Boolean indicating success
    """
    try:
        with open(filename, 'w') as f:
            json.dump(DASHBOARD_CONFIG, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving dashboard configuration: {e}")
        return False

# Function to get panel configuration by ID
def get_panel_config(panel_id: int) -> Dict[str, Any]:
    """
    Get configuration for a specific panel.
    
    Args:
        panel_id: Panel ID
        
    Returns:
        Panel configuration
    """
    for panel in DASHBOARD_CONFIG["panels"]:
        if panel.get("id") == panel_id:
            return panel
    return None

# Function to get metrics for a specific component
def get_component_metrics(component: str) -> List[str]:
    """
    Get metrics for a specific component.
    
    Args:
        component: Component name
        
    Returns:
        List of metric names
    """
    metrics = []
    for panel in DASHBOARD_CONFIG["panels"]:
        if "targets" in panel:
            for target in panel["targets"]:
                expr = target.get("expr", "")
                if component.lower() in expr.lower():
                    metrics.append(expr)
    return metrics

# Collection of metric specifications for Prometheus
METRICS_SPECS = {
    "system": [
        "system_cpu_usage_percent",
        "system_memory_usage_percent",
        "process_cpu_usage_percent",
        "process_memory_usage_percent"
    ],
    "gpu": [
        "gpu_memory_usage_percent",
        "gpu_utilization_percent"
    ],
    "api": [
        "api_request_count",
        "api_error_count",
        "api_response_time_seconds",
        "api_endpoint_response_time_seconds",
        "api_success_rate_percent"
    ],
    "cache": [
        "cache_hit_count",
        "cache_miss_count",
        "cache_store_count",
        "cache_hit_rate_percent"
    ],
    "database": [
        "database_read_count",
        "database_write_count",
        "database_query_count",
        "database_query_time_seconds"
    ],
    "model": [
        "model_inference_time_seconds"
    ],
    "computation": [
        "symbolic_computation_time_seconds"
    ],
    "ocr": [
        "ocr_processing_time_seconds",
        "latex_generation_time_seconds"
    ],
    "visualization": [
        "visualization_generation_time_seconds"
    ],
    "search": [
        "search_query_time_seconds",
        "search_processing_time_seconds"
    ],
    "messaging": [
        "message_delivery_time_seconds",
        "message_processing_time_seconds"
    ],
    "workflow": [
        "workflow_time_seconds",
        "workflow_step_time_seconds"
    ],
    "optimization": [
        "batch_size_average",
        "batch_processing_time_seconds",
        "parallel_tasks_count",
        "parallel_processing_speedup",
        "optimized_requests_percent",
        "resource_efficiency_score",
        "optimization_impact_score"
    ]
}
