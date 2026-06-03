"""
Grafana Dashboard Generation and Management
"""

import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class Panel:
    """Grafana panel configuration"""
    id: int
    title: str
    type: str
    targets: List[Dict[str, Any]]
    x: int = 0
    y: int = 0
    w: int = 12
    h: int = 8
    datasource: str = "Prometheus"
    options: Optional[Dict[str, Any]] = None
    field_config: Optional[Dict[str, Any]] = None


@dataclass
class Dashboard:
    """Grafana dashboard configuration"""
    title: str
    description: str
    panels: List[Panel]
    refresh: str = "30s"
    time_from: str = "now-1h"
    time_to: str = "now"
    tags: Optional[List[str]] = None


class GrafanaDashboardGenerator:
    """Generate Grafana dashboards for microservices monitoring"""
    
    def __init__(self):
        self.logger = structlog.get_logger(__name__)
    
    def generate_service_overview_dashboard(self, service_name: str) -> Dict[str, Any]:
        """Generate service overview dashboard"""
        panels = []
        panel_id = 1
        y_position = 0
        
        # Request Rate Panel
        panels.append(Panel(
            id=panel_id,
            title="Request Rate",
            type="stat",
            targets=[{
                "expr": f'rate(http_requests_total{{service="{service_name}"}}[5m])',
                "legendFormat": "Requests/sec",
                "refId": "A"
            }],
            x=0, y=y_position, w=6, h=4,
            field_config={
                "defaults": {
                    "unit": "reqps",
                    "color": {"mode": "thresholds"},
                    "thresholds": {
                        "steps": [
                            {"color": "green", "value": None},
                            {"color": "yellow", "value": 100},
                            {"color": "red", "value": 1000}
                        ]
                    }
                }
            }
        ))
        panel_id += 1
        
        # Error Rate Panel
        panels.append(Panel(
            id=panel_id,
            title="Error Rate",
            type="stat",
            targets=[{
                "expr": f'rate(http_requests_total{{service="{service_name}",status_code=~"5.."}}[5m]) / rate(http_requests_total{{service="{service_name}"}}[5m]) * 100',
                "legendFormat": "Error %",
                "refId": "A"
            }],
            x=6, y=y_position, w=6, h=4,
            field_config={
                "defaults": {
                    "unit": "percent",
                    "color": {"mode": "thresholds"},
                    "thresholds": {
                        "steps": [
                            {"color": "green", "value": None},
                            {"color": "yellow", "value": 1},
                            {"color": "red", "value": 5}
                        ]
                    }
                }
            }
        ))
        panel_id += 1
        y_position += 4
        
        # Response Time Panel
        panels.append(Panel(
            id=panel_id,
            title="Response Time",
            type="timeseries",
            targets=[
                {
                    "expr": f'histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{{service="{service_name}"}}[5m]))',
                    "legendFormat": "95th percentile",
                    "refId": "A"
                },
                {
                    "expr": f'histogram_quantile(0.50, rate(http_request_duration_seconds_bucket{{service="{service_name}"}}[5m]))',
                    "legendFormat": "50th percentile",
                    "refId": "B"
                }
            ],
            x=0, y=y_position, w=12, h=8,
            field_config={
                "defaults": {
                    "unit": "s",
                    "color": {"mode": "palette-classic"}
                }
            }
        ))
        panel_id += 1
        y_position += 8
        
        # Request Volume by Endpoint
        panels.append(Panel(
            id=panel_id,
            title="Request Volume by Endpoint",
            type="timeseries",
            targets=[{
                "expr": f'sum by(endpoint) (rate(http_requests_total{{service="{service_name}"}}[5m]))',
                "legendFormat": "{{endpoint}}",
                "refId": "A"
            }],
            x=0, y=y_position, w=12, h=8,
            field_config={
                "defaults": {
                    "unit": "reqps",
                    "color": {"mode": "palette-classic"}
                }
            }
        ))
        panel_id += 1
        y_position += 8
        
        dashboard = Dashboard(
            title=f"{service_name} - Service Overview",
            description=f"Overview dashboard for {service_name} microservice",
            panels=panels,
            tags=["microservices", service_name, "overview"]
        )
        
        return self._dashboard_to_json(dashboard)
    
    def generate_business_metrics_dashboard(self, service_name: str) -> Dict[str, Any]:
        """Generate business metrics dashboard"""
        panels = []
        panel_id = 1
        y_position = 0
        
        # Business Events Rate
        panels.append(Panel(
            id=panel_id,
            title="Business Events Rate",
            type="timeseries",
            targets=[{
                "expr": f'sum by(event_type) (rate(business_events_total{{service="{service_name}"}}[5m]))',
                "legendFormat": "{{event_type}}",
                "refId": "A"
            }],
            x=0, y=y_position, w=12, h=8,
            field_config={
                "defaults": {
                    "unit": "ops",
                    "color": {"mode": "palette-classic"}
                }
            }
        ))
        panel_id += 1
        y_position += 8
        
        # Success vs Failure Rate
        panels.append(Panel(
            id=panel_id,
            title="Success vs Failure Rate",
            type="timeseries",
            targets=[
                {
                    "expr": f'sum(rate(business_events_total{{service="{service_name}",status="success"}}[5m]))',
                    "legendFormat": "Success",
                    "refId": "A"
                },
                {
                    "expr": f'sum(rate(business_events_total{{service="{service_name}",status="failure"}}[5m]))',
                    "legendFormat": "Failure",
                    "refId": "B"
                }
            ],
            x=0, y=y_position, w=6, h=8,
            field_config={
                "defaults": {
                    "unit": "ops",
                    "color": {"mode": "palette-classic"}
                }
            }
        ))
        panel_id += 1
        
        # Queue Sizes
        panels.append(Panel(
            id=panel_id,
            title="Queue Sizes",
            type="timeseries",
            targets=[{
                "expr": f'queue_size{{service="{service_name}"}}',
                "legendFormat": "{{queue_name}}",
                "refId": "A"
            }],
            x=6, y=y_position, w=6, h=8,
            field_config={
                "defaults": {
                    "unit": "short",
                    "color": {"mode": "palette-classic"}
                }
            }
        ))
        panel_id += 1
        y_position += 8
        
        dashboard = Dashboard(
            title=f"{service_name} - Business Metrics",
            description=f"Business metrics dashboard for {service_name} microservice",
            panels=panels,
            tags=["microservices", service_name, "business"]
        )
        
        return self._dashboard_to_json(dashboard)
    
    def generate_system_resources_dashboard(self, service_name: str) -> Dict[str, Any]:
        """Generate system resources dashboard"""
        panels = []
        panel_id = 1
        y_position = 0
        
        # CPU Usage
        panels.append(Panel(
            id=panel_id,
            title="CPU Usage",
            type="stat",
            targets=[{
                "expr": f'cpu_usage_percent{{service="{service_name}"}}',
                "legendFormat": "CPU %",
                "refId": "A"
            }],
            x=0, y=y_position, w=3, h=4,
            field_config={
                "defaults": {
                    "unit": "percent",
                    "color": {"mode": "thresholds"},
                    "thresholds": {
                        "steps": [
                            {"color": "green", "value": None},
                            {"color": "yellow", "value": 70},
                            {"color": "red", "value": 90}
                        ]
                    }
                }
            }
        ))
        panel_id += 1
        
        # Memory Usage
        panels.append(Panel(
            id=panel_id,
            title="Memory Usage",
            type="stat",
            targets=[{
                "expr": f'memory_usage_bytes{{service="{service_name}"}} / 1024 / 1024 / 1024',
                "legendFormat": "Memory GB",
                "refId": "A"
            }],
            x=3, y=y_position, w=3, h=4,
            field_config={
                "defaults": {
                    "unit": "bytes",
                    "color": {"mode": "thresholds"},
                    "thresholds": {
                        "steps": [
                            {"color": "green", "value": None},
                            {"color": "yellow", "value": 1073741824},  # 1GB
                            {"color": "red", "value": 2147483648}      # 2GB
                        ]
                    }
                }
            }
        ))
        panel_id += 1
        
        # Active Connections
        panels.append(Panel(
            id=panel_id,
            title="Active Connections",
            type="stat",
            targets=[{
                "expr": f'active_connections{{service="{service_name}"}}',
                "legendFormat": "Connections",
                "refId": "A"
            }],
            x=6, y=y_position, w=3, h=4,
            field_config={
                "defaults": {
                    "unit": "short",
                    "color": {"mode": "thresholds"},
                    "thresholds": {
                        "steps": [
                            {"color": "green", "value": None},
                            {"color": "yellow", "value": 100},
                            {"color": "red", "value": 1000}
                        ]
                    }
                }
            }
        ))
        panel_id += 1
        
        # Database Connections
        panels.append(Panel(
            id=panel_id,
            title="DB Connections",
            type="stat",
            targets=[{
                "expr": f'sum(database_connections_active{{service="{service_name}"}}) by (database)',
                "legendFormat": "{{database}}",
                "refId": "A"
            }],
            x=9, y=y_position, w=3, h=4,
            field_config={
                "defaults": {
                    "unit": "short",
                    "color": {"mode": "thresholds"},
                    "thresholds": {
                        "steps": [
                            {"color": "green", "value": None},
                            {"color": "yellow", "value": 10},
                            {"color": "red", "value": 50}
                        ]
                    }
                }
            }
        ))
        panel_id += 1
        y_position += 4
        
        # Resource Usage Over Time
        panels.append(Panel(
            id=panel_id,
            title="Resource Usage Over Time",
            type="timeseries",
            targets=[
                {
                    "expr": f'cpu_usage_percent{{service="{service_name}"}}',
                    "legendFormat": "CPU %",
                    "refId": "A"
                },
                {
                    "expr": f'memory_usage_bytes{{service="{service_name}"}} / 1024 / 1024 / 1024',
                    "legendFormat": "Memory GB",
                    "refId": "B"
                }
            ],
            x=0, y=y_position, w=12, h=8,
            field_config={
                "defaults": {
                    "color": {"mode": "palette-classic"}
                }
            }
        ))
        panel_id += 1
        y_position += 8
        
        dashboard = Dashboard(
            title=f"{service_name} - System Resources",
            description=f"System resources dashboard for {service_name} microservice",
            panels=panels,
            tags=["microservices", service_name, "system"]
        )
        
        return self._dashboard_to_json(dashboard)
    
    def generate_alerts_dashboard(self) -> Dict[str, Any]:
        """Generate alerts overview dashboard"""
        panels = []
        panel_id = 1
        y_position = 0
        
        # Active Alerts Count
        panels.append(Panel(
            id=panel_id,
            title="Active Alerts",
            type="stat",
            targets=[{
                "expr": 'sum(up == 0)',  # Example: services that are down
                "legendFormat": "Critical Alerts",
                "refId": "A"
            }],
            x=0, y=y_position, w=4, h=4,
            field_config={
                "defaults": {
                    "unit": "short",
                    "color": {"mode": "thresholds"},
                    "thresholds": {
                        "steps": [
                            {"color": "green", "value": 0},
                            {"color": "red", "value": 1}
                        ]
                    }
                }
            }
        ))
        panel_id += 1
        
        # Service Health Status
        panels.append(Panel(
            id=panel_id,
            title="Service Health",
            type="table",
            targets=[{
                "expr": 'up',
                "legendFormat": "{{instance}}",
                "refId": "A",
                "format": "table"
            }],
            x=4, y=y_position, w=8, h=8,
            options={
                "showHeader": True
            }
        ))
        panel_id += 1
        y_position += 8
        
        dashboard = Dashboard(
            title="System Alerts Overview",
            description="Overview of system alerts and health status",
            panels=panels,
            tags=["alerts", "overview"]
        )
        
        return self._dashboard_to_json(dashboard)
    
    def generate_feature_flags_dashboard(self) -> Dict[str, Any]:
        """Generate feature flags A/B testing dashboard"""
        panels = []
        panel_id = 1
        y_position = 0
        
        # Feature Flag Usage
        panels.append(Panel(
            id=panel_id,
            title="Feature Flag Usage",
            type="timeseries",
            targets=[{
                "expr": 'sum by(flag_name) (rate(feature_flag_checks_total[5m]))',
                "legendFormat": "{{flag_name}}",
                "refId": "A"
            }],
            x=0, y=y_position, w=12, h=8,
            field_config={
                "defaults": {
                    "unit": "ops",
                    "color": {"mode": "palette-classic"}
                }
            }
        ))
        panel_id += 1
        y_position += 8
        
        # A/B Test Conversion Rates
        panels.append(Panel(
            id=panel_id,
            title="A/B Test Conversion Rates",
            type="timeseries",
            targets=[{
                "expr": 'sum by(flag_name, variation) (rate(ab_test_conversions_total[5m]))',
                "legendFormat": "{{flag_name}} - {{variation}}",
                "refId": "A"
            }],
            x=0, y=y_position, w=12, h=8,
            field_config={
                "defaults": {
                    "unit": "percent",
                    "color": {"mode": "palette-classic"}
                }
            }
        ))
        panel_id += 1
        y_position += 8
        
        dashboard = Dashboard(
            title="Feature Flags & A/B Testing",
            description="Feature flags usage and A/B testing metrics",
            panels=panels,
            tags=["feature-flags", "ab-testing"]
        )
        
        return self._dashboard_to_json(dashboard)
    
    def _dashboard_to_json(self, dashboard: Dashboard) -> Dict[str, Any]:
        """Convert dashboard to Grafana JSON format"""
        return {
            "dashboard": {
                "id": None,
                "title": dashboard.title,
                "description": dashboard.description,
                "tags": dashboard.tags or [],
                "timezone": "browser",
                "refresh": dashboard.refresh,
                "time": {
                    "from": dashboard.time_from,
                    "to": dashboard.time_to
                },
                "panels": [self._panel_to_json(panel) for panel in dashboard.panels],
                "templating": {"list": []},
                "annotations": {"list": []},
                "schemaVersion": 30,
                "version": 1,
                "links": []
            },
            "overwrite": True
        }
    
    def _panel_to_json(self, panel: Panel) -> Dict[str, Any]:
        """Convert panel to Grafana JSON format"""
        panel_json = {
            "id": panel.id,
            "title": panel.title,
            "type": panel.type,
            "datasource": {"type": "prometheus", "uid": "prometheus"},
            "targets": panel.targets,
            "gridPos": {
                "x": panel.x,
                "y": panel.y,
                "w": panel.w,
                "h": panel.h
            }
        }
        
        if panel.options:
            panel_json["options"] = panel.options
        
        if panel.field_config:
            panel_json["fieldConfig"] = panel.field_config
        
        return panel_json
    
    def export_dashboard(self, dashboard: Dict[str, Any], filename: str):
        """Export dashboard to JSON file"""
        try:
            with open(filename, 'w') as f:
                json.dump(dashboard, f, indent=2)
            self.logger.info("Dashboard exported", filename=filename)
        except Exception as e:
            self.logger.error("Failed to export dashboard", filename=filename, error=str(e))
    
    def generate_all_dashboards(self, service_names: List[str], output_dir: str = "./dashboards"):
        """Generate all dashboards for given services"""
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate per-service dashboards
        for service_name in service_names:
            # Service overview
            overview_dashboard = self.generate_service_overview_dashboard(service_name)
            self.export_dashboard(
                overview_dashboard, 
                f"{output_dir}/{service_name}_overview.json"
            )
            
            # Business metrics
            business_dashboard = self.generate_business_metrics_dashboard(service_name)
            self.export_dashboard(
                business_dashboard,
                f"{output_dir}/{service_name}_business.json"
            )
            
            # System resources
            resources_dashboard = self.generate_system_resources_dashboard(service_name)
            self.export_dashboard(
                resources_dashboard,
                f"{output_dir}/{service_name}_resources.json"
            )
        
        # Generate global dashboards
        alerts_dashboard = self.generate_alerts_dashboard()
        self.export_dashboard(alerts_dashboard, f"{output_dir}/alerts_overview.json")
        
        feature_flags_dashboard = self.generate_feature_flags_dashboard()
        self.export_dashboard(feature_flags_dashboard, f"{output_dir}/feature_flags.json")
        
        self.logger.info("All dashboards generated", output_dir=output_dir)