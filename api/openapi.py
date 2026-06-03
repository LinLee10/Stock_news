#!/usr/bin/env python3
"""
OpenAPI documentation generation for Stonk News API
Provides Swagger UI and OpenAPI specification
"""
from flask import Blueprint, jsonify, render_template_string
from config.feature_flags import is_api_endpoints_enabled
import json

openapi_blueprint = Blueprint('openapi', __name__)

# OpenAPI specification
OPENAPI_SPEC = {
    "openapi": "3.0.0",
    "info": {
        "title": "Stonk News API",
        "version": "1.0.0",
        "description": "REST API for financial news analysis and stock recommendations",
        "contact": {
            "name": "Stonk News API Support"
        }
    },
    "servers": [
        {
            "url": "http://localhost:8000/api/v1",
            "description": "Development server"
        }
    ],
    "components": {
        "securitySchemes": {
            "ApiKeyAuth": {
                "type": "apiKey",
                "in": "header",
                "name": "X-API-Key",
                "description": "API key for authentication"
            }
        }
    },
    "security": [
        {
            "ApiKeyAuth": []
        }
    ],
    "paths": {
        "/health": {
            "get": {
                "summary": "Health Check",
                "description": "Check API health status",
                "security": [],
                "responses": {
                    "200": {
                        "description": "API is healthy",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "status": {"type": "string"},
                                        "timestamp": {"type": "string"},
                                        "version": {"type": "string"},
                                        "service": {"type": "string"}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "/flags": {
            "get": {
                "summary": "Feature Flags Status",
                "description": "Get current feature flag states",
                "security": [],
                "responses": {
                    "200": {
                        "description": "Feature flags retrieved successfully",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "flags": {"type": "object"},
                                        "timestamp": {"type": "string"}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "/config": {
            "get": {
                "summary": "Configuration Debug",
                "description": "Get current configuration (debug mode only)",
                "security": [],
                "responses": {
                    "200": {
                        "description": "Configuration retrieved successfully"
                    },
                    "403": {
                        "description": "Config debug not enabled"
                    }
                }
            }
        },
        "/recs": {
            "get": {
                "summary": "Get Recommendations",
                "description": "Generate buy/hold/reduce/exit recommendations",
                "parameters": [
                    {
                        "name": "scope",
                        "in": "query",
                        "required": True,
                        "schema": {
                            "type": "string",
                            "enum": ["watchlist", "portfolio"]
                        },
                        "description": "Scope of recommendations"
                    },
                    {
                        "name": "include_details",
                        "in": "query",
                        "schema": {
                            "type": "boolean",
                            "default": False
                        },
                        "description": "Include detailed analysis"
                    },
                    {
                        "name": "max_age_hours",
                        "in": "query",
                        "schema": {
                            "type": "integer",
                            "default": 24
                        },
                        "description": "Maximum age of data in hours"
                    }
                ],
                "responses": {
                    "200": {
                        "description": "Recommendations generated successfully"
                    },
                    "400": {
                        "description": "Invalid parameters"
                    },
                    "401": {
                        "description": "API key required"
                    },
                    "403": {
                        "description": "Invalid API key"
                    },
                    "503": {
                        "description": "Feature disabled"
                    }
                }
            }
        },
        "/symbols/intake": {
            "post": {
                "summary": "Submit Symbol for Analysis",
                "description": "Add a new symbol for analysis and tracking",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "required": ["ticker", "company_name"],
                                "properties": {
                                    "ticker": {
                                        "type": "string",
                                        "maxLength": 10,
                                        "description": "Stock ticker symbol"
                                    },
                                    "company_name": {
                                        "type": "string",
                                        "maxLength": 255,
                                        "description": "Company name"
                                    },
                                    "priority": {
                                        "type": "string",
                                        "enum": ["low", "normal", "high"],
                                        "default": "normal",
                                        "description": "Processing priority"
                                    },
                                    "force_refresh": {
                                        "type": "boolean",
                                        "default": False,
                                        "description": "Force refresh if symbol exists"
                                    }
                                }
                            }
                        }
                    }
                },
                "responses": {
                    "202": {
                        "description": "Symbol queued for analysis"
                    },
                    "400": {
                        "description": "Invalid request"
                    },
                    "401": {
                        "description": "API key required"
                    },
                    "403": {
                        "description": "Invalid API key"
                    },
                    "429": {
                        "description": "Queue capacity exceeded"
                    },
                    "503": {
                        "description": "Feature disabled"
                    }
                }
            }
        },
        "/symbols/jobs/{job_id}": {
            "get": {
                "summary": "Get Job Status",
                "description": "Check status of a symbol intake job",
                "parameters": [
                    {
                        "name": "job_id",
                        "in": "path",
                        "required": True,
                        "schema": {
                            "type": "string"
                        },
                        "description": "Job ID from intake request"
                    }
                ],
                "responses": {
                    "200": {
                        "description": "Job status retrieved"
                    },
                    "401": {
                        "description": "API key required"
                    },
                    "403": {
                        "description": "Invalid API key"
                    },
                    "404": {
                        "description": "Job not found"
                    }
                }
            }
        },
        "/symbols/{symbol}/intake_status": {
            "get": {
                "summary": "Get Symbol Intake Status",
                "description": "Get detailed intake progress for a symbol",
                "parameters": [
                    {
                        "name": "symbol",
                        "in": "path",
                        "required": True,
                        "schema": {
                            "type": "string"
                        },
                        "description": "Stock ticker symbol"
                    },
                    {
                        "name": "job_id",
                        "in": "query",
                        "schema": {
                            "type": "string"
                        },
                        "description": "Specific job ID to check"
                    }
                ],
                "responses": {
                    "200": {
                        "description": "Intake status retrieved"
                    },
                    "401": {
                        "description": "API key required"
                    },
                    "403": {
                        "description": "Invalid API key"
                    },
                    "404": {
                        "description": "Symbol or job not found"
                    }
                }
            }
        },
        "/symbols/{symbol}/snapshot": {
            "get": {
                "summary": "Get Symbol Snapshot",
                "description": "Get comprehensive overview of a symbol",
                "parameters": [
                    {
                        "name": "symbol",
                        "in": "path",
                        "required": True,
                        "schema": {
                            "type": "string"
                        },
                        "description": "Stock ticker symbol"
                    }
                ],
                "responses": {
                    "200": {
                        "description": "Symbol snapshot retrieved"
                    },
                    "401": {
                        "description": "API key required"
                    },
                    "403": {
                        "description": "Invalid API key"
                    },
                    "404": {
                        "description": "Symbol not found"
                    }
                }
            }
        },
        "/earnings/upcoming": {
            "get": {
                "summary": "Get Upcoming Earnings",
                "description": "Retrieve upcoming earnings within specified timeframe",
                "parameters": [
                    {
                        "name": "days",
                        "in": "query",
                        "schema": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 90,
                            "default": 21
                        },
                        "description": "Number of days to look ahead"
                    }
                ],
                "responses": {
                    "200": {
                        "description": "Upcoming earnings retrieved"
                    },
                    "400": {
                        "description": "Invalid parameters"
                    },
                    "401": {
                        "description": "API key required"
                    },
                    "403": {
                        "description": "Invalid API key"
                    },
                    "503": {
                        "description": "Feature disabled"
                    }
                }
            }
        },
        "/earnings/{symbol}/explain": {
            "get": {
                "summary": "Explain Earnings Analysis",
                "description": "Get detailed explanation of earnings analysis for a symbol",
                "parameters": [
                    {
                        "name": "symbol",
                        "in": "path",
                        "required": True,
                        "schema": {
                            "type": "string",
                            "maxLength": 10
                        },
                        "description": "Stock ticker symbol"
                    }
                ],
                "responses": {
                    "200": {
                        "description": "Earnings analysis explanation retrieved"
                    },
                    "400": {
                        "description": "Invalid symbol"
                    },
                    "401": {
                        "description": "API key required"
                    },
                    "403": {
                        "description": "Invalid API key"
                    },
                    "404": {
                        "description": "Analysis not available"
                    },
                    "503": {
                        "description": "Feature disabled"
                    }
                }
            }
        },
        "/admin/logs": {
            "get": {
                "summary": "Get Recent Logs",
                "description": "Retrieve recent audit logs with filtering options",
                "parameters": [
                    {
                        "name": "hours",
                        "in": "query",
                        "schema": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 168,
                            "default": 24
                        },
                        "description": "Hours of logs to retrieve"
                    },
                    {
                        "name": "feature",
                        "in": "query",
                        "schema": {
                            "type": "string",
                            "default": "all"
                        },
                        "description": "Filter by feature name"
                    },
                    {
                        "name": "operation",
                        "in": "query",
                        "schema": {
                            "type": "string",
                            "default": "all"
                        },
                        "description": "Filter by operation name"
                    },
                    {
                        "name": "status",
                        "in": "query",
                        "schema": {
                            "type": "string",
                            "default": "all"
                        },
                        "description": "Filter by status"
                    }
                ],
                "responses": {
                    "200": {
                        "description": "Logs retrieved successfully"
                    },
                    "400": {
                        "description": "Invalid parameters"
                    },
                    "401": {
                        "description": "API key required"
                    },
                    "403": {
                        "description": "Invalid API key"
                    },
                    "503": {
                        "description": "API endpoints disabled"
                    }
                }
            }
        }
    }
}

@openapi_blueprint.route('/openapi.json', methods=['GET'])
def get_openapi_spec():
    """Return OpenAPI specification in JSON format"""
    if not is_api_endpoints_enabled():
        return jsonify({'error': 'API endpoints are disabled'}), 503
    
    return jsonify(OPENAPI_SPEC), 200

@openapi_blueprint.route('/docs', methods=['GET'])
def swagger_ui():
    """Serve Swagger UI for API documentation"""
    if not is_api_endpoints_enabled():
        return jsonify({'error': 'API endpoints are disabled'}), 503
    
    # Simple Swagger UI HTML template
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Stonk News API Documentation</title>
        <link rel="stylesheet" type="text/css" href="https://unpkg.com/swagger-ui-dist@4.15.5/swagger-ui.css" />
        <style>
            html { box-sizing: border-box; overflow: -moz-scrollbars-vertical; overflow-y: scroll; }
            *, *:before, *:after { box-sizing: inherit; }
            body { margin: 0; background: #fafafa; }
        </style>
    </head>
    <body>
        <div id="swagger-ui"></div>
        <script src="https://unpkg.com/swagger-ui-dist@4.15.5/swagger-ui-bundle.js"></script>
        <script src="https://unpkg.com/swagger-ui-dist@4.15.5/swagger-ui-standalone-preset.js"></script>
        <script>
            window.onload = function() {
                const ui = SwaggerUIBundle({
                    url: '/api/v1/openapi.json',
                    dom_id: '#swagger-ui',
                    deepLinking: true,
                    presets: [
                        SwaggerUIBundle.presets.apis,
                        SwaggerUIStandalonePreset
                    ],
                    plugins: [
                        SwaggerUIBundle.plugins.DownloadUrl
                    ],
                    layout: "StandaloneLayout"
                });
            };
        </script>
    </body>
    </html>
    """
    return render_template_string(html_template)