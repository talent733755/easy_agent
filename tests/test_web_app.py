"""Tests for FastAPI Web Application."""

import pytest
import json
from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient

from web.app import create_app


@pytest.fixture
def client():
    """Create test client."""
    app = create_app()
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_check_returns_200(self, client):
        """Health check should return 200 status."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_check_returns_json(self, client):
        """Health check should return JSON response."""
        response = client.get("/health")
        data = response.json()

        assert data["status"] == "healthy"
        assert data["version"] == "0.1.0"
        assert "provider" in data


class TestIndexPage:
    """Tests for index page."""

    def test_index_returns_html(self, client):
        """Index should return HTML page."""
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_index_contains_title(self, client):
        """Index page should contain title."""
        response = client.get("/")
        assert "Easy Agent" in response.text

    def test_index_links_static_files(self, client):
        """Index page should link to static files."""
        response = client.get("/")
        assert "/static/style.css" in response.text
        assert "/static/chat.js" in response.text


class TestStaticFiles:
    """Tests for static file serving."""

    def test_css_file_served(self, client):
        """CSS file should be served correctly."""
        response = client.get("/static/style.css")
        assert response.status_code == 200
        assert "text/css" in response.headers["content-type"]

    def test_js_file_served(self, client):
        """JavaScript file should be served correctly."""
        response = client.get("/static/chat.js")
        assert response.status_code == 200
        assert "javascript" in response.headers["content-type"]


class TestWebSocket:
    """Tests for WebSocket endpoint."""

    def test_websocket_connect(self, client):
        """WebSocket should accept connection."""
        with client.websocket_connect("/ws") as websocket:
            # Should receive session message
            data = websocket.receive_json()
            assert data["type"] == "session"
            assert "session_id" in data
            assert "provider" in data

    def test_websocket_echo_message(self, client):
        """WebSocket should handle message and respond."""
        with client.websocket_connect("/ws") as websocket:
            # Receive session info
            session_data = websocket.receive_json()
            assert session_data["type"] == "session"

            # Send a simple message
            # Note: This will invoke the actual graph which may fail without proper setup
            # In a real test, we would mock the graph
            websocket.send_json({
                "type": "message",
                "content": "/help"
            })

            # Should receive response
            response = websocket.receive_json()
            assert response["type"] == "response"
            assert "Commands:" in response["content"]


class TestSlashCommands:
    """Tests for slash command handling."""

    def test_help_command(self, client):
        """Help command should return help text."""
        with client.websocket_connect("/ws") as websocket:
            websocket.receive_json()  # Skip session message

            websocket.send_json({
                "type": "message",
                "content": "/help"
            })

            response = websocket.receive_json()
            assert response["type"] == "response"
            assert "/help" in response["content"]
            assert "/memory" in response["content"]

    def test_unknown_command(self, client):
        """Unknown command should return error message."""
        with client.websocket_connect("/ws") as websocket:
            websocket.receive_json()  # Skip session message

            websocket.send_json({
                "type": "message",
                "content": "/unknown"
            })

            response = websocket.receive_json()
            assert response["type"] == "response"
            assert "Unknown command" in response["content"]


class TestAppCreation:
    """Tests for app creation."""

    def test_create_app_returns_fastapi(self):
        """create_app should return FastAPI instance."""
        from fastapi import FastAPI
        app = create_app()
        assert isinstance(app, FastAPI)

    def test_create_app_with_config_path(self):
        """create_app should accept config path."""
        app = create_app(config_path="config.yaml")
        assert app is not None

    def test_app_has_routes(self):
        """App should have expected routes."""
        app = create_app()
        routes = [route.path for route in app.routes]

        assert "/" in routes
        assert "/health" in routes
        assert "/ws" in routes
