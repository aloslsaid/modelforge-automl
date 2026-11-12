from app.main import app


def test_documented_asgi_entrypoint_exposes_health_and_api_docs():
    paths = {route.path for route in app.routes}
    assert "/" in paths
    assert "/health" in paths
    assert "/prepare" in paths
    assert "/train" in paths
    assert any(path.startswith("/download/") for path in paths)
