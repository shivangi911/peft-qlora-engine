"""Server tests. The model call is mocked, so these run anywhere (including CI)."""

from fastapi.testclient import TestClient

from peft_engine import server

client = TestClient(server.app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_stream_sse(monkeypatch):
    monkeypatch.setattr(server, "generate_tokens", lambda req: iter(["Hel", "lo"]))
    r = client.post("/v1/chat/completions/stream", json={"prompt": "hi", "max_tokens": 4})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    assert 'data: {"token": "Hel"}' in r.text
    assert r.text.rstrip().endswith("data: [DONE]")


def test_stream_reports_errors(monkeypatch):
    def boom(req):
        raise RuntimeError("no model")
        yield  # pragma: no cover

    monkeypatch.setattr(server, "generate_tokens", boom)
    r = client.post("/v1/chat/completions/stream", json={"prompt": "hi"})
    assert "no model" in r.text
    assert "[DONE]" in r.text


def test_validation():
    assert (
        client.post(
            "/v1/chat/completions/stream", json={"prompt": "x", "max_tokens": 0}
        ).status_code
        == 422
    )
