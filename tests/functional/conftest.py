import pytest
from fastapi.testclient import TestClient

import api


@pytest.fixture(autouse=True)
def reset_rebuild_state():
    def _reset():
        api.rebuild_state["status"] = "ready"
        api.rebuild_state["events_indexed"] = None
        api.rebuild_state["detail"] = None

    _reset()
    yield
    _reset()


@pytest.fixture
def client():
    return TestClient(api.app)
