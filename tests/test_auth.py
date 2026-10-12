from app import create_app


def test_pages_redirect_to_login_when_not_authenticated():
    client = create_app().test_client()
    response = client.get("/")
    assert response.status_code == 302
    assert "/login" in response.headers["Location"]


def test_api_requires_authentication():
    client = create_app().test_client()
    response = client.get("/api/config")
    assert response.status_code == 401
    assert response.get_json()["error"] == "Autenticação necessária."


def test_demo_accounts_can_login_and_logout():
    client = create_app().test_client()
    for username in ("ubi", "inesctec"):
        response = client.post("/login", data={"username": username, "password": username, "next": "/"})
        assert response.status_code == 302
        assert response.headers["Location"].endswith("/")
        assert client.get("/").status_code == 200
        client.get("/logout")
