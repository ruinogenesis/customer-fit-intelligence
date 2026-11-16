"""T303: python -m cfi_api.roles, on the Auth and Firestore emulators.

These run in CI's emulator job (or locally with firebase emulators:exec
--only firestore,auth); elsewhere they are skipped. The last test signs in
as the emulator's users would and calls the API with their real tokens.
"""

import json
import os
import urllib.request
import uuid

import pytest
from firebase_admin import auth
from fitcore.sim_events import simulate_events

from cfi_api import create_app, roles
from cfi_api.roles import Project, RolesError, grant, revoke, staff

pytestmark = pytest.mark.skipif(
    not (
        os.environ.get("FIREBASE_AUTH_EMULATOR_HOST")
        and os.environ.get("FIRESTORE_EMULATOR_HOST")
    ),
    reason="needs the Auth and Firestore emulators: run via firebase emulators:exec",
)

PROJECT = "demo-cfi"
PASSWORD = "a-test-password"


@pytest.fixture(scope="module")
def project() -> Project:
    return roles.connect(PROJECT)


@pytest.fixture
def email() -> str:
    """A new email for each test: the emulator keeps accounts between tests."""
    return f"staff-{uuid.uuid4().hex[:12]}@example.com"


def new_password() -> str:
    return PASSWORD


def user_document(project: Project, uid: str) -> dict[str, object] | None:
    return project.db.document(f"users/{uid}").get().to_dict()


def test_grant_create_makes_the_account_with_the_claim_and_the_document(
    project: Project, email: str
) -> None:
    uid = grant(project, email, "analyst", new_password)
    assert auth.get_user(uid, app=project.firebase).custom_claims == {"role": "analyst"}
    document = user_document(project, uid)
    assert document is not None
    assert (document["uid"], document["email"], document["role"]) == (
        uid,
        email,
        "analyst",
    )


def test_grant_needs_create_for_an_email_without_an_account(
    project: Project, email: str
) -> None:
    with pytest.raises(RolesError, match="add --create"):
        grant(project, email, "viewer")


def test_grant_changes_the_role_and_keeps_other_claims(
    project: Project, email: str
) -> None:
    uid = grant(project, email, "viewer", new_password)
    auth.set_custom_user_claims(
        uid, {"role": "viewer", "team": "ops"}, app=project.firebase
    )
    assert grant(project, email, "admin") == uid
    claims = auth.get_user(uid, app=project.firebase).custom_claims
    assert claims == {"role": "admin", "team": "ops"}
    document = user_document(project, uid)
    assert document is not None and document["role"] == "admin"


def test_grant_refuses_an_unknown_role(project: Project, email: str) -> None:
    with pytest.raises(RolesError, match="must be one of"):
        grant(project, email, "owner", new_password)


def test_revoke_takes_the_role_and_the_document_away(
    project: Project, email: str
) -> None:
    uid = grant(project, email, "analyst", new_password)
    assert revoke(project, email) == uid
    assert not auth.get_user(uid, app=project.firebase).custom_claims
    assert user_document(project, uid) is None
    assert uid not in {found_uid for _, _, found_uid in staff(project)}


def test_revoke_refuses_an_email_without_an_account(
    project: Project, email: str
) -> None:
    with pytest.raises(RolesError, match="no account"):
        revoke(project, email)


def test_staff_lists_every_account_with_a_role(project: Project, email: str) -> None:
    uid = grant(project, email, "viewer", new_password)
    assert ("viewer", email, uid) in staff(project)


def test_main_grants_lists_and_revokes(
    email: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(roles.getpass, "getpass", lambda prompt: PASSWORD)
    base = ["--project", PROJECT]
    assert roles.main([*base, "grant", email, "viewer", "--create"]) == 0
    assert roles.main([*base, "list"]) == 0
    assert roles.main([*base, "revoke", email]) == 0
    out = capsys.readouterr().out
    assert f"{email} (" in out and "is now viewer" in out
    assert f"viewer   {email}" in out
    assert "has no role now" in out


def test_main_reports_a_refused_change_without_a_traceback(
    email: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert roles.main(["--project", PROJECT, "grant", email, "viewer"]) == 1
    assert "add --create" in capsys.readouterr().err


def sign_in(email: str) -> str:
    """An ID token for email, from the Auth emulator's sign-in endpoint."""
    host = os.environ["FIREBASE_AUTH_EMULATOR_HOST"]
    request = urllib.request.Request(
        f"http://{host}/identitytoolkit.googleapis.com/v1/"
        "accounts:signInWithPassword?key=any",
        data=json.dumps(
            {"email": email, "password": PASSWORD, "returnSecureToken": True}
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request) as response:
        token: str = json.load(response)["idToken"]
    return token


def test_a_granted_role_opens_the_api_and_revoke_closes_it(
    project: Project, email: str, reference_model: str
) -> None:
    app = create_app(
        {
            "MODEL_PATH": reference_model,
            "MODEL_VERSION": "t112-sim",
            "AUTH": "firebase",
            "PROJECT": PROJECT,
        }
    )
    client = app.test_client()
    event = simulate_events(1, seed=1, days=7.0)[0]

    def post(path: str) -> int:
        headers = {"Authorization": f"Bearer {sign_in(email)}"}
        return client.post(f"/api/v1/{path}", json=event, headers=headers).status_code

    grant(project, email, "analyst", new_password)
    assert (post("score"), post("events")) == (200, 403)
    revoke(project, email)
    assert post("score") == 403
