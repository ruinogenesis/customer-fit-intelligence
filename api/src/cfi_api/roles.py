"""Staff accounts and their roles: python -m cfi_api.roles (T303).

    python -m cfi_api.roles --project cfi-dev-th26 grant EMAIL ROLE [--create]
    python -m cfi_api.roles --project cfi-dev-th26 revoke EMAIL
    python -m cfi_api.roles --project cfi-dev-th26 list

A role is kept in two places, in step: the custom claim role on the
Firebase Auth account, which the API (cfi_api.access) and the Firestore
rules check, and users/{uid} in Firestore, for the dashboard to show
(fitcore.documents). grant --create makes the account first if there is
none, with a password typed at the prompt. A new role reaches a person's
token at their next sign-in, or within an hour, when it refreshes. revoke
signs them out everywhere, but a token already issued works until it
expires, within an hour.

It runs as you, with Application Default Credentials (gcloud auth
application-default login, and set-quota-project with the project). With
FIREBASE_AUTH_EMULATOR_HOST and FIRESTORE_EMULATOR_HOST set, as
firebase emulators:exec sets them, it works on the emulators instead.
"""

import argparse
import getpass
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import uuid4

import firebase_admin
from firebase_admin import auth
from fitcore.documents import ROLES, check
from google.cloud import firestore


class RolesError(Exception):
    """A change the command refuses, such as one for an unknown email."""


@dataclass(frozen=True)
class Project:
    """Firebase Auth and Firestore of one project."""

    firebase: firebase_admin.App
    db: firestore.Client


def connect(project: str) -> Project:
    """Firebase Auth and Firestore of project, as you (or the emulators)."""
    firebase = firebase_admin.initialize_app(
        options={"projectId": project}, name=f"cfi-roles-{uuid4().hex}"
    )
    return Project(firebase, firestore.Client(project=project))


def _find(project: Project, email: str) -> auth.UserRecord | None:
    try:
        return auth.get_user_by_email(email, app=project.firebase)
    except auth.UserNotFoundError:
        return None


def grant(
    project: Project,
    email: str,
    role: str,
    new_password: Callable[[], str] | None = None,
) -> str:
    """Give the account for email role, and answer its uid.

    Other custom claims stay as they are. If there is no account,
    new_password makes one with the password it answers; without it, that
    raises RolesError, as does an unknown role.
    """
    if role not in ROLES:
        raise RolesError(f"The role must be one of {', '.join(ROLES)}, not {role!r}")
    user = _find(project, email)
    if user is None:
        if new_password is None:
            raise RolesError(f"There is no account for {email}: add --create")
        user = auth.create_user(
            email=email, password=new_password(), app=project.firebase
        )
    document = {
        "uid": user.uid,
        "email": user.email,
        "role": role,
        "updated_at": datetime.now(UTC),
    }
    check("users", document)
    claims = {**(user.custom_claims or {}), "role": role}
    auth.set_custom_user_claims(user.uid, claims, app=project.firebase)
    project.db.document(f"users/{user.uid}").set(document)
    uid: str = user.uid
    return uid


def revoke(project: Project, email: str) -> str:
    """Take the role from the account for email and sign it out everywhere.

    Answers its uid. Other custom claims stay as they are. Raises
    RolesError if there is no account for email.
    """
    user = _find(project, email)
    if user is None:
        raise RolesError(f"There is no account for {email}")
    claims = {k: v for k, v in (user.custom_claims or {}).items() if k != "role"}
    auth.set_custom_user_claims(user.uid, claims or None, app=project.firebase)
    auth.revoke_refresh_tokens(user.uid, app=project.firebase)
    project.db.document(f"users/{user.uid}").delete()
    uid: str = user.uid
    return uid


def staff(project: Project) -> list[tuple[str, str, str]]:
    """(role, email, uid) for every account with a role, by role then email."""
    found = []
    for user in auth.list_users(app=project.firebase).iterate_all():
        role = (user.custom_claims or {}).get("role")
        if role is not None:
            found.append((str(role), user.email or "", user.uid))
    return sorted(found)


def ask_password(email: str) -> str:
    """A password for email's new account, typed twice; it is never shown."""
    password = getpass.getpass(f"Password for {email}: ")
    if getpass.getpass("The same password again: ") != password:
        raise RolesError("The two passwords differ")
    return password


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m cfi_api.roles", description="Staff accounts and their roles"
    )
    parser.add_argument(
        "--project", required=True, help="the project, such as cfi-dev-th26"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    grant_command = commands.add_parser("grant", help="give an account a role")
    grant_command.add_argument("email")
    grant_command.add_argument("role", choices=ROLES)
    grant_command.add_argument(
        "--create",
        action="store_true",
        help="make the account if there is none, with a password you type",
    )
    revoke_command = commands.add_parser(
        "revoke", help="take an account's role and sign it out"
    )
    revoke_command.add_argument("email")
    commands.add_parser("list", help="every account with a role")
    args = parser.parse_args(argv)

    project = connect(args.project)
    try:
        if args.command == "grant":
            asker = (lambda: ask_password(args.email)) if args.create else None
            uid = grant(project, args.email, args.role, asker)
            print(f"{args.email} ({uid}) is now {args.role}.")
            print("It reaches their token at their next sign-in, or within an hour.")
        elif args.command == "revoke":
            uid = revoke(project, args.email)
            print(f"{args.email} ({uid}) has no role now, and is signed out.")
            print("A token already issued works until it expires, within an hour.")
        else:
            rows = staff(project)
            for role, email, uid in rows:
                print(f"{role:<8} {email}  {uid}")
            if not rows:
                print("No account has a role.")
    except (RolesError, ValueError) as err:
        # ValueError: what firebase-admin raises for a password that is too
        # short or an email it cannot take.
        print(f"error: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
