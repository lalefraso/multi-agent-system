from __future__ import annotations

import argparse
import sys
from urllib.parse import urlsplit

from app.authorization import (
    CustomerAlreadyAssignedError,
    CustomerAuthorizationError,
    CustomerMembershipNotFoundError,
    grant_customer_membership,
    revoke_customer_membership,
)


def _principal_id(issuer: str, subject: str) -> str:
    parsed = urlsplit(issuer)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.fragment
    ):
        raise ValueError("issuer must be an absolute HTTPS URL")
    if not subject or not subject.strip() or "\0" in subject:
        raise ValueError("subject must be a non-empty OIDC subject")
    return f"{issuer}\0{subject}"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manage unique principal-to-customer authorization."
    )
    commands = parser.add_subparsers(dest="action", required=True)
    for action in ("grant", "revoke"):
        command = commands.add_parser(action)
        command.add_argument("--issuer", required=True)
        command.add_argument("--subject", required=True)
        command.add_argument("--customer-id", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        principal_id = _principal_id(args.issuer, args.subject)
        if args.action == "grant":
            grant_customer_membership(principal_id, args.customer_id)
        else:
            revoke_customer_membership(principal_id, args.customer_id)
    except (
        CustomerAlreadyAssignedError,
        CustomerAuthorizationError,
        CustomerMembershipNotFoundError,
        ValueError,
    ) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"Customer membership {args.action} completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
