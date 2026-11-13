from __future__ import annotations

import argparse
import sys
from typing import Literal

from app.manage_authorization import _principal_id
from app.operator_authorization import OperatorAuthorizationError, set_staff_access


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Manage authorized support staff.")
    parser.add_argument("action", choices=("grant", "revoke"))
    parser.add_argument(
        "--role", choices=("operator", "manager"), default="operator"
    )
    parser.add_argument("--issuer", required=True)
    parser.add_argument("--subject", required=True)
    args = parser.parse_args(argv)
    try:
        principal_id = _principal_id(args.issuer, args.subject)
        role: Literal["operator", "manager"] = args.role
        set_staff_access(principal_id, role=role, grant=args.action == "grant")
    except (OperatorAuthorizationError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"{args.role.title()} access {args.action} completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
