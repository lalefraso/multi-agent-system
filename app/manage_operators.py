from __future__ import annotations

import argparse
import sys

from app.manage_authorization import _principal_id
from app.operator_authorization import OperatorAuthorizationError, set_operator_access


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Manage authorized operators.")
    parser.add_argument("action", choices=("grant", "revoke"))
    parser.add_argument("--issuer", required=True)
    parser.add_argument("--subject", required=True)
    args = parser.parse_args(argv)
    try:
        principal_id = _principal_id(args.issuer, args.subject)
        set_operator_access(principal_id, grant=args.action == "grant")
    except (OperatorAuthorizationError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"Operator access {args.action} completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
