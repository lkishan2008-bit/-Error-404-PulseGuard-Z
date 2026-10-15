#!/usr/bin/env python3
"""
==============================================================================
PulseGuard-Z  -  Password Hash Helper Script
==============================================================================
Generates a cryptographically secure PBKDF2-HMAC-SHA256 salt and hash
for use with DASHBOARD_PASSWORD_HASH in the PulseGuard-Z .env file.

Usage:
    python scripts/hash_password.py [optional_password]

If no password argument is supplied, you will be prompted securely without echo.
==============================================================================
"""

import getpass
import hashlib
import secrets
import sys

ITERATIONS = 260_000


def generate_hash(password: str) -> str:
    salt_bytes = secrets.token_bytes(16)
    salt_hex = salt_bytes.hex()
    hash_bytes = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt_bytes,
        ITERATIONS,
    )
    return f"{salt_hex}:{hash_bytes.hex()}"


def main():
    if len(sys.argv) > 1:
        pwd = sys.argv[1]
    else:
        pwd = getpass.getpass("Enter dashboard password to hash: ")
        if not pwd:
            print("Error: Password cannot be empty.", file=sys.stderr)
            sys.exit(1)

    hash_val = generate_hash(pwd)
    print("\nGenerated PBKDF2-HMAC-SHA256 Hash:")
    print("-" * 50)
    print(hash_val)
    print("-" * 50)
    print("\nTo configure authentication, add the following to your .env file:")
    print("DASHBOARD_USER=admin")
    print(f"DASHBOARD_PASSWORD_HASH={hash_val}")
    print("SESSION_TTL_MINUTES=30")


if __name__ == "__main__":
    main()
