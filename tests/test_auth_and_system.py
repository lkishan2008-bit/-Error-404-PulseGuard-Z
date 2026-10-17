"""
==============================================================================
PulseGuard-Z  -  Test Suite: User Registration, Non-Admin Security & System Info
==============================================================================
Project   : PulseGuard-Z (IBM Z Hackathon, event code SSF2026)
Team      : Error-404
File      : tests/test_auth_and_system.py
"""

import platform
import sys
import unittest
from pathlib import Path

# Always ensure repo root is in python path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import backend first to install Python 3.14 typing shim before FastAPI testclient
import backend
from fastapi.testclient import TestClient

from backend.auth import (
    hash_password,
    login_rate_limiter,
    session_store,
    user_store,
)
from backend.main import app, repo


class TestAuthAndSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        repo.load_data()

    def setUp(self):
        # Reset in-memory users and rate limiter between tests
        user_store._users.clear()
        session_store._sessions.clear()
        login_rate_limiter._attempts.clear()

    def test_system_endpoint_returns_machine(self):
        """Verify GET /api/system returns actual platform.machine()."""
        res = self.client.get("/api/system")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        expected_machine = platform.machine() or "unknown"
        self.assertEqual(data["machine"], expected_machine)
        self.assertIn("system", data)
        self.assertIn("processor", data)
        self.assertEqual(data["status"], "online")

    def test_registration_success_with_operator_role(self):
        """Verify registration creates an operator account with PBKDF2 hashing."""
        payload = {"username": "analyst_sarah", "password": "SecurePassword123!"}
        res = self.client.post("/api/auth/register", json=payload)
        self.assertEqual(res.status_code, 201)
        data = res.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["username"], "analyst_sarah")
        self.assertEqual(data["role"], "operator")  # Never admin

        # Verify password is not in plaintext in user store
        stored = user_store.get_user("analyst_sarah")
        self.assertIsNotNone(stored)
        self.assertNotIn("SecurePassword123!", stored["password_hash"])
        self.assertIn(":", stored["password_hash"])  # salt:hash format
        self.assertEqual(stored["role"], "operator")

    def test_registration_never_grants_admin_privileges(self):
        """Verify registration cannot assign administrator privileges."""
        payload = {"username": "new_user_1", "password": "ValidPassword99"}
        res = self.client.post("/api/auth/register", json=payload)
        self.assertEqual(res.status_code, 201)
        user = user_store.get_user("new_user_1")
        self.assertNotEqual(user["role"], "admin")
        self.assertEqual(user["role"], "operator")

    def test_registration_rejects_admin_reserved_usernames(self):
        """Verify attempts to register admin or reserved names are blocked."""
        for reserved in ["admin", "Administrator", "root", "system"]:
            res = self.client.post(
                "/api/auth/register",
                json={"username": reserved, "password": "ValidPassword123!"},
            )
            self.assertEqual(res.status_code, 409)
            self.assertIn("reserved", res.json()["detail"].lower())

    def test_registration_validation_rules(self):
        """Verify input validation: short usernames, bad characters, weak passwords."""
        # Username too short (<3)
        res1 = self.client.post(
            "/api/auth/register", json={"username": "ab", "password": "ValidPassword123"}
        )
        self.assertEqual(res1.status_code, 422)

        # Username with invalid characters (spaces)
        res2 = self.client.post(
            "/api/auth/register",
            json={"username": "bad user", "password": "ValidPassword123"},
        )
        self.assertEqual(res2.status_code, 422)

        # Password too short (<8)
        res3 = self.client.post(
            "/api/auth/register", json={"username": "valid_user", "password": "short"}
        )
        self.assertEqual(res3.status_code, 422)

        # Password without numbers or letters
        res4 = self.client.post(
            "/api/auth/register", json={"username": "valid_user", "password": "allcharacters"}
        )
        self.assertEqual(res4.status_code, 422)

    def test_registration_duplicate_username_rejected(self):
        """Verify duplicate registrations of the same username are rejected."""
        payload = {"username": "duplicate_check", "password": "Password1234"}
        res1 = self.client.post("/api/auth/register", json=payload)
        self.assertEqual(res1.status_code, 201)

        res2 = self.client.post("/api/auth/register", json=payload)
        self.assertEqual(res2.status_code, 409)
        self.assertIn("already registered", res2.json()["detail"])

    def test_login_with_registered_operator(self):
        """Verify newly registered operator can log in and receives operator role token."""
        # Register user
        reg_payload = {"username": "op_test_user", "password": "TestPassword2026!"}
        reg_res = self.client.post("/api/auth/register", json=reg_payload)
        self.assertEqual(reg_res.status_code, 201)

        # Login
        login_res = self.client.post(
            "/api/auth/login",
            json={"username": "op_test_user", "password": "TestPassword2026!"},
        )
        self.assertEqual(login_res.status_code, 200)
        login_data = login_res.json()
        self.assertIn("token", login_data)
        self.assertEqual(login_data["username"], "op_test_user")
        self.assertEqual(login_data["role"], "operator")

        # Session validation
        token = login_data["token"]
        session_res = self.client.get(
            "/api/auth/session", headers={"Authorization": f"Bearer {token}"}
        )
        self.assertEqual(session_res.status_code, 200)
        self.assertTrue(session_res.json()["valid"])
        self.assertEqual(session_res.json()["role"], "operator")

    def test_login_invalid_password_rejected(self):
        """Verify wrong password fails with 401."""
        reg_payload = {"username": "op_wrong_pass", "password": "CorrectPassword123"}
        self.client.post("/api/auth/register", json=reg_payload)

        login_res = self.client.post(
            "/api/auth/login",
            json={"username": "op_wrong_pass", "password": "IncorrectPassword"},
        )
        self.assertEqual(login_res.status_code, 401)


if __name__ == "__main__":
    unittest.main()
