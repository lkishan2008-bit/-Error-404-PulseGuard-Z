"""
==============================================================================
PulseGuard-Z  -  Test Suite: CSV Upload, Validation & Per-Session Isolation
==============================================================================
Project   : PulseGuard-Z (IBM Z Hackathon, event code SSF2026)
Team      : Error-404
File      : tests/test_upload.py
"""

import sys
import unittest
from pathlib import Path

# Always ensure repo root is in python path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import backend to install Python 3.14 typing shim before FastAPI testclient
import backend
from fastapi.testclient import TestClient

from backend.auth import verify_api_key
from backend.main import app, repo
from backend.upload import (
    MAX_UPLOAD_ROWS,
    UploadValidationError,
    generate_sample_csv,
    parse_and_validate_csv,
    upload_manager,
)


class TestUploadFeature(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.dependency_overrides[verify_api_key] = lambda: "test-upload-client"
        cls.client = TestClient(app)
        repo.load_data()

    @classmethod
    def tearDownClass(cls):
        app.dependency_overrides.pop(verify_api_key, None)

    def setUp(self):
        # Clear upload manager between tests
        upload_manager._datasets.clear()

    def test_sample_csv_download(self):
        """Verify GET /api/upload/sample returns compliant downloadable CSV."""
        res = self.client.get("/api/upload/sample")
        self.assertEqual(res.status_code, 200)
        self.assertIn("text/csv", res.headers.get("content-type", ""))
        self.assertIn("attachment", res.headers.get("content-disposition", ""))
        content = res.text
        self.assertTrue(content.startswith("timestamp,sender_id,receiver_id,amount"))
        self.assertNotIn("is_fraud", content)

    def test_empty_file_rejected(self):
        """Verify empty CSV upload is rejected with clear validation error."""
        res = self.client.post(
            "/api/upload",
            content=b"",
            headers={"Content-Type": "text/csv", "X-Session-ID": "test_user_empty"},
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn("Empty upload body", res.json()["detail"])

    def test_prohibited_fraud_label_rejected(self):
        """Verify CSV containing ground-truth label columns is strictly rejected."""
        bad_csv = (
            "timestamp,sender_id,receiver_id,amount,is_fraud\n"
            "2026-10-16 10:00:00,ACC_01,ACC_02,500.0,1\n"
        )
        res = self.client.post(
            "/api/upload",
            content=bad_csv.encode("utf-8"),
            headers={"Content-Type": "text/csv", "X-Session-ID": "test_fraud_col"},
        )
        self.assertEqual(res.status_code, 422)
        detail = res.json()["detail"]
        self.assertIn("Prohibited label column", detail)
        self.assertIn("is_fraud", detail)

    def test_missing_required_column_rejected(self):
        """Verify CSV missing required amount column is rejected."""
        bad_csv = (
            "timestamp,sender_id,receiver_id\n"
            "2026-10-16 10:00:00,ACC_01,ACC_02\n"
        )
        res = self.client.post(
            "/api/upload",
            content=bad_csv.encode("utf-8"),
            headers={"Content-Type": "text/csv", "X-Session-ID": "test_missing_col"},
        )
        self.assertEqual(res.status_code, 422)
        self.assertIn("Missing required column", res.json()["detail"])

    def test_self_transfer_rejected(self):
        """Verify transactions where sender == receiver are rejected."""
        bad_csv = (
            "timestamp,sender_id,receiver_id,amount\n"
            "2026-10-16 10:00:00,ACC_SAME,ACC_SAME,100.0\n"
        )
        res = self.client.post(
            "/api/upload",
            content=bad_csv.encode("utf-8"),
            headers={"Content-Type": "text/csv", "X-Session-ID": "test_self_tx"},
        )
        self.assertEqual(res.status_code, 422)
        self.assertIn("self-transfer", res.json()["detail"].lower())

    def test_non_positive_amount_rejected(self):
        """Verify zero or negative amounts are rejected."""
        bad_csv = (
            "timestamp,sender_id,receiver_id,amount\n"
            "2026-10-16 10:00:00,ACC_01,ACC_02,-50.0\n"
        )
        res = self.client.post(
            "/api/upload",
            content=bad_csv.encode("utf-8"),
            headers={"Content-Type": "text/csv", "X-Session-ID": "test_neg_amt"},
        )
        self.assertEqual(res.status_code, 422)
        self.assertIn("positive", res.json()["detail"].lower())

    def test_invalid_timestamp_rejected(self):
        """Verify unparseable timestamp is rejected."""
        bad_csv = (
            "timestamp,sender_id,receiver_id,amount\n"
            "not-a-valid-date,ACC_01,ACC_02,150.0\n"
        )
        res = self.client.post(
            "/api/upload",
            content=bad_csv.encode("utf-8"),
            headers={"Content-Type": "text/csv", "X-Session-ID": "test_bad_ts"},
        )
        self.assertEqual(res.status_code, 422)
        self.assertIn("timestamp", res.json()["detail"].lower())

    def test_valid_upload_and_session_isolation(self):
        """
        Critical Security Test:
        - User A uploads a custom CSV (20 transactions, 13 accounts).
        - User A queries dashboard and sees 20 transactions, 13 accounts.
        - User B (different session ID) queries dashboard and sees the default 50,000 transactions and 2,000 accounts.
        - Verifies User A's upload NEVER contaminates User B or the shared demo.
        """
        sample_csv = generate_sample_csv()

        session_a_headers = {
            "Content-Type": "text/csv",
            "X-Session-ID": "session_user_alice",
        }
        session_b_headers = {
            "X-Session-ID": "session_user_bob",
        }

        # User A uploads data
        upload_res = self.client.post(
            "/api/upload",
            content=sample_csv.encode("utf-8"),
            headers=session_a_headers,
        )
        self.assertEqual(upload_res.status_code, 200)
        up_data = upload_res.json()
        self.assertEqual(up_data["status"], "success")
        self.assertEqual(up_data["total_transactions"], 20)
        self.assertEqual(up_data["total_accounts"], 13)

        # 1. User A checks /dashboard/summary: sees 20 txns and 13 accounts
        summary_a = self.client.get("/api/dashboard/summary", headers={"X-Session-ID": "session_user_alice"}).json()
        self.assertEqual(summary_a["dataset_summary"]["total_transactions"], 20)
        self.assertEqual(summary_a["dataset_summary"]["total_accounts"], 13)
        self.assertIn("Custom uploaded dataset", summary_a["data_fidelity_notice"])

        # 2. User B checks /dashboard/summary: sees standard demo (50,000 txns, 2,000 accounts)
        summary_b = self.client.get("/api/dashboard/summary", headers=session_b_headers).json()
        self.assertEqual(summary_b["dataset_summary"]["total_transactions"], 50000)
        self.assertEqual(summary_b["dataset_summary"]["total_accounts"], 2000)
        self.assertIn("Metrics computed directly from local synthetic G2P", summary_b["data_fidelity_notice"])

        # 3. User A checks /transactions: gets 20 transactions
        tx_a = self.client.get("/api/transactions", headers={"X-Session-ID": "session_user_alice"}).json()
        self.assertEqual(tx_a["total_records"], 20)

        # 4. User B checks /transactions: gets 50,000 transactions
        tx_b = self.client.get("/api/transactions", headers=session_b_headers).json()
        self.assertEqual(tx_b["total_records"], 50000)

        # 5. Account detail isolation
        # Account 'ACC_MULE_901' only exists in Alice's upload
        res_acct_a = self.client.get("/api/accounts/ACC_MULE_901", headers={"X-Session-ID": "session_user_alice"})
        self.assertEqual(res_acct_a.status_code, 200)
        self.assertEqual(res_acct_a.json()["account_id"], "ACC_MULE_901")

        # Bob queries 'ACC_MULE_901' -> 404 Not Found in default demo
        res_acct_b = self.client.get("/api/accounts/ACC_MULE_901", headers=session_b_headers)
        self.assertEqual(res_acct_b.status_code, 404)

        # 6. Graph isolation
        graph_a = self.client.get("/api/graph", headers={"X-Session-ID": "session_user_alice"}).json()
        self.assertIn("custom uploaded dataset", graph_a["data_notice"].lower())

        graph_b = self.client.get("/api/graph", headers=session_b_headers).json()
        self.assertIn("bounded real transaction network", graph_b["data_notice"].lower())

        # 7. Check upload status endpoint
        status_a = self.client.get("/api/upload/status", headers={"X-Session-ID": "session_user_alice"}).json()
        self.assertTrue(status_a["is_uploaded"])
        self.assertEqual(status_a["total_transactions"], 20)

        status_b = self.client.get("/api/upload/status", headers=session_b_headers).json()
        self.assertFalse(status_b["is_uploaded"])

        # 8. User A resets upload to return to demo
        reset_res = self.client.post("/api/upload/reset", headers={"X-Session-ID": "session_user_alice"})
        self.assertEqual(reset_res.status_code, 200)
        self.assertTrue(reset_res.json()["was_active"])

        # Now User A sees default demo again
        summary_a_after = self.client.get("/api/dashboard/summary", headers={"X-Session-ID": "session_user_alice"}).json()
        self.assertEqual(summary_a_after["dataset_summary"]["total_transactions"], 50000)

    def test_explain_and_suspicious_accounts_on_upload(self):
        """Verify /api/accounts/suspicious and /api/explain/{account_id} function on uploaded data without labels."""
        sample_csv = generate_sample_csv()
        sess_hdr = {"X-Session-ID": "session_user_charlie"}
        upload_res = self.client.post(
            "/api/upload",
            content=sample_csv.encode("utf-8"),
            headers={"Content-Type": "text/csv", **sess_hdr},
        )
        self.assertEqual(upload_res.status_code, 200)

        # Query suspicious accounts
        susp_res = self.client.get("/api/accounts/suspicious?threshold=0.05", headers=sess_hdr)
        self.assertEqual(susp_res.status_code, 200)
        susp_data = susp_res.json()
        self.assertIn("accounts", susp_data)
        self.assertGreater(len(susp_data["accounts"]), 0)

        # Inspect first flagged account
        first_acct = susp_data["accounts"][0]["account_id"]
        exp_res = self.client.get(f"/api/explain/{first_acct}", headers=sess_hdr)
        self.assertEqual(exp_res.status_code, 200)
        exp_data = exp_res.json()
        self.assertEqual(exp_data["account_id"], first_acct)
        self.assertIn("top_signals", exp_data)
        self.assertIn("signals_detail", exp_data)
        self.assertIn("risk_level", exp_data)

    def test_max_rows_limit_enforced(self):
        """Verify parse_and_validate_csv rejects uploads exceeding MAX_UPLOAD_ROWS."""
        # Test directly against parser for performance without building 200k strings in memory
        with self.assertRaises(UploadValidationError) as ctx:
            from io import StringIO
            # Mock or check threshold
            header = "timestamp,sender_id,receiver_id,amount\n"
            # 5 rows is fine, but passing row_limit=3 should trigger
            content = header + "2026-10-16 10:00:00,A,B,10\n" * 5
            parse_and_validate_csv(content.encode("utf-8"), max_rows=3)
        self.assertIn("exceeds the maximum limit", str(ctx.exception))


    def test_money_at_risk_computation_and_na_when_unflagged(self):
        """
        Verify:
        (1) Money at Risk for uploads is computed only from uploaded flows of flagged accounts.
        (2) Never exceeds the upload total volume.
        (3) Shows 'n/a' when nothing is flagged (e.g. high threshold).
        (4) Completely removes the demo figure (Rs 30,29,456.92).
        """
        sample_csv = generate_sample_csv()
        sess_hdr = {"X-Session-ID": "session_test_mar"}
        upload_res = self.client.post(
            "/api/upload",
            content=sample_csv.encode("utf-8"),
            headers={"Content-Type": "text/csv", **sess_hdr},
        )
        self.assertEqual(upload_res.status_code, 200)
        up_data = upload_res.json()
        total_upload_vol = up_data["total_volume"]

        # Case A: Low threshold (0.05) flags accounts
        sum_flagged = self.client.get("/api/dashboard/summary?threshold=0.05", headers=sess_hdr).json()
        r_sum = sum_flagged["risk_summary"]
        d_sum = sum_flagged["dataset_summary"]
        self.assertGreater(r_sum["suspicious_accounts_count"], 0)
        mar = r_sum["money_at_risk"]
        self.assertIsNotNone(mar)
        self.assertGreater(mar, 0.0)
        # MUST NEVER exceed the upload total
        self.assertLessEqual(mar, total_upload_vol)
        # MUST NOT be the demo figure
        self.assertNotEqual(mar, 3029456.92)
        self.assertNotEqual(d_sum.get("fraud_volume"), 3029456.92)
        self.assertIn("₹", r_sum["money_at_risk_display"])

        # Case B: High threshold (0.999) flags NOTHING
        sum_unflagged = self.client.get("/api/dashboard/summary?threshold=0.999", headers=sess_hdr).json()
        r_unflagged = sum_unflagged["risk_summary"]
        d_unflagged = sum_unflagged["dataset_summary"]
        self.assertEqual(r_unflagged["suspicious_accounts_count"], 0)
        # MUST show "n/a" when nothing is flagged
        self.assertIsNone(r_unflagged["money_at_risk"])
        self.assertEqual(r_unflagged["money_at_risk_display"], "n/a")
        self.assertEqual(d_unflagged["money_at_risk_display"], "n/a")
        # Demo figure must NEVER appear
        self.assertNotEqual(r_unflagged["money_at_risk"], 3029456.92)
        self.assertNotEqual(d_unflagged.get("fraud_volume"), 3029456.92)

    def test_time_windows_real_count_for_upload(self):
        """Verify upload replaces '144 Time Windows' with the real time windows count."""
        sample_csv = generate_sample_csv()
        sess_hdr = {"X-Session-ID": "session_test_tw"}
        up_res = self.client.post(
            "/api/upload",
            content=sample_csv.encode("utf-8"),
            headers={"Content-Type": "text/csv", **sess_hdr},
        ).json()
        # 90 minutes / 30 min window = 4 windows
        real_tw = up_res["time_windows"]
        self.assertEqual(real_tw, 4)
        self.assertNotEqual(real_tw, 144)

        # /api/upload/status also returns real time windows
        st = self.client.get("/api/upload/status", headers=sess_hdr).json()
        self.assertEqual(st["time_windows"], 4)

        # /api/dashboard/summary dataset_summary also returns real time windows
        sum_data = self.client.get("/api/dashboard/summary", headers=sess_hdr).json()
        self.assertEqual(sum_data["dataset_summary"]["time_windows"], 4)

        # Resetting back to demo returns 144
        self.client.post("/api/upload/reset", headers=sess_hdr)
        demo_st = self.client.get("/api/upload/status", headers=sess_hdr).json()
        self.assertEqual(demo_st["time_windows"], 144)
        demo_sum = self.client.get("/api/dashboard/summary", headers=sess_hdr).json()
        self.assertEqual(demo_sum["dataset_summary"]["time_windows"], 144)

    def test_badge_uploaded_filename_and_demo_after_reset(self):
        """Verify badge shows 'UPLOADED: <file name>' during upload and 'DEMO (G2P)' after reset."""
        sample_csv = generate_sample_csv()
        sess_hdr = {
            "X-Session-ID": "session_test_badge",
            "X-Filename": "quarterly_audit_2026.csv",
        }
        upload_res = self.client.post(
            "/api/upload",
            content=sample_csv.encode("utf-8"),
            headers={"Content-Type": "text/csv", **sess_hdr},
        )
        self.assertEqual(upload_res.status_code, 200)
        self.assertEqual(upload_res.json()["dataset_name"], "quarterly_audit_2026.csv")

        # Status check reflects the uploaded filename
        st_res = self.client.get("/api/upload/status", headers={"X-Session-ID": "session_test_badge"}).json()
        self.assertTrue(st_res["is_uploaded"])
        self.assertEqual(st_res["dataset_name"], "quarterly_audit_2026.csv")

        # Reset back to demo
        reset_res = self.client.post("/api/upload/reset", headers={"X-Session-ID": "session_test_badge"})
        self.assertEqual(reset_res.status_code, 200)

        # Status now reports DEMO (G2P)
        demo_st = self.client.get("/api/upload/status", headers={"X-Session-ID": "session_test_badge"}).json()
        self.assertFalse(demo_st["is_uploaded"])
        self.assertEqual(demo_st["dataset_name"], "DEMO (G2P)")

    def test_new_browser_session_always_starts_on_demo_and_expires(self):
        """
        Verify:
        (1) A new browser session always starts on the demo (isolated from other sessions).
        (2) Upload state expires after TTL.
        """
        from datetime import datetime, timedelta, timezone

        sample_csv = generate_sample_csv()
        # Session A uploads
        self.client.post(
            "/api/upload",
            content=sample_csv.encode("utf-8"),
            headers={"Content-Type": "text/csv", "X-Session-ID": "session_tab_user_a"},
        )
        st_a = self.client.get("/api/upload/status", headers={"X-Session-ID": "session_tab_user_a"}).json()
        self.assertTrue(st_a["is_uploaded"])

        # Brand new browser session (different session ID generated by sessionStorage)
        st_new = self.client.get("/api/upload/status", headers={"X-Session-ID": "session_tab_brand_new"}).json()
        self.assertFalse(st_new["is_uploaded"])
        self.assertEqual(st_new["dataset_name"], "DEMO (G2P)")
        self.assertEqual(st_new["active_mode"], "default_demo")

        # Session with no header at all (anonymous browser session)
        st_anon = self.client.get("/api/upload/status").json()
        self.assertFalse(st_anon["is_uploaded"])
        self.assertEqual(st_anon["dataset_name"], "DEMO (G2P)")

        # Verify expiration: simulate TTL elapsed
        ds_a = upload_manager.get_dataset("client:session_tab_user_a")
        self.assertIsNotNone(ds_a)
        # Artificially set expires_at in the past
        ds_a.expires_at = datetime.now(timezone.utc) - timedelta(seconds=10)
        self.assertTrue(ds_a.is_expired())

        # Now querying should return demo because it expired
        st_a_expired = self.client.get("/api/upload/status", headers={"X-Session-ID": "session_tab_user_a"}).json()
        self.assertFalse(st_a_expired["is_uploaded"])
        self.assertEqual(st_a_expired["dataset_name"], "DEMO (G2P)")


if __name__ == "__main__":
    unittest.main()


