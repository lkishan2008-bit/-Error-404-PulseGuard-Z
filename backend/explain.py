"""
==============================================================================
PulseGuard-Z  -  Explainability & Fraud Signal Interpretability
==============================================================================
Project   : PulseGuard-Z (IBM Z Hackathon, event code SSF2026)
Team      : Error-404
File      : backend/explain.py

Purpose
-------
Provides explainability for the 8 temporal graph signals extracted by the TGNN
model during snapshot construction. Maps technical feature names (from
data/scores.csv) to human-interpretable fraud typologies, operational severity,
and recommended investigator actions in G2P (Government-to-Person) welfare
distribution systems.

Signals Catalog
---------------
1. forwarding_ratio  : Ratio of outbound to inbound transaction amounts
2. in_count          : Inflow transaction frequency in the monitoring window
3. out_count         : Outflow transaction frequency in the monitoring window
4. log_in_amount     : Logarithmic magnitude of incoming transfer volume
5. log_out_amount    : Logarithmic magnitude of outgoing transfer volume
6. unique_senders    : Number of distinct originators sending funds to the account
7. unique_receivers  : Number of distinct destinations receiving funds from the account
8. amount_std        : Standard deviation of transaction amounts in the window
==============================================================================
"""

from typing import Any, Dict, List, Optional


SIGNAL_CATALOG: Dict[str, Dict[str, Any]] = {
    "forwarding_ratio": {
        "name": "forwarding_ratio",
        "title": "High Fund Pass-Through Velocity (Mule Forwarding)",
        "category": "Velocity & Flow",
        "severity": "HIGH",
        "description": (
            "Measures the ratio of outbound funds to inbound funds within the 30-minute snapshot window. "
            "A high forwarding ratio close to or exceeding 1.0 indicates that money deposited into the "
            "account is immediately routed outward with minimal dwell time."
        ),
        "fraud_pattern": (
            "Mule Account / Transit Relay: Fraudsters utilize compromised or recruited accounts to "
            "rapidly drain incoming G2P social welfare disbursements into secondary accounts or cash-out "
            "terminals before detection rules can freeze the funds."
        ),
        "investigation_focus": (
            "Inspect the time delta between incoming welfare deposits and subsequent outgoing transfers. "
            "Check if destination accounts belong to known crypto off-ramps, high-risk agents, or foreign entities."
        ),
        "recommended_action": (
            "Apply a temporary automated hold on outbound transfers pending two-factor re-authentication or "
            "manual compliance review."
        ),
    },
    "in_count": {
        "name": "in_count",
        "title": "Abnormal Inbound Transaction Frequency (Fan-In Aggregation)",
        "category": "Network Velocity",
        "severity": "HIGH",
        "description": (
            "Reflects an unusually high burst of incoming transactions directed toward this single account "
            "within a short temporal snapshot."
        ),
        "fraud_pattern": (
            "Syndicate Collector / Aggregator Node: A central orchestrator or collection mule receiving "
            "stolen funds or micro-transfers diverted from multiple legitimate welfare beneficiaries."
        ),
        "investigation_focus": (
            "Examine whether multiple distinct citizen IDs or payment cards are directing stipends into "
            "this single repository account."
        ),
        "recommended_action": (
            "Flag account for syndicate investigation; correlate with common IP/device fingerprints across senders."
        ),
    },
    "out_count": {
        "name": "out_count",
        "title": "High Outbound Burst Rate (Rapid Fund Dispersal)",
        "category": "Network Velocity",
        "severity": "MEDIUM",
        "description": (
            "Indicates a high number of outgoing transactions initiated in rapid succession within the "
            "snapshot window."
        ),
        "fraud_pattern": (
            "Scripted Smurfing / Automated Drainage: Bot-driven or automated multi-split transfers designed "
            "to rapidly liquidate stolen funds across numerous downstream recipients."
        ),
        "investigation_focus": (
            "Verify whether transaction intervals follow automated bot periodicity (< 2 seconds between calls) "
            "or irregular human interaction."
        ),
        "recommended_action": (
            "Throttle API rate limits on this account and trigger step-up biometric/OTP verification on new transfers."
        ),
    },
    "log_in_amount": {
        "name": "log_in_amount",
        "title": "Excessive Cumulative Inbound Volume",
        "category": "Volume Anomaly",
        "severity": "HIGH",
        "description": (
            "Logarithm of total amount received. Highlights accounts absorbing capital volumes that far "
            "exceed the standard demographic profile of social protection program recipients."
        ),
        "fraud_pattern": (
            "Bulk Benefit Hijacking / Shell Entity: Fraud ring funneling massive aggregate payouts from "
            "multiple fabricated identity claims into a single account."
        ),
        "investigation_focus": (
            "Compare inbound volume against standard program entitlement caps ($200–$800 standard stipend limits)."
        ),
        "recommended_action": (
            "Freeze unspent balances immediately and audit the underlying identity documentation associated with the account."
        ),
    },
    "log_out_amount": {
        "name": "log_out_amount",
        "title": "Excessive Cumulative Outbound Volume",
        "category": "Volume Anomaly",
        "severity": "HIGH",
        "description": (
            "Logarithm of total amount transferred out. Highlights large-scale capital exit or massive "
            "liquidations executed over a short temporal interval."
        ),
        "fraud_pattern": (
            "Flash Cash-Out / Liquidation: Organized crime rings transferring large accumulated welfare "
            "balances to high-value merchant terminals, ATM networks, or external banking institutions."
        ),
        "investigation_focus": (
            "Trace recipient banking routes to identify primary off-ramps and point-of-sale terminals."
        ),
        "recommended_action": (
            "Notify destination bank of suspected illicit origin under inter-bank fraud sharing protocols."
        ),
    },
    "unique_senders": {
        "name": "unique_senders",
        "title": "High In-Degree Topology (Multi-Source Collection)",
        "category": "Graph Topology",
        "severity": "CRITICAL",
        "description": (
            "Measures the diversity of distinct source nodes sending money into this account. Normal beneficiaries "
            "receive payments from only 1 or 2 sources (government agency or employer), whereas collectors show high in-degree."
        ),
        "fraud_pattern": (
            "Mule Ring Funneling: Graph topology where dozens of compromised beneficiary accounts funneled "
            "their disbursements toward a single intermediary coordinator."
        ),
        "investigation_focus": (
            "Map 1-hop and 2-hop neighborhood graphs to reveal the full circumference of the compromised beneficiary ring."
        ),
        "recommended_action": (
            "Coordinate with G2P welfare ministry to notify impacted citizen accounts and revoke compromised credentials."
        ),
    },
    "unique_receivers": {
        "name": "unique_receivers",
        "title": "High Out-Degree Topology (Fan-Out Layering)",
        "category": "Graph Topology",
        "severity": "CRITICAL",
        "description": (
            "Measures the count of distinct destination accounts receiving money from this single node. "
            "High out-degree indicates deliberate fan-out layering across the financial network."
        ),
        "fraud_pattern": (
            "Smurfing & Layering Network: Dispersing large illicit sums into micro-portions across a web of "
            "downstream accounts to avoid statutory anti-money laundering (AML) reporting thresholds."
        ),
        "investigation_focus": (
            "Check if downstream receiver accounts were newly opened within the last 72 hours."
        ),
        "recommended_action": (
            "Expand watchlists to include all destination accounts in the fan-out cluster."
        ),
    },
    "amount_std": {
        "name": "amount_std",
        "title": "High Amount Volatility / Structuring Pattern",
        "category": "Statistical Variance",
        "severity": "MEDIUM",
        "description": (
            "Measures standard deviation of transaction amounts. Identifies either irregular wild spikes "
            "or synthetic perturbation used by fraudsters attempting to mimic benign consumer spending."
        ),
        "fraud_pattern": (
            "Evasion Structuring / Anomaly Spikes: Alternating between small micro-probes and massive withdrawal "
            "bursts to test account limits and avoid rule-based thresholds."
        ),
        "investigation_focus": (
            "Review transaction distribution histogram to detect non-standard pricing steps or repeated rounded amounts."
        ),
        "recommended_action": (
            "Apply enhanced transaction monitoring rules to observe subsequent transaction sequences."
        ),
    },
}


def get_all_signal_definitions() -> Dict[str, Dict[str, Any]]:
    """
    Returns the comprehensive dictionary of all signal explanations.
    """
    return SIGNAL_CATALOG


def explain_signal(signal_name: Optional[str]) -> Dict[str, Any]:
    """
    Returns the structured explanation for a single signal name.
    If the signal is unknown, returns a generic fallback.
    """
    if not signal_name:
        return {
            "name": "none",
            "title": "No Signal Identified",
            "category": "General",
            "severity": "LOW",
            "description": "No dominant behavioral feature was isolated for this snapshot.",
            "fraud_pattern": "Standard transactional activity.",
            "investigation_focus": "Standard account lifecycle review.",
            "recommended_action": "No immediate remediation required.",
        }

    key = str(signal_name).strip().lower()
    if key in SIGNAL_CATALOG:
        return SIGNAL_CATALOG[key]

    return {
        "name": key,
        "title": f"Custom Signal ({key})",
        "category": "Unclassified",
        "severity": "MEDIUM",
        "description": f"Feature '{key}' contributed significantly to the account's graph risk embedding.",
        "fraud_pattern": "Unusual graph or temporal deviation detected by the TGNN neural encoder.",
        "investigation_focus": "Analyze underlying tabular features in the relevant transaction window.",
        "recommended_action": "Review account transaction logs for irregular patterns.",
    }


def explain_signals_list(signals: List[str]) -> List[Dict[str, Any]]:
    """
    Takes a list of signal names and returns a list of detailed explanations.
    """
    return [explain_signal(s) for s in signals if s]


def assess_risk_level(tgnn_score: float, threshold: float = 0.20, rule_flagged: bool = False) -> str:
    """
    Determines an operational risk tier based on the TGNN score and calibrated threshold.
    Calibrated based on Phase 2 evaluation where score peak is ~0.288 due to extreme class imbalance (0.35%).
    """
    if tgnn_score >= 0.25:
        return "CRITICAL"
    elif tgnn_score >= threshold:
        return "HIGH"
    elif tgnn_score >= 0.15 or rule_flagged:
        return "ELEVATED"
    elif tgnn_score >= 0.05:
        return "MODERATE"
    else:
        return "LOW"


def generate_account_narrative(
    account_id: str,
    tgnn_score: float,
    rule_flagged: bool,
    top_signals: List[str],
    threshold: float = 0.20,
    account_type: str = "unknown",
) -> Dict[str, Any]:
    """
    Synthesizes a human-readable investigation brief explaining the model's risk score,
    how the top signals interact, and why the account warrants attention.
    """
    is_suspicious = (tgnn_score >= threshold) or bool(rule_flagged)
    risk_level = assess_risk_level(tgnn_score, threshold, rule_flagged)
    clean_signals = [s for s in top_signals if s and str(s).lower() != "nan"]

    # Build plain-English summary paragraphs
    summary_parts = []
    summary_parts.append(
        f"Account {account_id} ({account_type.capitalize()}) was evaluated by the Temporal Graph Neural Network (TGNN). "
        f"It produced a risk score of {tgnn_score:.4f} (operational threshold = {threshold:.2f})."
    )

    if is_suspicious:
        summary_parts.append(
            f"The account is classified as {risk_level} risk. "
            f"{'Static rule-based checks also flagged this entity.' if rule_flagged else 'Traditional static rules missed this account; detection relies on temporal graph topology.'}"
        )
    else:
        summary_parts.append(
            f"The account is within normal operational risk parameters ({risk_level} risk). "
            f"Static rules did not flag this entity."
        )

    # Narrative on signals
    signal_explanations = [explain_signal(s) for s in clean_signals]
    signal_titles = [s["title"] for s in signal_explanations]

    if clean_signals:
        summary_parts.append(
            f"The top driving graph signals isolated by the model are: {', '.join(signal_titles)}."
        )

        # Contextual synergy analysis
        sig_set = set(clean_signals)
        if "forwarding_ratio" in sig_set and ("in_count" in sig_set or "unique_senders" in sig_set):
            summary_parts.append(
                "Synergy detected: The co-occurrence of high forwarding ratio with multi-source inbound aggregation "
                "strongly correlates with a classic welfare money mule pattern (pass-through node)."
            )
        elif "unique_receivers" in sig_set and "log_out_amount" in sig_set:
            summary_parts.append(
                "Synergy detected: Rapid fan-out across multiple unique destinations paired with high outbound volume "
                "is indicative of automated fund smurfing or mass cash-out."
            )
        elif "amount_std" in sig_set and "log_out_amount" in sig_set:
            summary_parts.append(
                "Synergy detected: Volatile payment amounts alongside substantial outbound volume indicates potential "
                "evasion structuring."
            )

    narrative_text = " ".join(summary_parts)

    recommendations = []
    if risk_level in {"CRITICAL", "HIGH"}:
        recommendations.append("Immediate compliance hold on pending outbound disbursements.")
        recommendations.append("Audit beneficial ownership and linked social welfare beneficiary claims.")
        recommendations.append("Trace outgoing funds across counterparty financial institutions.")
    elif risk_level == "ELEVATED":
        recommendations.append("Queue for Level 2 compliance officer review within 24 hours.")
        recommendations.append("Enable high-resolution transaction logging and alert triggers.")
    else:
        recommendations.append("Maintain routine automated monitoring; no active intervention necessary.")

    return {
        "account_id": account_id,
        "account_type": account_type,
        "tgnn_score": tgnn_score,
        "rule_flagged": bool(rule_flagged),
        "is_suspicious": is_suspicious,
        "risk_level": risk_level,
        "threshold": threshold,
        "top_signals": clean_signals,
        "signals_detail": signal_explanations,
        "narrative": narrative_text,
        "recommended_actions": recommendations,
        "data_notice": "Analysis generated from synthetic benchmark dataset simulating IBM LinuxONE G2P flash fraud.",
    }
