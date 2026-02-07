"""BPO (Business Process Outsourcing) scoring logic."""
from typing import Tuple


# Keywords that indicate BPO-heavy operations
BPO_SIGNAL_KEYWORDS = [
    "outsourcing",
    "bpo",
    "business process outsourcing",
    "managed services",
    "operations outsourcing",
    "back-office",
    "backoffice",
    "processing services",
    "operational support",
    "ops team",
    "headcount",
    "fte",
    "full-time equivalent",
    "offshore",
    "nearshore",
    "service delivery",
    "run operations",
    "operational takeover",
]

# Keywords that indicate software/product-led business
SOFTWARE_SIGNAL_KEYWORDS = [
    "saas",
    "software platform",
    "api",
    "cloud-native",
    "self-service",
    "automation",
    "workflow",
    "no-code",
    "low-code",
    "integration",
    "real-time",
    "dashboard",
    "analytics platform",
]


def compute_bpo_score(candidate: dict) -> Tuple[int, str]:
    """
    Compute BPO likelihood score (0-100) from candidate signals.
    
    Returns:
        Tuple of (score, rationale)
    """
    bpo_signals = candidate.get("bpo_signals", [])
    software_signals = candidate.get("software_signals", [])
    why_fit = candidate.get("why_fit", "").lower()
    
    # Count BPO indicators
    bpo_count = 0
    matched_bpo = []
    
    for signal in bpo_signals:
        signal_lower = signal.lower()
        for keyword in BPO_SIGNAL_KEYWORDS:
            if keyword in signal_lower:
                bpo_count += 1
                matched_bpo.append(keyword)
                break
    
    # Also check the why_fit text
    for keyword in BPO_SIGNAL_KEYWORDS:
        if keyword in why_fit:
            bpo_count += 0.5
            if keyword not in matched_bpo:
                matched_bpo.append(keyword)
    
    # Count software indicators (reduce score)
    software_count = 0
    matched_software = []
    
    for signal in software_signals:
        signal_lower = signal.lower()
        for keyword in SOFTWARE_SIGNAL_KEYWORDS:
            if keyword in signal_lower:
                software_count += 1
                matched_software.append(keyword)
                break
    
    for keyword in SOFTWARE_SIGNAL_KEYWORDS:
        if keyword in why_fit:
            software_count += 0.5
            if keyword not in matched_software:
                matched_software.append(keyword)
    
    # Compute score
    # Base score influenced by ratio of BPO to software signals
    if bpo_count == 0 and software_count == 0:
        score = 30  # Unknown, moderate risk
    elif software_count == 0:
        score = min(90, 40 + bpo_count * 15)
    elif bpo_count == 0:
        score = max(10, 30 - software_count * 10)
    else:
        ratio = bpo_count / (bpo_count + software_count)
        score = int(20 + ratio * 70)
    
    # Build rationale
    rationale_parts = []
    if matched_bpo:
        rationale_parts.append(f"BPO signals detected: {', '.join(list(set(matched_bpo))[:3])}")
    if matched_software:
        rationale_parts.append(f"Software signals detected: {', '.join(list(set(matched_software))[:3])}")
    if not rationale_parts:
        rationale_parts.append("Limited signals available for assessment")
    
    rationale = "; ".join(rationale_parts)
    
    return score, rationale

