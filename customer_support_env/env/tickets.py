"""
Ticket scenario database for CustomerSupportEnv.
Each ticket includes: metadata, customer history, KB articles,
canonical solution, and keyword-based solution validator.
"""
from __future__ import annotations
from typing import Dict, List, Any


TICKETS: Dict[str, Dict[str, Any]] = {
    "TKT-001": {
        "subject": "Cannot log in to my account",
        "customer": "Aria Shah",
        "priority": "high",
        "category": "auth",
        "sentiment": "frustrated",
        "history": [
            {"role": "customer", "text": "I've been locked out for 2 days! I tried resetting my password three times and nothing works. This is extremely urgent.", "turn": 0}
        ],
        "kb_articles": [
            "Password reset: Visit /forgot-password and enter your registered email. Reset links expire in 15 minutes.",
            "Account lockout policy: Accounts lock after 5 failed attempts. Auto-unlock after 30 minutes, or contact support for manual unlock.",
            "2FA issues: If locked out due to 2FA, an admin can bypass the second factor temporarily via the admin console."
        ],
        "canonical_solution": "I have manually unlocked your account and sent a fresh password reset link to your registered email. The link will expire in 15 minutes. If 2FA is causing issues I can temporarily bypass it.",
        "solution_keywords": ["unlock", "reset", "link", "email", "password"],
        "customer_followups": [
            "Thank you! I got the email and it worked.",
            "That fixed it, appreciate your help.",
            "Great, I'm back in now."
        ]
    },
    "TKT-002": {
        "subject": "Wrong item shipped — order #482923",
        "customer": "Bryce Lee",
        "priority": "urgent",
        "category": "fulfillment",
        "sentiment": "angry",
        "history": [
            {"role": "customer", "text": "This is unacceptable. I ordered a Red T-Shirt size L but you sent me a Blue size M. Order #482923. I need the right item immediately.", "turn": 0}
        ],
        "kb_articles": [
            "Return policy: Customers have 30 days to initiate a return. Use the portal at /returns. We cover return shipping for our errors.",
            "Priority re-ship: For fulfilment errors on orders >$25, approve a priority reship within 24h after return label is issued. No need to wait for return arrival.",
            "Compensation policy: For urgent orders or repeat fulfilment errors, issue a 15% discount code on next purchase."
        ],
        "canonical_solution": "I sincerely apologise. I've raised a priority reship for the Red T-Shirt size L — it will ship within 24 hours. I've emailed a pre-paid return label for the incorrect item, and added a 15% discount code to your account for the inconvenience.",
        "solution_keywords": ["reship", "return", "label", "correct", "apologise", "apologi", "discount"],
        "customer_followups": [
            "OK, as long as it ships today I'm fine with that.",
            "Got the email with the label. Thank you.",
            "Alright, I appreciate the quick response."
        ]
    },
    "TKT-003": {
        "subject": "Invoice #8821 shows wrong amount",
        "customer": "Cleo Park",
        "priority": "medium",
        "category": "billing",
        "sentiment": "neutral",
        "history": [
            {"role": "customer", "text": "Hello, invoice #8821 shows $49 but I'm on the $29/month Basic plan. I downgraded last month. Can you check?", "turn": 0}
        ],
        "kb_articles": [
            "Plan changes: Downgrades take effect at the start of the next billing cycle. The current period is charged at the old rate.",
            "Prorate credits: If a downgrade was confirmed before the cycle closed, a manual credit can be issued for the difference.",
            "Billing disputes: Finance team can adjust invoices within 60 days of issue date. Requires the invoice number and account email."
        ],
        "canonical_solution": "I've reviewed your account. Your downgrade was confirmed before the billing cycle closed, so I'm issuing a $20 credit to your account which will appear on your next invoice. Going forward you will be billed $29/month.",
        "solution_keywords": ["credit", "$20", "twenty", "correct", "billing", "downgrade", "refund"],
        "customer_followups": [
            "Perfect, that makes sense. Thanks.",
            "Great, I can see the credit on my account.",
            "Thanks for sorting that out quickly."
        ]
    },
    "TKT-004": {
        "subject": "App crashes on iOS 17 during PDF export",
        "customer": "Dev Okonkwo",
        "priority": "medium",
        "category": "bug",
        "sentiment": "neutral",
        "history": [
            {"role": "customer", "text": "Every time I tap 'Export PDF' the app force-quits. iPhone 14 Pro, iOS 17.4.1. Started after the last app update.", "turn": 0}
        ],
        "kb_articles": [
            "Known iOS 17 crash: The PDF export feature has a memory issue on iOS 17.3 and above introduced in app v4.1.0. Fix is in v4.2.1.",
            "Workaround: Use the web app at app.example.com/export for PDF exports until v4.2.1 is released (ETA: 5 business days).",
            "Bug reporting: Collect crash logs from Settings > Privacy > Analytics & Improvements > Analytics Data and share with devs@example.com."
        ],
        "canonical_solution": "This is a known bug in v4.1.0 on iOS 17.3+ — our engineering team has a fix ready in v4.2.1, releasing in 5 days. In the meantime, use our web app at app.example.com/export. I've also flagged your report to the engineering team.",
        "solution_keywords": ["known", "bug", "v4.2", "workaround", "web", "fix", "engineering"],
        "customer_followups": [
            "Good to know it's being fixed. I'll use the web app for now.",
            "Thanks for the workaround, that works.",
            "OK, I'll wait for the update."
        ]
    },
    "TKT-005": {
        "subject": "Bulk licence pricing for 50 seats",
        "customer": "Emma Ng",
        "priority": "low",
        "category": "sales",
        "sentiment": "positive",
        "history": [
            {"role": "customer", "text": "Hi! We're a team of about 50 and are considering your Pro plan. Do you offer bulk discounts? Also, is there an enterprise contract option?", "turn": 0}
        ],
        "kb_articles": [
            "Volume discounts: 10–24 seats: 10% off. 25–49 seats: 15% off. 50+ seats: 25% off annual plan.",
            "Enterprise contracts: Custom SLA, SSO, dedicated support, and invoice billing. Contact sales@example.com. Average deal closes in 2 weeks.",
            "Trial: Teams of 5+ can get a 30-day free trial of the Pro plan. No credit card required."
        ],
        "canonical_solution": "Great news — 50 seats qualifies for our 25% volume discount on the annual Pro plan. We also offer enterprise contracts with SSO, dedicated support, and custom SLA. I'd love to connect you with our enterprise team at sales@example.com, or I can have an account executive reach out directly.",
        "solution_keywords": ["25%", "twenty-five", "enterprise", "volume", "discount", "sales@", "executive"],
        "customer_followups": [
            "That sounds great, please have someone reach out.",
            "25% is better than I expected! I'll email sales.",
            "Perfect, we'll set up a call with the enterprise team."
        ]
    },
    "TKT-006": {
        "subject": "Data export taking over 6 hours",
        "customer": "Felix Martín",
        "priority": "high",
        "category": "bug",
        "sentiment": "frustrated",
        "history": [
            {"role": "customer", "text": "I started a full data export 6 hours ago and it's still at 12%. I have a compliance deadline tomorrow. This is critical.", "turn": 0}
        ],
        "kb_articles": [
            "Export timeouts: Large exports (>10GB) can time out. The system retries automatically but may take 8-12 hours total.",
            "Priority export queue: Support can manually move a job to the priority queue, cutting estimated time to 1-2 hours.",
            "Partial exports: Users can export data by date range to reduce file size. Recommended for compliance: export by quarter."
        ],
        "canonical_solution": "I've moved your export job to the priority queue — it should complete within 1-2 hours. As a backup, I recommend also starting a partial export by date range which will be much faster. I'll monitor and send you a confirmation email when the full export completes.",
        "solution_keywords": ["priority", "queue", "1-2 hour", "partial", "monitor", "email"],
        "customer_followups": [
            "Thank you! I'll start the partial export as backup.",
            "OK, I can see the progress picked up. Thanks.",
            "The priority queue worked, it's done now."
        ]
    }
}


def get_ticket(ticket_id: str) -> Dict[str, Any]:
    if ticket_id not in TICKETS:
        raise ValueError(f"Unknown ticket_id: {ticket_id}")
    return TICKETS[ticket_id]


def all_ticket_ids() -> List[str]:
    return list(TICKETS.keys())
