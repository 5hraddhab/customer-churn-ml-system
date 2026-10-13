"""
Configurable Retention Recommendations Module
Provides contextual, industry-aware retention action suggestions based on
customer risk tiers, key driving factors, and behavioral indicators.
Presents recommendations as actionable suggestions rather than guaranteed outcomes.
"""

from typing import Any, Dict, List, Optional


INDUSTRY_PROFILES = {
    "General / Configurable": {
        "description": "Cross-industry behavioral rules based on risk levels and customer engagement signals.",
        "strategies": [
            {
                "trigger_check": lambda c, r: r.get("risk_tier") == "High Risk",
                "title": "Proactive Account Specialist Outreach",
                "priority": "High",
                "rationale": "Customer exhibits high churn probability. Immediate proactive human outreach allows discovering root causes before cancellation.",
                "action": "Schedule a personalized review call or direct concierge inquiry within 24-48 hours.",
            },
            {
                "trigger_check": lambda c, r: _get_num(c, ["complaint", "ticket", "support"]) >= 2,
                "title": "Service Recovery & Resolution Audit",
                "priority": "High",
                "rationale": "Customer has accumulated multiple service or support complaints, indicating friction in customer experience.",
                "action": "Audit pending tickets, escalate unresolved issues to a senior support lead, and offer courtesy service credits.",
            },
            {
                "trigger_check": lambda c, r: _get_num(c, ["satisfaction", "csat", "nps"]) <= 4 and _get_num(c, ["satisfaction", "csat", "nps"]) > 0,
                "title": "Customer Satisfaction Remediation",
                "priority": "High",
                "rationale": "Reported satisfaction score is low, signaling disappointment with product delivery or service response.",
                "action": "Trigger an executive satisfaction follow-up survey and provide a tailored compensation or goodwill gesture.",
            },
            {
                "trigger_check": lambda c, r: _get_num(c, ["late_payment", "overdue", "payment"]) >= 1,
                "title": "Flexible Billing & Payment Review",
                "priority": "Medium",
                "rationale": "Record indicates billing friction or late payments, which often leads to involuntary churn or financial fatigue.",
                "action": "Offer structured payment schedules, grace period reminders, or optimized billing dates.",
            },
            {
                "trigger_check": lambda c, r: _get_num(c, ["usage", "hours", "activity", "streaming"]) < 50 and _get_num(c, ["usage", "hours", "activity"]) >= 0,
                "title": "Engagement Re-Activation Campaign",
                "priority": "Medium",
                "rationale": "Customer exhibits below-average usage or activity, suggesting declining perceived value.",
                "action": "Send guided feature walkthroughs, relevant use-case spotlights, or a 1-on-1 adoption refresher.",
            },
            {
                "trigger_check": lambda c, r: _get_num(c, ["spending", "monthly", "mrr", "charges"]) > 100,
                "title": "High-Value Account Loyalty Tier Enrollment",
                "priority": "Medium",
                "rationale": "Customer is in the upper spending bracket; preserving this account protects significant business revenue.",
                "action": "Enroll account in VIP support priority, offer bespoke contract terms or loyalty reward points.",
            },
        ],
    },
    "Banking & Financial Services": {
        "description": "Tailored for retail and commercial banks, credit unions, and fintech apps.",
        "strategies": [
            {
                "trigger_check": lambda c, r: r.get("risk_tier") == "High Risk",
                "title": "Dedicated Relationship Manager Intervention",
                "priority": "High",
                "rationale": "High churn risk in banking often precedes account closure or primary bank migration.",
                "action": "Assign personal banker to review portfolio health, fee waivers, and customized credit or deposit rates.",
            },
            {
                "trigger_check": lambda c, r: _get_num(c, ["late_payment", "overdue"]) >= 1,
                "title": "Financial Advisory & Restructuring Options",
                "priority": "High",
                "rationale": "Repayment delays signal financial stress; supportive banking strengthens long-term loyalty.",
                "action": "Offer flexible repayment plans, temporary interest adjustments, or debt consolidation advisory.",
            },
            {
                "trigger_check": lambda c, r: _get_num(c, ["complaint", "support"]) >= 1,
                "title": "Compliance & Branch Escalation Review",
                "priority": "High",
                "rationale": "Financial service complaints have high regulatory and attrition impact.",
                "action": "Expedite case review with branch manager and send official resolution letter with fee reimbursement where applicable.",
            },
            {
                "trigger_check": lambda c, r: True,
                "title": "Cross-Product Value Reinforcement",
                "priority": "Low",
                "rationale": "Customers with multiple banking products (checking, savings, credit) have significantly lower churn rates.",
                "action": "Incentivize multi-product adoption with cash-back incentives or bundled savings account bonuses.",
            },
        ],
    },
    "Telecommunications & ISPs": {
        "description": "Tailored for mobile network operators, broadband providers, and fiber ISPs.",
        "strategies": [
            {
                "trigger_check": lambda c, r: _get_num(c, ["support", "calls", "tickets"]) >= 3,
                "title": "Network Quality & Hardware Diagnostic",
                "priority": "High",
                "rationale": "Frequent support calls in telecom typically stem from connectivity drops, throttling, or hardware faults.",
                "action": "Dispatch automated line diagnostic test, offer free router upgrade or technician inspection.",
            },
            {
                "trigger_check": lambda c, r: _get_num(c, ["spending", "monthly"]) > 80,
                "title": "Rate-Plan Optimization Review",
                "priority": "Medium",
                "rationale": "Competitors frequently target high-bill telecom customers with lower-cost bundle porting promotions.",
                "action": "Offer a locked promotional tariff or complimentary data/speed booster for a 12-month contract renewal.",
            },
            {
                "trigger_check": lambda c, r: r.get("risk_tier") in ["Medium Risk", "High Risk"],
                "title": "Contract Renewal & Loyalty Perk",
                "priority": "High",
                "rationale": "At-risk telecom subscribers are likely near contract expiration or comparing competitive offers.",
                "action": "Present exclusive device upgrade discount or zero-cost streaming bundle add-on.",
            },
        ],
    },
    "E-Commerce & Retail": {
        "description": "Tailored for online retail stores, marketplaces, and omnichannel brands.",
        "strategies": [
            {
                "trigger_check": lambda c, r: _get_num(c, ["complaint", "tickets"]) >= 1,
                "title": "Post-Order Satisfaction Assurance",
                "priority": "High",
                "rationale": "Negative delivery or product experiences rapidly alienate e-commerce buyers.",
                "action": "Issue instant store credit voucher and expedited return/replacement processing.",
            },
            {
                "trigger_check": lambda c, r: _get_num(c, ["discount", "promo"]) > 50,
                "title": "Personalized Promo Incentive",
                "priority": "Medium",
                "rationale": "Customer responds favorably to promotional campaigns and discounts.",
                "action": "Send targeted category coupon code with expiration within 7 days to trigger repeat purchase.",
            },
            {
                "trigger_check": lambda c, r: r.get("risk_tier") == "High Risk",
                "title": "VIP Free Shipping & Exclusive Previews",
                "priority": "High",
                "rationale": "Dormant high-risk shoppers need high perceived value incentives to re-engage.",
                "action": "Grant complimentary 90-day express shipping perk and early access to upcoming sales.",
            },
        ],
    },
    "Subscription Services & SaaS": {
        "description": "Tailored for recurring subscription platforms, media streaming, and B2B SaaS.",
        "strategies": [
            {
                "trigger_check": lambda c, r: _get_num(c, ["usage", "hours", "streaming"]) < 30,
                "title": "Product Re-Engagement & Content Discovery",
                "priority": "High",
                "rationale": "Low product activity is the single strongest precursor to subscription cancellation.",
                "action": "Deliver personalized content recommendations, tailored playlist/workflow digests, and in-app tooltips.",
            },
            {
                "trigger_check": lambda c, r: r.get("risk_tier") == "High Risk",
                "title": "Subscription Pause or Tier Downgrade Option",
                "priority": "High",
                "rationale": "Offering a temporary pause or lower tier prevents total customer loss when facing budget constraints.",
                "action": "Present a 1-3 month subscription pause option or discounted basic plan before final checkout cancellation.",
            },
            {
                "trigger_check": lambda c, r: _get_num(c, ["support", "tickets"]) >= 2,
                "title": "Customer Success Health Check",
                "priority": "Medium",
                "rationale": "Unresolved workflow blockers prevent customers from realizing product value.",
                "action": "Connect customer with dedicated Customer Success Specialist for a live workflow optimization session.",
            },
        ],
    },
}


def _get_num(customer_dict: Dict[str, Any], substrings: List[str]) -> float:
    """
    Helper function to safely extract numerical values from keys matching substrings.
    """
    for k, v in customer_dict.items():
        k_lower = str(k).lower()
        if any(sub in k_lower for sub in substrings):
            try:
                return float(v)
            except (ValueError, TypeError):
                continue
    return 0.0


def generate_retention_recommendations(
    customer_dict: Dict[str, Any],
    prediction_result: Dict[str, Any],
    industry: str = "General / Configurable",
    custom_actions: Optional[List[Dict[str, str]]] = None,
) -> List[Dict[str, Any]]:
    """
    Generates actionable retention recommendations tailored to customer attributes,
    prediction risk tier, and selected business profile.
    Presents recommendations as actionable hypotheses rather than guaranteed fixes.
    """
    profile = INDUSTRY_PROFILES.get(industry, INDUSTRY_PROFILES["General / Configurable"])
    strategies = profile["strategies"]

    recommendations = []

    for item in strategies:
        try:
            if item["trigger_check"](customer_dict, prediction_result):
                recommendations.append({
                    "title": item["title"],
                    "priority": item["priority"],
                    "rationale": item["rationale"],
                    "suggested_action": item["action"],
                })
        except Exception:
            continue

    # Incorporate user-defined custom retention actions if provided
    if custom_actions:
        for custom in custom_actions:
            recommendations.append({
                "title": custom.get("title", "Custom Action"),
                "priority": custom.get("priority", "Medium"),
                "rationale": custom.get("rationale", "User-specified business rule."),
                "suggested_action": custom.get("action", "Follow organizational retention protocol."),
            })

    # Default fallback if no specific triggers fired
    if not recommendations:
        if prediction_result.get("risk_tier") == "Low Risk":
            recommendations.append({
                "title": "Loyalty Appreciation & Advocacy Invitation",
                "priority": "Low",
                "rationale": "Customer exhibits low churn probability and positive engagement metrics.",
                "action": "Send thank-you message and invite to customer advisory community or referral rewards program.",
            })
        else:
            recommendations.append({
                "title": "General Retention Check-In",
                "priority": "Medium",
                "rationale": "Moderate risk profile requires active relationship monitoring.",
                "action": "Send brief satisfaction pulse survey and highlight relevant underutilized benefits.",
            })

    return recommendations
