"""
scripts/seed_demo.py
─────────────────────────────────────────────────────────────────────────────
Demo seeding script — creates sample documents and runs test queries.

Usage:
    python scripts/seed_demo.py

Requires the API to be running: make up
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations
import sys, time, textwrap, json
sys.path.insert(0, ".")

import requests

API  = "http://localhost:8000/api/v1"
AUTH = {"username": "admin", "password": "changeme"}

SAMPLE_DOCS = [
    {
        "filename":  "leave_policy.txt",
        "content":   textwrap.dedent("""\
            EMPLOYEE LEAVE POLICY
            =====================
            Annual Leave
            All full-time employees are entitled to 20 days of annual leave per year.
            Leave must be approved by the line manager at least 5 working days in advance.

            Sick Leave
            Employees are entitled to 10 days of paid sick leave per year.
            A medical certificate is required for absences exceeding 3 consecutive days.

            Maternity Leave
            Female employees are entitled to 16 weeks of paid maternity leave.
            This may commence up to 4 weeks before the expected delivery date.

            Emergency Leave
            Up to 3 days of emergency leave may be granted for unforeseen personal circumstances.
        """),
        "department": "HR",
        "doc_category": "policy",
    },
    {
        "filename":  "expense_reimbursement_sop.txt",
        "content":   textwrap.dedent("""\
            EXPENSE REIMBURSEMENT STANDARD OPERATING PROCEDURE
            ====================================================
            1. Submit expenses within 30 days of incurring them.
            2. Attach original receipts for all expenses above $25.
            3. Meal allowance is capped at $50 per day during business travel.
            4. Hotel accommodation must not exceed $200 per night without pre-approval.
            5. Economy class is required for all flights under 6 hours.
            6. Submit the completed form to finance@company.com for processing.
            7. Reimbursements are processed within 5 business days of approval.
        """),
        "department": "Finance",
        "doc_category": "SOP",
    },
    {
        "filename":  "it_security_policy.txt",
        "content":   textwrap.dedent("""\
            IT SECURITY POLICY
            ==================
            Password Requirements
            Passwords must be at least 12 characters and include uppercase, lowercase,
            numbers and special characters. Passwords must be changed every 90 days.

            Device Security
            All company devices must have full-disk encryption enabled.
            Employees must lock their screen when leaving their desk.
            Personal devices must not be used to access company data without MDM enrollment.

            Data Classification
            Confidential: Customer PII, financial records, source code.
            Internal: Internal communications, project plans.
            Public: Marketing materials, public documentation.

            Incident Reporting
            Security incidents must be reported to security@company.com within 1 hour of discovery.
        """),
        "department": "IT",
        "doc_category": "policy",
    },
]

DEMO_QUESTIONS = [
    "How many days of annual leave do employees get?",
    "What is the meal allowance during business travel?",
    "When must security incidents be reported?",
    "What are the password requirements?",
]


def main():
    print("=" * 60)
    print("EKIP Demo Seed Script")
    print("=" * 60)

    # 1. Login
    print("\n[1/3] Authenticating…")
    resp = requests.post(f"{API}/auth/token", json=AUTH)
    if not resp.ok:
        print(f"❌ Login failed: {resp.text}")
        sys.exit(1)
    token   = resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    print("✅ Authenticated")

    # 2. Upload sample docs
    print("\n[2/3] Uploading sample documents…")
    doc_ids = []
    for doc in SAMPLE_DOCS:
        content = doc["content"].encode("utf-8")
        resp    = requests.post(
            f"{API}/documents/upload",
            files={"file": (doc["filename"], content, "text/plain")},
            data={
                "department":   doc.get("department", ""),
                "doc_category": doc.get("doc_category", ""),
            },
            headers=headers,
        )
        if resp.ok:
            data = resp.json()["data"]
            doc_ids.append(data["document_id"])
            print(f"  ✅ {doc['filename']}  →  doc_id: {data['document_id'][:8]}…")
        else:
            print(f"  ❌ {doc['filename']}: {resp.text}")

    # 3. Wait for ingestion (polling)
    print("\n[3/3] Waiting for ingestion…")
    for doc_id in doc_ids:
        for _ in range(30):
            resp   = requests.get(f"{API}/documents/{doc_id}/status", headers=headers)
            status = resp.json()["data"]["status"] if resp.ok else "unknown"
            if status in ("indexed", "failed"):
                icon = "✅" if status == "indexed" else "❌"
                print(f"  {icon} {doc_id[:8]}… → {status}")
                break
            time.sleep(2)

    # 4. Demo questions (will be stubs until RAG is fully wired)
    print("\n[Demo] Running sample questions…")
    print("─" * 60)
    for q in DEMO_QUESTIONS:
        resp = requests.post(f"{API}/ask/", json={"question": q}, headers=headers)
        if resp.ok:
            answer = resp.json()["data"]["answer"]
            print(f"\nQ: {q}")
            print(f"A: {answer[:200]}{'…' if len(answer) > 200 else ''}")
        else:
            print(f"\nQ: {q}\nA: ❌ {resp.text[:100]}")

    print("\n" + "=" * 60)
    print("Done! Open http://localhost:8501 to explore the UI.")
    print("=" * 60)


if __name__ == "__main__":
    main()
