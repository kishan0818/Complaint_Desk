"""
Evaluation script for Activity 18.2 — Local Ollama (Mistral).
Measures classification accuracy and latency (cold-start vs warm-state) across the 10 benchmark complaints.
"""

import sys
import csv
import time
import requests
from pathlib import Path
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Force UTF-8 stdout for clean console output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

ALLOWED_CATEGORIES = ["billing", "loan", "fraud", "app_issue"]

def normalize_category(raw_output: str) -> str:
    cleaned = raw_output.strip().lower().replace(".", "").replace(",", "")
    for cat in ALLOWED_CATEGORIES:
        if cat in cleaned:
            return cat
    return "billing"

def run_evaluation():
    csv_path = Path(__file__).parent / "test_cases.csv"
    if not csv_path.exists():
        print(f"ERROR: Test file not found at {csv_path}")
        sys.exit(1)

    # Health check
    try:
        res = requests.get("http://localhost:11434/api/tags", timeout=2)
        if res.status_code != 200:
            raise ConnectionError("Ollama returned non-200")
    except Exception:
        print("=" * 75)
        print("❌ ERROR: Ollama server is not running at http://localhost:11434")
        print("Please start Ollama in a separate terminal: 'ollama serve'")
        print("And make sure mistral is installed: 'ollama pull mistral'")
        print("=" * 75)
        sys.exit(1)

    print("=" * 75)
    print("🚀 Initializing ChatOllama(model='mistral', temperature=0.3)...")
    print("=" * 75)
    
    llm = ChatOllama(
        model="mistral",
        temperature=0.3,
        base_url="http://localhost:11434"
    )

    classification_prompt = ChatPromptTemplate.from_template(
        """Classify this customer complaint into exactly ONE of these four categories:
billing
loan
fraud
app_issue

Few-shot examples:
- "I was charged an extra fee on my EMI statement." -> billing
- "My account was debited twice for a grocery purchase." -> billing
- "How can I apply for a personal loan and what is the interest rate?" -> loan
- "My loan application is still pending after 2 weeks." -> loan
- "I do not recognize a transaction on my debit card." -> fraud
- "Someone used my card credentials without permission." -> fraud
- "The mobile app crashes on the login screen." -> app_issue
- "I cannot download my bank statement from the app." -> app_issue

Rules:
1. Output ONLY the single category word in lowercase (billing, loan, fraud, or app_issue).
2. Do not include punctuation, explanations, or introductory text.

Customer Complaint:
{text}

Category:"""
    )

    chain = classification_prompt | llm | StrOutputParser()

    total = 0
    correct = 0
    incorrect = 0
    latencies = []

    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            cid = row.get("id", str(total))
            complaint = row["complaint"]
            expected = row.get("category", row.get("expected_category", "")).strip().lower()

            t0 = time.time()
            try:
                raw_pred = chain.invoke({"text": complaint})
                predicted = normalize_category(raw_pred)
                lat = time.time() - t0
                latencies.append(lat)

                is_correct = (predicted == expected)
                if is_correct:
                    correct += 1
                    status = "✅ PASS"
                else:
                    incorrect += 1
                    status = "❌ FAIL"

                print(f"[{cid:>2}] [{status}] Expected: {expected:<10} | Predicted: {predicted:<10} | Latency: {lat:6.2f}s | Complaint: {complaint[:30]}...", flush=True)
            except Exception as e:
                lat = time.time() - t0
                latencies.append(lat)
                incorrect += 1
                print(f"[{cid:>2}] [⚠️ ERROR] Error: {e} | Latency: {lat:6.2f}s", flush=True)

    acc = (correct / total) * 100 if total > 0 else 0.0
    avg_lat = sum(latencies) / len(latencies) if latencies else 0.0
    cold_lat = latencies[0] if latencies else 0.0
    warm_lat = sum(latencies[1:]) / (len(latencies) - 1) if len(latencies) > 1 else avg_lat

    print("=" * 75)
    print("📊 Ollama Mistral Evaluation Summary:")
    print(f"  Total Test Cases:                       {total}")
    print(f"  Correct Classifications:                {correct}")
    print(f"  Accuracy:                               {acc:.2f}%")
    print(f"  Cold-Start Latency (Case 1):            {cold_lat:.2f}s")
    print(f"  Warm-State Average Latency (Cases 2-10):{warm_lat:.2f}s")
    print(f"  Overall Average Latency:                {avg_lat:.2f}s")
    print("=" * 75)

if __name__ == "__main__":
    run_evaluation()
