"""
Automated Evaluation Script for Activity 18.1: Complaint Desk (Cloud LLM)
Evaluates the LangChain LCEL classification chain against benchmark test cases.
Supports OpenAI, NVIDIA NIM, and Groq automatically based on configured keys.
"""

import os
import csv
import sys
import time
from pathlib import Path

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# pyrefly: ignore [missing-import]
from langchain_core.prompts import ChatPromptTemplate
# pyrefly: ignore [missing-import]
from langchain_core.output_parsers import StrOutputParser

ALLOWED_CATEGORIES = ["billing", "loan", "fraud", "app_issue"]

def find_api_key():
    """Discover available API keys from environment, .env, or secrets.toml."""
    # Check OpenAI
    if os.environ.get("OPENAI_API_KEY"):
        return "openai", os.environ.get("OPENAI_API_KEY")
    # Check NVIDIA
    if os.environ.get("NVIDIA_API_KEY"):
        return "nvidia", os.environ.get("NVIDIA_API_KEY")
    # Check Groq
    if os.environ.get("GROQ_API_KEY"):
        return "groq", os.environ.get("GROQ_API_KEY")

    # Check local .env file
    project_root = Path(__file__).resolve().parent.parent
    env_file = project_root / ".env"
    if env_file.exists():
        try:
            # pyrefly: ignore [missing-import]
            from dotenv import load_dotenv
            load_dotenv(dotenv_path=env_file)
            if os.environ.get("OPENAI_API_KEY"):
                return "openai", os.environ.get("OPENAI_API_KEY")
            if os.environ.get("NVIDIA_API_KEY"):
                return "nvidia", os.environ.get("NVIDIA_API_KEY")
            if os.environ.get("GROQ_API_KEY"):
                return "groq", os.environ.get("GROQ_API_KEY")
        except ImportError:
            pass

    return None, None

def get_llm(provider: str, key: str):
    """Instantiate the corresponding LangChain chat model."""
    if provider == "openai":
        # pyrefly: ignore [missing-import]
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o-mini", temperature=0.3, api_key=key, timeout=30), "OpenAI gpt-4o-mini"
    elif provider == "nvidia":
        # pyrefly: ignore [missing-import]
        from langchain_nvidia_ai_endpoints import ChatNVIDIA
        model_name = os.getenv("NVIDIA_MODEL", "meta/llama-3.2-11b-vision-instruct").strip()
        return ChatNVIDIA(model=model_name, temperature=0.3, nvidia_api_key=key, timeout=30), f"NVIDIA NIM ({model_name})"
    elif provider == "groq":
        # pyrefly: ignore [missing-import]
        from langchain_groq import ChatGroq
        return ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3, groq_api_key=key, timeout=30), "Groq llama-3.3-70b-versatile"
    return None, "None"

def normalize_category(raw_output: str) -> str:
    cleaned = raw_output.strip().lower().replace(".", "").replace(",", "")
    for cat in ALLOWED_CATEGORIES:
        if cat in cleaned:
            return cat
    return "billing"

def run_evaluation():
    provider, key = find_api_key()
    if not provider or not key:
        print("=" * 75)
        print("⚠️ No live API key found in environment (.env or secrets.toml).")
        print("Please configure OPENAI_API_KEY, NVIDIA_API_KEY, or GROQ_API_KEY.")
        print("=" * 75)
        sys.exit(1)

    llm, model_desc = get_llm(provider, key)

    csv_path = Path(__file__).parent / "test_cases.csv"
    if not csv_path.exists():
        print(f"ERROR: Test file not found at {csv_path}")
        sys.exit(1)

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
    errors = 0
    total_time = 0.0

    print("=" * 75)
    print(f"Running Activity 18.1 Classification Evaluation on {model_desc}")
    print("=" * 75)

    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            cid = row.get("id", str(total))
            complaint = row["complaint"]
            expected = row.get("expected_category", row.get("category", "")).strip().lower()

            t0 = time.time()
            try:
                raw_pred = chain.invoke({"text": complaint})
                predicted = normalize_category(raw_pred)
                lat = time.time() - t0
                total_time += lat
                
                is_correct = (predicted == expected)
                if is_correct:
                    correct += 1
                    status = "✅ PASS"
                else:
                    incorrect += 1
                    status = "❌ FAIL"
                print(f"[{cid:>2}] [{status}] Expected: {expected:<10} | Predicted: {predicted:<10} | Latency: {lat:.2f}s | Complaint: {complaint[:30]}...", flush=True)
            except Exception as e:
                errors += 1
                lat = time.time() - t0
                total_time += lat
                status = "⚠️ ERROR"
                print(f"[{cid:>2}] [{status}] Expected: {expected:<10} | API Error: {str(e)[:35]}...", flush=True)

    evaluated_count = correct + incorrect
    accuracy = (correct / evaluated_count) * 100 if evaluated_count > 0 else 0.0
    avg_latency = (total_time / total) if total > 0 else 0.0

    print("=" * 75)
    print(f"📊 Evaluation Summary:")
    print(f"  Provider/Model:                         {model_desc}")
    print(f"  Total test cases:                       {total}")
    print(f"  Successfully evaluated:                 {evaluated_count}")
    print(f"  Correct classifications:                {correct}")
    print(f"  Incorrect classifications:              {incorrect}")
    print(f"  API / Provider errors:                  {errors}")
    print(f"  Classification Accuracy:                {accuracy:.2f}%")
    print(f"  Average Turn Latency:                   {avg_latency:.2f}s")
    print("=" * 75)

if __name__ == "__main__":
    run_evaluation()
