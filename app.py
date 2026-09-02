"""Minimal Flask app tying Part 1 (doc assistant) and Part 3 (lead scoring) together."""

from flask import Flask, render_template, request

import part1_doc_assistant
import part3_lead_scoring

app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", qa_result=None, qa_error=None, score_result=None, score_error=None)


@app.route("/ask", methods=["POST"])
def ask():
    question = request.form.get("question", "").strip()
    qa_result = None
    qa_error = None
    if not question:
        qa_error = "Please enter a question."
    else:
        try:
            qa_result = part1_doc_assistant.answer_question(question)
        except RuntimeError as exc:
            qa_error = str(exc)
    return render_template(
        "index.html",
        qa_result=qa_result,
        qa_error=qa_error,
        qa_question=question,
        score_result=None,
        score_error=None,
    )


@app.route("/score", methods=["POST"])
def score():
    score_error = None
    score_result = None
    lead_dict = {
        "source": request.form.get("source", ""),
        "city": request.form.get("city", ""),
        "area": request.form.get("area", ""),
        "property_type": request.form.get("property_type", ""),
        "budget_pkr_lac": request.form.get("budget_pkr_lac") or None,
        "bedrooms": request.form.get("bedrooms") or None,
        "first_response_minutes": request.form.get("first_response_minutes") or None,
        "calls_made": request.form.get("calls_made") or 0,
        "total_call_seconds": request.form.get("total_call_seconds") or 0,
        "whatsapp_replies": request.form.get("whatsapp_replies") or 0,
        "site_visits": request.form.get("site_visits") or 0,
        "agent_experience_years": request.form.get("agent_experience_years") or None,
        "is_overseas": request.form.get("is_overseas") == "on",
        "referred_by_existing_client": request.form.get("referred_by_existing_client") == "on",
        "has_financing_approved": request.form.get("has_financing_approved") == "on",
    }
    try:
        score_result = part3_lead_scoring.score_lead(lead_dict)
    except FileNotFoundError:
        score_error = "model.pkl not found. Run 'python part3_lead_scoring.py' first to train and save the model."
    except Exception as exc:
        score_error = f"Scoring failed: {exc}"

    return render_template(
        "index.html",
        qa_result=None,
        qa_error=None,
        score_result=score_result,
        score_error=score_error,
        lead_form=lead_dict,
    )


if __name__ == "__main__":
    app.run(debug=True)
