import json
import logging

from document_processor import extract_full_text
from legal_analyzer.orchestrator import run_deep_analysis
from legal_analyzer.presentation import generate_persian_presentation

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    contract_text = extract_full_text("contract.pdf")
    if not contract_text:
        raise SystemExit("Could not extract text from contract.pdf")

    results = None
    for event in run_deep_analysis(contract_text):
        if event["event"] == "progress":
            logger.info(
                "Progress: step %d/%d - %s",
                event["step"],
                event["total"],
                event["title"],
            )
        elif event["event"] == "complete":
            results = event["results"]

    with open("analysis_result.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("Deep legal analysis completed.")

    generate_persian_presentation(
        analysis_json_path="analysis_result.json",
        output_path="presentation_fa.txt",
    )
    print("Persian legal presentation generated.")
