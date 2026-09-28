"""Validate part1/answers.json against part1/answers.schema.json.

Also checks that the set of ids in answers.json exactly matches the set of
ids in part1/questions.json, since the schema alone won't catch a missing or
duplicated question.

usage:
    python -m backend.validate
"""
import json
import os
import sys

import jsonschema

BASE_DIR = os.path.dirname(__file__)
QUESTIONS_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "part1", "questions.json"))
ANSWERS_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "part1", "answers.json"))
SCHEMA_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "part1", "answers.schema.json"))

MAX_ERRORS_SHOWN = 5


def validate_answers() -> list[str]:
    """Validate answers.json against the schema and against questions.json's id set.

    :return: list of human-readable error messages; empty if everything passed
    """
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        questions = json.load(f)
    with open(ANSWERS_PATH, encoding="utf-8") as f:
        answers = json.load(f)
    with open(SCHEMA_PATH, encoding="utf-8") as f:
        schema = json.load(f)

    errors = [
        f"{e.json_path}: {e.message}"
        for e in jsonschema.Draft202012Validator(schema).iter_errors(answers)
    ]

    question_ids = {q["id"] for q in questions}
    answer_ids = {a["id"] for a in answers} if isinstance(answers, list) else set()
    if answer_ids != question_ids:
        missing = question_ids - answer_ids
        extra = answer_ids - question_ids
        if missing:
            errors.append(f"answers.json is missing ids from questions.json: {sorted(missing)}")
        if extra:
            errors.append(f"answers.json has ids not in questions.json: {sorted(extra)}")

    return errors


def main() -> None:
    """Run validate_answers() and print PASS/FAIL, exiting 1 on failure."""
    errors = validate_answers()
    if not errors:
        print("PASS: part1/answers.json is schema-valid and its ids match part1/questions.json.")
        return
    print(f"FAIL: {len(errors)} error(s) found in part1/answers.json.")
    for msg in errors[:MAX_ERRORS_SHOWN]:
        print(f"  - {msg}")
    if len(errors) > MAX_ERRORS_SHOWN:
        print(f"  ... and {len(errors) - MAX_ERRORS_SHOWN} more")
    sys.exit(1)


if __name__ == "__main__":
    main()
