"""Retrieve joined game, box-score, recap, and availability evidence for Q&A.

Questions are loaded from part1/questions.json at runtime. Their return blocks
control answer fields and evidence keys, so the same retrieval and formatting
path works for public and hidden questions.
"""
from __future__ import annotations

import calendar
import json
import os
import re
from datetime import date

import sqlalchemy as sa
from sqlalchemy import bindparam, text
from sqlalchemy.engine import Connection

from backend.config import DB_DSN, EMBED_MODEL, LLM_MODEL
from backend.utils import ollama_embed, ollama_generate

BASE_DIR = os.path.dirname(__file__)
QUESTIONS_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "part1", "questions.json"))
ANSWERS_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "part1", "answers.json"))

TOP_K = 8
QUERY_PREFIX = "search_query: "
EVIDENCE_TABLES = {"game_details", "player_box_scores", "game_recaps", "injury_notes"}
MONTHS = {name.casefold(): number for number, name in enumerate(calendar.month_name) if name}
MONTHS.update({name.casefold(): number for number, name in enumerate(calendar.month_abbr) if name})


def retrieve(cx: Connection, qvec: list[float], k: int = TOP_K) -> list[dict]:
    """Exact, stable cosine search over every embedded source table."""
    queries = {
        "game_details": (
            "SELECT 'game_details' AS source_table, game_id AS id, game_id, "
            "NULL::bigint AS person_id, doc, 1 - (embedding <=> (:q)::vector) AS score "
            "FROM game_details WHERE embedding IS NOT NULL "
            "ORDER BY embedding <=> (:q)::vector, game_id LIMIT :k"
        ),
        "player_box_scores": (
            "SELECT 'player_box_scores' AS source_table, person_id AS id, game_id, person_id, doc, "
            "1 - (embedding <=> (:q)::vector) AS score FROM player_box_scores "
            "WHERE embedding IS NOT NULL ORDER BY embedding <=> (:q)::vector, game_id, person_id LIMIT :k"
        ),
        "game_recaps": (
            "SELECT 'game_recaps' AS source_table, recap_id AS id, game_id, "
            "NULL::bigint AS person_id, doc, 1 - (embedding <=> (:q)::vector) AS score "
            "FROM game_recaps WHERE embedding IS NOT NULL "
            "ORDER BY embedding <=> (:q)::vector, recap_id LIMIT :k"
        ),
        "injury_notes": (
            "SELECT 'injury_notes' AS source_table, note_id AS id, game_id, "
            "NULL::bigint AS person_id, doc, 1 - (embedding <=> (:q)::vector) AS score "
            "FROM injury_notes WHERE embedding IS NOT NULL "
            "ORDER BY embedding <=> (:q)::vector, note_id LIMIT :k"
        ),
    }
    found = []
    for query in queries.values():
        found.extend(dict(row) for row in cx.execute(text(query), {"q": qvec, "k": k}).mappings())
    return sorted(found, key=lambda row: (-float(row["score"]), row["source_table"], int(row["id"])))


def load_teams(cx: Connection) -> list[dict]:
    """Load canonical team names and ids used for entity matching/formatting."""
    return [dict(row) for row in cx.execute(text(
        "SELECT team_id, city, name, abbreviation FROM teams ORDER BY team_id"
    )).mappings()]


def load_players(cx: Connection) -> list[dict]:
    """Load canonical player names and ids used for entity matching/formatting."""
    return [dict(row) for row in cx.execute(text(
        "SELECT player_id, first_name, last_name FROM players ORDER BY player_id"
    )).mappings()]


def _contains(text_value: str, phrase: str, *, insensitive: bool = True) -> bool:
    if not phrase:
        return False
    flags = re.IGNORECASE if insensitive else 0
    return re.search(rf"(?<!\w){re.escape(str(phrase).strip())}(?!\w)", text_value, flags) is not None


def resolve_mentions(question: str, teams: list[dict], players: list[dict]) -> tuple[set[int], set[int]]:
    """Resolve names from the ingested catalogs, without question-id rules."""
    team_ids: set[int] = set()
    explicit_team_ids: set[int] = set()
    city_matches: set[int] = set()
    for team in teams:
        canonical_aliases = [f"{team['city']} {team['name']}", team["name"]]
        if any(_contains(question, alias) for alias in canonical_aliases):
            explicit_team_ids.add(int(team["team_id"]))
        if _contains(question, team["city"]):
            city_matches.add(int(team["team_id"]))
        # Abbreviations are useful when explicitly typed as uppercase codes.
        abbreviation = str(team.get("abbreviation", ""))
        if abbreviation and _contains(question, abbreviation, insensitive=False):
            explicit_team_ids.add(int(team["team_id"]))
    team_ids.update(explicit_team_ids or city_matches)

    surnames: dict[str, list[dict]] = {}
    for player in players:
        surnames.setdefault(str(player["last_name"]).casefold(), []).append(player)
    player_ids: set[int] = set()
    for player in players:
        full_name = f"{player['first_name']} {player['last_name']}"
        if _contains(question, full_name):
            player_ids.add(int(player["player_id"]))
            continue
        surname = str(player["last_name"])
        # A unique surname in the roster is a safe match for recap-style wording.
        if len(surnames[surname.casefold()]) == 1 and len(surname) >= 4 and _contains(question, surname):
            player_ids.add(int(player["player_id"]))
    return team_ids, player_ids


def date_window(question: str) -> tuple[date | None, date | None]:
    """Extract explicit calendar windows and the stated 2026 post-break rule."""
    q = question.casefold()
    if re.search(r"after\s+the\s+(?:all[- ]star\s+)?break", q):
        return date(2026, 2, 19), None

    month_pattern = "|".join(sorted(MONTHS, key=len, reverse=True))
    full_date = re.search(rf"\b({month_pattern})\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,?\s+(\d{{4}}))?\b", q)
    if full_date:
        month = MONTHS[full_date.group(1)]
        day = int(full_date.group(2))
        year = int(full_date.group(3)) if full_date.group(3) else (2025 if month >= 10 else 2026)
        try:
            exact = date(year, month, day)
            return exact, exact
        except ValueError:
            return None, None

    month_year = re.search(rf"\b({month_pattern})\s+(\d{{4}})\b", q)
    if month_year:
        month = MONTHS[month_year.group(1)]
        year = int(month_year.group(2))
        return date(year, month, 1), date(year, month, calendar.monthrange(year, month)[1])

    month_only = re.search(rf"\b(?:in|during|for|throughout|month of)\s+({month_pattern})\b", q)
    if month_only:
        month = MONTHS[month_only.group(1)]
        year = 2025 if month >= 10 else 2026
        return date(year, month, 1), date(year, month, calendar.monthrange(year, month)[1])

    iso = re.search(r"\b(202\d)-(\d{2})-(\d{2})\b", q)
    if iso:
        try:
            exact = date(*(int(part) for part in iso.groups()))
            return exact, exact
        except ValueError:
            return None, None
    return None, None


def _expand_game_ids(
    cx: Connection,
    question: str,
    team_ids: set[int],
    player_ids: set[int],
    retrieved: list[dict],
) -> list[int]:
    """Expand semantic hits to the named teams/players' full relevant game set."""
    clauses = []
    params: dict = {}
    if team_ids:
        ids = sorted(team_ids)
        if len(ids) == 2:
            clauses.append("g.home_team_id IN :team_ids AND g.away_team_id IN :team_ids")
        else:
            clauses.append("(g.home_team_id IN :team_ids OR g.away_team_id IN :team_ids)")
        params["team_ids"] = ids
    first_game = re.search(r"\b(?:season opener|opening game|first game)\b", question, re.I) is not None
    if player_ids and not team_ids:
        clauses.append(
            "(EXISTS (SELECT 1 FROM player_box_scores b "
            "WHERE b.game_id = g.game_id AND b.person_id IN :player_ids) "
            "OR EXISTS (SELECT 1 FROM injury_notes n "
            "WHERE n.game_id = g.game_id AND n.player_id IN :player_ids))"
        )
        params["player_ids"] = sorted(player_ids)

    start, end = date_window(question)
    if start:
        clauses.append("g.game_timestamp::date >= :start_date")
        params["start_date"] = start
    if end:
        clauses.append("g.game_timestamp::date <= :end_date")
        params["end_date"] = end

    if clauses:
        stmt = text(
            "SELECT g.game_id, g.game_timestamp FROM game_details g WHERE "
            + " AND ".join(clauses)
            + " ORDER BY g.game_timestamp, g.game_id"
        )
        expanding = []
        if team_ids:
            expanding.append(bindparam("team_ids", expanding=True))
        if player_ids and not team_ids:
            expanding.append(bindparam("player_ids", expanding=True))
        if expanding:
            stmt = stmt.bindparams(*expanding)
        rows = [dict(row) for row in cx.execute(stmt, params).mappings()]
        game_ids = [int(row["game_id"]) for row in rows]
        if re.search(r"\b(?:season opener|opening game|first game)\b", question, re.I) and game_ids:
            return game_ids[:1]
        return game_ids

    # With no catalog entity, keep the semantic candidates and apply any date window.
    game_ids = sorted({int(row["game_id"]) for row in retrieved})
    if not game_ids:
        return game_ids
    date_clauses = []
    params = {"ids": game_ids}
    if start:
        date_clauses.append("game_timestamp::date >= :start_date")
        params["start_date"] = start
    if end:
        date_clauses.append("game_timestamp::date <= :end_date")
        params["end_date"] = end
    if not date_clauses:
        return game_ids
    stmt = text(
        "SELECT game_id FROM game_details WHERE game_id IN :ids AND "
        + " AND ".join(date_clauses)
        + " ORDER BY game_timestamp, game_id"
    ).bindparams(bindparam("ids", expanding=True))
    rows = cx.execute(stmt, params).all()
    matched = [int(row[0]) for row in rows]
    return matched[:1] if first_game and matched else matched


def _fetch_by_games(cx: Connection, sql: str, game_ids: list[int]) -> list[dict]:
    if not game_ids:
        return []
    stmt = text(sql).bindparams(bindparam("ids", expanding=True))
    return [dict(row) for row in cx.execute(stmt, {"ids": game_ids}).mappings()]


def _fetch_games(cx: Connection, game_ids: list[int]) -> list[dict]:
    rows = _fetch_by_games(cx, (
        "SELECT g.game_id, g.game_timestamp, g.home_team_id, g.away_team_id, "
        "g.home_points, g.away_points, g.winning_team_id, g.doc, "
        "home.city AS home_city, home.name AS home_name, "
        "away.city AS away_city, away.name AS away_name "
        "FROM game_details g JOIN teams home ON home.team_id = g.home_team_id "
        "JOIN teams away ON away.team_id = g.away_team_id "
        "WHERE g.game_id IN :ids ORDER BY g.game_timestamp, g.game_id"
    ), game_ids)
    for row in rows:
        if not row.get("doc"):
            date_text = row["game_timestamp"].strftime("%Y-%m-%d")
            row["doc"] = (
                f"On {date_text}, {row['home_city']} {row['home_name']} played "
                f"{row['away_city']} {row['away_name']}. Final score: "
                f"{row['home_points']}-{row['away_points']}. (game_id={row['game_id']})"
            )
    return rows


def _fetch_box_scores(cx: Connection, game_ids: list[int]) -> list[dict]:
    return _fetch_by_games(cx, (
        "SELECT b.game_id, b.person_id, b.team_id, b.starter, b.points, "
        "b.offensive_reb, b.defensive_reb, b.assists, b.steals, b.blocks, b.turnovers, b.doc, "
        "p.first_name || ' ' || p.last_name AS player_name, own.name AS team_name, "
        "own.city AS team_city, opp.name AS opponent_name, opp.city AS opponent_city "
        "FROM player_box_scores b JOIN players p ON p.player_id = b.person_id "
        "JOIN game_details g ON g.game_id = b.game_id "
        "LEFT JOIN teams own ON own.team_id = b.team_id "
        "LEFT JOIN teams opp ON opp.team_id = CASE WHEN b.team_id = g.home_team_id "
        "THEN g.away_team_id ELSE g.home_team_id END "
        "WHERE b.game_id IN :ids ORDER BY b.game_id, b.person_id"
    ), game_ids)


def _fetch_recaps(cx: Connection, game_ids: list[int]) -> list[dict]:
    return _fetch_by_games(cx, (
        "SELECT recap_id, game_id, doc FROM game_recaps "
        "WHERE game_id IN :ids ORDER BY recap_id"
    ), game_ids)


def _fetch_notes(cx: Connection, game_ids: list[int]) -> list[dict]:
    return _fetch_by_games(cx, (
        "SELECT note_id, game_id, player_id, status, doc FROM injury_notes "
        "WHERE game_id IN :ids ORDER BY note_id"
    ), game_ids)


def _box_rows_for_question(
    rows: list[dict],
    question: str,
    team_ids: set[int],
    player_ids: set[int],
    games: list[dict],
) -> list[dict]:
    """Trim joined box scores using explicit entities and ordinary query filters."""
    q = question.casefold()
    game_by_id = {int(game["game_id"]): game for game in games}
    selected = rows
    if player_ids:
        selected = [row for row in selected if int(row["person_id"]) in player_ids]
    elif team_ids:
        against = re.search(r"\b(?:against|versus|vs\.?)\b", q) is not None
        if against and len(team_ids) == 1:
            selected = [row for row in selected if int(row["team_id"]) not in team_ids]
        elif len(game_by_id) != 1:
            selected = [row for row in selected if int(row["team_id"]) in team_ids]

    if re.search(r"\b(starter|starters)\b", q):
        selected = [row for row in selected if bool(row["starter"])]

    loss_margin = re.search(
        r"\b(?:lost|loss|defeat|defeated)\b.{0,32}?\bby\s+"
        r"(?:(at least|no less than|more than|over)\s+)?(\d+)\s*"
        r"(?:or\s+more|or\s+greater|or\s+higher|plus|\+)?",
        q,
    )
    if loss_margin and team_ids:
        minimum = int(loss_margin.group(2))
        if loss_margin.group(1) in {"more than", "over"}:
            minimum += 1
        selected = [
            row for row in selected
            if int(row["team_id"]) in team_ids
            and int(row["team_id"]) != int(game_by_id[int(row["game_id"])]["winning_team_id"])
            and abs(int(game_by_id[int(row["game_id"])]["home_points"])
                    - int(game_by_id[int(row["game_id"])]["away_points"])) >= minimum
        ]
    else:
        point_threshold = re.search(
            r"\b(?:at least|no fewer than|minimum of)\s+(\d+)\s+points\b|"
            r"\b(\d+)\s*\+\s*points?\b|\b(\d+)\s+or\s+more\s+points\b",
            q,
        )
        if point_threshold:
            minimum = int(next(value for value in point_threshold.groups() if value is not None))
            selected = [row for row in selected if int(row["points"]) >= minimum]
    return selected


def _prepare_sources(
    cx: Connection,
    question: str,
    return_spec: dict,
    teams: list[dict],
    players: list[dict],
) -> tuple[dict[str, list[dict]], list[dict]]:
    qvec = ollama_embed(EMBED_MODEL, QUERY_PREFIX + question)
    retrieved = retrieve(cx, qvec)
    team_ids, player_ids = resolve_mentions(question, teams, players)
    game_ids = _expand_game_ids(cx, question, team_ids, player_ids, retrieved)
    games = _fetch_games(cx, game_ids)
    declared = return_spec.get("evidence", [])
    requested_tables = {shape.get("table") for shape in declared if shape.get("table") in EVIDENCE_TABLES}

    sources: dict[str, list[dict]] = {table: [] for table in EVIDENCE_TABLES}
    sources["game_details"] = games
    if "player_box_scores" in requested_tables:
        all_box_rows = _fetch_box_scores(cx, game_ids)
        sources["player_box_scores"] = _box_rows_for_question(
            all_box_rows, question, team_ids, player_ids, games
        )
    if "game_recaps" in requested_tables:
        sources["game_recaps"] = _fetch_recaps(cx, game_ids)
    if "injury_notes" in requested_tables:
        notes = _fetch_notes(cx, game_ids)
        if player_ids:
            notes = [row for row in notes if int(row["player_id"]) in player_ids]
        sources["injury_notes"] = notes
    return sources, retrieved


def _relevant_excerpt(doc: str, question: str, limit: int = 2) -> str:
    """Keep the question-relevant sentences from a retrieved narrative row."""
    sentences = re.split(r"(?<=[.!?])\s+", doc)
    stop = {
        "what", "which", "who", "when", "where", "why", "how", "did", "does", "was", "were",
        "the", "and", "for", "with", "from", "this", "that", "their", "them", "they", "against",
        "game", "games", "season", "team", "points", "player", "players", "have", "had", "does",
    }
    q_words = {word.casefold() for word in re.findall(r"[\w'-]+", question) if len(word) > 2} - stop
    ranked = []
    for index, sentence in enumerate(sentences):
        words = {word.casefold() for word in re.findall(r"[\w'-]+", sentence) if len(word) > 2}
        overlap = len(q_words & words)
        if overlap:
            ranked.append((overlap, -index, sentence.strip()))
    if not ranked:
        return doc
    ranked.sort(reverse=True)
    chosen = [item[2] for item in ranked[:limit]]
    return " ".join(chosen)


def build_context(rows, question: str | None = None) -> str:
    """Render retrieved or joined source rows as keyed, citable context."""
    if isinstance(rows, dict):
        sources = rows
        lines = []
        for row in sources.get("game_details", []):
            lines.append(f"[game_details id={int(row['game_id'])}] {row['doc']}")
        for row in sources.get("player_box_scores", []):
            lines.append(
                f"[player_box_scores game_id={int(row['game_id'])} person_id={int(row['person_id'])}] "
                f"{row['doc']}"
            )
        for row in sources.get("game_recaps", []):
            doc = _relevant_excerpt(row["doc"], question) if question else row["doc"]
            lines.append(f"[game_recaps id={int(row['recap_id'])} game_id={int(row['game_id'])}] {doc}")
        for row in sources.get("injury_notes", []):
            doc = _relevant_excerpt(row["doc"], question) if question else row["doc"]
            lines.append(f"[injury_notes id={int(row['note_id'])} game_id={int(row['game_id'])}] {doc}")
        return "\n".join(lines) if lines else "[No matching source rows found.]"
    # Retain the starter helper's simple list input for callers that only have search hits.
    return "\n".join(
        f"[{row['source_table']} id={row['id']}] {row['doc']}" for row in rows
    )


def coerce_type(value, type_name: str):
    """Convert one answer value using the type in a question's return block."""
    if type_name == "bool":
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().casefold()
            if normalized in {"true", "yes", "1"}:
                return True
            if normalized in {"false", "no", "0"}:
                return False
            raise ValueError(f"Not a boolean value: {value}")
        return bool(value)
    if type_name == "int":
        return int(round(float(value)))
    if type_name == "str":
        return str(value)
    raise ValueError(f"Unknown type in return block: {type_name}")


def extract_json(raw: str) -> dict | None:
    """Extract the first JSON object from a model response."""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def abstain_result(return_spec: dict) -> dict:
    """Create false plus zero/empty defaults from the runtime return block."""
    defaults = {"bool": False, "int": 0, "str": ""}
    result = {
        key: defaults[type_name]
        for key, type_name in return_spec.items()
        if key != "evidence"
    }
    result["answerable"] = False
    result["evidence"] = []
    return result


def _identity(table: str, item: dict) -> tuple | None:
    """Map either public (`id`) or descriptive primary-key names to a row key."""
    try:
        if table == "game_details":
            return table, int(item.get("id", item.get("game_id")))
        if table == "player_box_scores":
            return table, int(item["game_id"]), int(item["person_id"])
        if table == "game_recaps":
            return table, int(item.get("id", item.get("recap_id")))
        if table == "injury_notes":
            return table, int(item.get("id", item.get("note_id")))
    except (KeyError, TypeError, ValueError):
        return None
    return None


def _available_identities(sources: dict[str, list[dict]]) -> set[tuple]:
    identities = set()
    for row in sources.get("game_details", []):
        identities.add(("game_details", int(row["game_id"])))
    for row in sources.get("player_box_scores", []):
        identities.add(("player_box_scores", int(row["game_id"]), int(row["person_id"])))
    for row in sources.get("game_recaps", []):
        identities.add(("game_recaps", int(row["recap_id"])))
    for row in sources.get("injury_notes", []):
        identities.add(("injury_notes", int(row["note_id"])))
    return identities


def _evidence_shape_item(table: str, row: dict, shape: dict) -> dict:
    item = {"table": table}
    for key in shape:
        if key == "table":
            continue
        if table == "game_details" and key in {"id", "game_id"}:
            value = row["game_id"]
        elif table == "player_box_scores" and key in {"game_id", "person_id"}:
            value = row[key]
        elif table == "game_recaps" and key in {"id", "recap_id"}:
            value = row["recap_id"]
        elif table == "injury_notes" and key in {"id", "note_id"}:
            value = row["note_id"]
        else:
            continue
        item[key] = coerce_type(value, shape[key])
    return item


def ground_evidence(parsed_evidence, declared_shapes: list[dict], sources: dict[str, list[dict]]) -> list[dict]:
    """Keep only declared citations that match rows actually present in context."""
    if not isinstance(parsed_evidence, list):
        return []
    available = _available_identities(sources)
    grounded = []
    seen = set()
    shape_by_table = {shape.get("table"): shape for shape in declared_shapes}
    for entry in parsed_evidence:
        if not isinstance(entry, dict) or entry.get("table") not in shape_by_table:
            continue
        shape = shape_by_table[entry["table"]]
        try:
            item = {"table": entry["table"]}
            for key, type_name in shape.items():
                if key == "table":
                    continue
                item[key] = coerce_type(entry[key], type_name)
        except (KeyError, TypeError, ValueError):
            continue
        identity = _identity(item["table"], item)
        if identity in available and identity not in seen:
            grounded.append(item)
            seen.add(identity)
    return grounded


def evidence_fully_grounded(declared_shapes: list[dict], grounded: list[dict]) -> bool:
    """Require at least one verified row from every evidence table requested."""
    required = {shape["table"] for shape in declared_shapes if shape.get("table")}
    present = {item["table"] for item in grounded}
    return bool(required) and required.issubset(present)


def normalize_team_names(result: dict, teams: list[dict]) -> dict:
    """Normalize team-valued answer fields to the exact mascot in teams.csv."""
    normalized = dict(result)
    skipped = {"evidence", "player_name", "reason", "status", "answer"}
    for key, value in result.items():
        if key in skipped or not isinstance(value, str):
            continue
        matches = set()
        for team in teams:
            aliases = [f"{team['city']} {team['name']}", team["name"], team["abbreviation"]]
            if any(_contains(value, alias) for alias in aliases if alias):
                matches.add(team["name"])
        if len(matches) == 1:
            normalized[key] = next(iter(matches))
    return normalized


def normalize_player_name(value: str, players: list[dict]) -> str:
    """Return the roster spelling when an answer contains one known player."""
    matches = []
    for player in players:
        full = f"{player['first_name']} {player['last_name']}"
        if _contains(value, full):
            matches.append(full)
    if len(matches) == 1:
        return matches[0]
    surname_matches = [
        player for player in players
        if len(str(player["last_name"])) >= 4 and _contains(value, str(player["last_name"]))
    ]
    if len(surname_matches) == 1:
        player = surname_matches[0]
        return f"{player['first_name']} {player['last_name']}"
    return value


def _complete_player_aggregate_evidence(
    question: str,
    result: dict,
    sources: dict[str, list[dict]],
    declared_shapes: list[dict],
    players: list[dict],
    teams: list[dict],
    team_ids: set[int],
    player_ids: set[int],
    grounded: list[dict],
) -> list[dict]:
    """Include all fetched player-game rows for a player selected by an aggregate."""
    if not re.search(r"\b(total|sum|most|highest|largest|maximum|top|greatest|how many)\b", question, re.I):
        return grounded
    box_shape = next((shape for shape in declared_shapes if shape.get("table") == "player_box_scores"), None)
    if box_shape is None:
        return grounded

    target_ids = set(player_ids)
    answer_name = result.get("player_name")
    if isinstance(answer_name, str) and answer_name:
        canonical = normalize_player_name(answer_name, players)
        for player in players:
            if f"{player['first_name']} {player['last_name']}" == canonical:
                target_ids = {int(player["player_id"])}
                break
    if not target_ids:
        return grounded

    by_table = {shape.get("table"): shape for shape in declared_shapes}
    details_shape = by_table.get("game_details")
    detail_rows = {int(row["game_id"]): row for row in sources.get("game_details", [])}
    grounded = list(grounded)
    present = {_identity(item["table"], item) for item in grounded}
    for row in sources.get("player_box_scores", []):
        if int(row["person_id"]) not in target_ids:
            continue
        item = _evidence_shape_item("player_box_scores", row, box_shape)
        identity = _identity("player_box_scores", item)
        if identity in _available_identities(sources) and identity not in present:
            grounded.append(item)
            present.add(identity)
        if details_shape:
            game = detail_rows.get(int(row["game_id"]))
            if game:
                detail_item = _evidence_shape_item("game_details", game, details_shape)
                detail_identity = _identity("game_details", detail_item)
                if detail_identity in _available_identities(sources) and detail_identity not in present:
                    grounded.append(detail_item)
                    present.add(detail_identity)
    return grounded


def _pair_text_evidence(grounded: list[dict], sources: dict[str, list[dict]], declared_shapes: list[dict]) -> list[dict]:
    """Pair each cited recap or note with its game row when that shape is allowed."""
    detail_shape = next((shape for shape in declared_shapes if shape.get("table") == "game_details"), None)
    if not detail_shape:
        return grounded
    game_for_text = {}
    game_for_text.update({("game_recaps", int(row["recap_id"])): int(row["game_id"])
                          for row in sources.get("game_recaps", [])})
    game_for_text.update({("injury_notes", int(row["note_id"])): int(row["game_id"])
                          for row in sources.get("injury_notes", [])})
    game_rows = {int(row["game_id"]): row for row in sources.get("game_details", [])}
    paired = list(grounded)
    present = {_identity(item["table"], item) for item in paired}
    for item in grounded:
        table = item["table"]
        if table not in {"game_recaps", "injury_notes"}:
            continue
        source_id = int(item.get("id", item.get("recap_id", item.get("note_id"))))
        game = game_rows.get(game_for_text.get((table, source_id)))
        if not game:
            continue
        game_item = _evidence_shape_item("game_details", game, detail_shape)
        identity = _identity("game_details", game_item)
        if identity not in present:
            paired.append(game_item)
            present.add(identity)
    return paired


def _ordered_team_ids(question: str, teams: list[dict], team_ids: set[int]) -> list[int]:
    """Order matched teams as they appear, for questions about one side of a game."""
    positions = []
    for team in teams:
        team_id = int(team["team_id"])
        if team_id not in team_ids:
            continue
        matches = []
        for alias in (f"{team['city']} {team['name']}", team["name"], team["city"], team.get("abbreviation", "")):
            if alias:
                found = re.search(rf"(?<!\w){re.escape(str(alias))}(?!\w)", question, re.I)
                if found:
                    matches.append(found.start())
        positions.append((min(matches) if matches else len(question), team_id))
    return [team_id for _, team_id in sorted(positions)]


def _make_derived_result(
    return_spec: dict,
    values: dict,
    evidence_rows: dict[str, list[dict]],
    sources: dict[str, list[dict]],
) -> dict:
    """Format a verified structured result and cite all rows used to derive it."""
    defaults = {"bool": False, "int": 0, "str": ""}
    result = {
        key: defaults[type_name]
        for key, type_name in return_spec.items()
        if key != "evidence"
    }
    result["answerable"] = True
    for key, value in values.items():
        if key in return_spec and key != "evidence":
            result[key] = coerce_type(value, return_spec[key])

    shapes = {shape.get("table"): shape for shape in return_spec.get("evidence", [])}
    citations = []
    present = set()
    for table in ("game_details", "player_box_scores", "game_recaps", "injury_notes"):
        shape = shapes.get(table)
        if not shape:
            continue
        for row in evidence_rows.get(table, []):
            item = _evidence_shape_item(table, row, shape)
            identity = _identity(table, item)
            if identity not in present:
                citations.append(item)
                present.add(identity)
    citations = _pair_text_evidence(citations, sources, return_spec.get("evidence", []))
    if not evidence_fully_grounded(return_spec.get("evidence", []), citations):
        return abstain_result(return_spec)
    result["evidence"] = citations
    return result


def _event_player(recap_doc: str, players: list[dict]) -> dict | None:
    """Find the roster player nearest the go-ahead/game-winning event in a recap."""
    for sentence in re.split(r"(?<=[.!?])\s+", recap_doc):
        cue = re.search(r"go[- ]ahead|game[- ]winning|deciding", sentence, re.I)
        if not cue:
            continue
        mentions = []
        for player in players:
            full_name = f"{player['first_name']} {player['last_name']}"
            found = re.search(rf"(?<!\w){re.escape(full_name)}(?!\w)", sentence, re.I)
            if found:
                mentions.append((found.start(), player))
        before = [item for item in mentions if item[0] < cue.start()]
        after = [item for item in mentions if item[0] >= cue.start()]
        if before:
            return max(before, key=lambda item: item[0])[1]
        if after:
            return min(after, key=lambda item: item[0])[1]
    return None


def derive_structured_answer(
    question: str,
    return_spec: dict,
    sources: dict[str, list[dict]],
    teams: list[dict],
    players: list[dict],
    team_ids: set[int],
    player_ids: set[int],
) -> dict | None:
    """Apply general table-backed lookups and aggregations before formatting.

    This uses the question's requested fields, recognized catalog entities, and
    ordinary aggregation language. It never depends on question ids or tiers.
    """
    q = question.casefold()
    fields = set(return_spec) - {"answerable", "evidence"}
    game_rows = sources.get("game_details", [])
    box_rows = sources.get("player_box_scores", [])
    recap_rows = sources.get("game_recaps", [])
    note_rows = sources.get("injury_notes", [])
    team_order = _ordered_team_ids(question, teams, team_ids)
    focus_team = team_order[0] if team_order else None
    games_by_id = {int(row["game_id"]): row for row in game_rows}
    teams_by_id = {int(team["team_id"]): team for team in teams}

    def successful(values: dict, used: dict[str, list[dict]]) -> dict:
        return _make_derived_result(return_spec, values, used, sources)

    def score_text(game: dict) -> str:
        home_points, away_points = int(game["home_points"]), int(game["away_points"])
        if int(game["winning_team_id"]) == int(game["home_team_id"]):
            return f"{home_points}-{away_points}"
        return f"{away_points}-{home_points}"

    def opponent_name(game: dict, target_team_id: int) -> str:
        return game["away_name"] if int(game["home_team_id"]) == target_team_id else game["home_name"]

    def team_game_rows(target_team_id: int) -> list[dict]:
        return [row for row in game_rows if target_team_id in {
            int(row["home_team_id"]), int(row["away_team_id"])
        }]

    def rows_for_player(person_id: int) -> list[dict]:
        return [row for row in box_rows if int(row["person_id"]) == person_id]

    def add_game_rows(rows: list[dict]) -> list[dict]:
        ids = {int(row["game_id"]) for row in rows}
        return [game for game in game_rows if int(game["game_id"]) in ids]

    # A named player's period total and appearance count are a direct box-score aggregate.
    if {"points", "games"}.issubset(fields) and player_ids and re.search(r"\bhow many points\b", q):
        player_rows = [row for row in box_rows if int(row["person_id"]) in player_ids]
        if player_rows:
            return successful(
                {"points": sum(int(row["points"]) for row in player_rows),
                 "games": len({int(row["game_id"]) for row in player_rows})},
                {"player_box_scores": player_rows, "game_details": add_game_rows(player_rows)},
            )

    # Rank players by their total points over the filtered game set.
    if {"player_name", "points"}.issubset(fields) and box_rows and re.search(
        r"\b(?:most|highest|greatest|top)\b.{0,40}\btotal points\b|\btotal points\b.{0,40}\b(?:most|highest|greatest)\b", q
    ):
        totals: dict[int, int] = {}
        for row in box_rows:
            person_id = int(row["person_id"])
            totals[person_id] = totals.get(person_id, 0) + int(row["points"])
        winner_id = min(totals, key=lambda person_id: (-totals[person_id], person_id))
        winner_rows = rows_for_player(winner_id)
        return successful(
            {"player_name": winner_rows[0]["player_name"], "points": totals[winner_id]},
            {"player_box_scores": winner_rows, "game_details": add_game_rows(winner_rows)},
        )

    # Count a player's qualifying appearances, after generic row filters were applied.
    if {"player_name", "games"}.issubset(fields) and box_rows and re.search(
        r"\bmost\b.{0,35}\bgames?\b|\bmost\b.{0,35}\bappearances?\b", q
    ):
        appearances: dict[int, set[int]] = {}
        for row in box_rows:
            appearances.setdefault(int(row["person_id"]), set()).add(int(row["game_id"]))
        winner_id = min(appearances, key=lambda person_id: (-len(appearances[person_id]), person_id))
        winner_rows = [row for row in box_rows if int(row["person_id"]) == winner_id]
        return successful(
            {"player_name": winner_rows[0]["player_name"], "games": len(appearances[winner_id])},
            {"player_box_scores": winner_rows, "game_details": add_game_rows(winner_rows)},
        )

    # A team scoring leader is the maximum points row for the named side in a game.
    if {"player_name", "points"}.issubset(fields) and box_rows and focus_team and re.search(
        r"\b(?:led|leader|leading|top scorer|most points)\b", q
    ):
        team_rows = [row for row in box_rows if int(row["team_id"]) == focus_team]
        if team_rows:
            best = min(team_rows, key=lambda row: (-int(row["points"]), int(row["person_id"])))
            return successful(
                {"player_name": best["player_name"], "points": int(best["points"])},
                {"player_box_scores": [best], "game_details": add_game_rows([best])},
            )

    # A named-player single-game box-score lookup.
    if "points" in fields and player_ids and box_rows and len(player_ids) == 1:
        player_rows = [row for row in box_rows if int(row["person_id"]) in player_ids]
        if len(player_rows) == 1:
            row = player_rows[0]
            return successful(
                {"points": int(row["points"])},
                {"player_box_scores": [row], "game_details": add_game_rows([row])},
            )

    # A named matchup's winner and winner-first score come only from game_details.
    if {"winner", "score"}.issubset(fields) and len(game_rows) == 1:
        game = game_rows[0]
        winner = teams_by_id.get(int(game["winning_team_id"]))
        if winner:
            return successful(
                {"winner": winner["name"], "score": score_text(game)},
                {"game_details": [game]},
            )

    # Largest loss margin is an exact maximum over the selected team's schedule.
    if "margin" in fields and focus_team and re.search(r"\b(?:largest|biggest|greatest|maximum)\b", q):
        losses = []
        for game in team_game_rows(focus_team):
            if int(game["winning_team_id"]) == focus_team:
                continue
            team_points = int(game["home_points"] if int(game["home_team_id"]) == focus_team else game["away_points"])
            opponent_points = int(game["away_points"] if int(game["home_team_id"]) == focus_team else game["home_points"])
            losses.append((opponent_points - team_points, int(game["game_id"]), game))
        if losses:
            margin, _, game = min(losses, key=lambda item: (-item[0], item[1]))
            return successful(
                {"opponent": opponent_name(game, focus_team), "margin": margin, "score": score_text(game)},
                {"game_details": [game]},
            )

    # Largest reported comeback deficit uses recap prose, while the game result and score remain table-backed.
    if "deficit" in fields and focus_team and recap_rows and re.search(r"\bdeficit\b", q):
        candidates = []
        patterns = (
            r"\b(\d+)\s*[- ]point\s+deficit\b",
            r"\bdeficit\s+of\s+(\d+)\s+points?\b",
            r"\bdown\s+by\s+(\d+)\s+points?\b",
        )
        for recap in recap_rows:
            game = games_by_id.get(int(recap["game_id"]))
            if not game or int(game["winning_team_id"]) != focus_team:
                continue
            values = [int(match.group(1)) for pattern in patterns for match in re.finditer(pattern, recap["doc"], re.I)]
            if values:
                candidates.append((max(values), int(game["game_id"]), game, recap))
        if candidates:
            deficit, _, game, recap = min(candidates, key=lambda item: (-item[0], item[1]))
            return successful(
                {"opponent": opponent_name(game, focus_team), "deficit": deficit, "score": score_text(game)},
                {"game_details": [game], "game_recaps": [recap]},
            )

    # Resolve a go-ahead or game-winning scorer from the closest roster name in its recap sentence.
    if "player_name" in fields and recap_rows and re.search(r"go[- ]ahead|game[- ]winning|deciding", q):
        for recap in recap_rows:
            player = _event_player(recap["doc"], players)
            if not player:
                continue
            person_id = int(player["player_id"])
            player_row = next((row for row in box_rows if int(row["game_id"]) == int(recap["game_id"])
                               and int(row["person_id"]) == person_id), None)
            game = games_by_id.get(int(recap["game_id"]))
            if player_row and game:
                return successful(
                    {"player_name": f"{player['first_name']} {player['last_name']}"},
                    {"game_details": [game], "game_recaps": [recap], "player_box_scores": [player_row]},
                )
        return abstain_result(return_spec)

    return None


def _answer_prompt(question: str, return_spec: dict, context: str) -> str:
    return (
        "Answer the question using only these joined source rows. The database tables are authoritative "
        "for scores, results, dates, player stats, and availability status; recap/note prose supplies "
        "narrative detail and must not override a conflicting table value.\n"
        f"Return exactly one JSON object with this question-specific shape: {json.dumps(return_spec)}\n"
        "Use only the listed field names and types. If the data has no supporting row, answerable must be "
        "false, all other values must be their empty/zero defaults, and evidence must be []. Never guess. "
        "Cite only rows shown below, using exactly the key names in the evidence shapes. Cite every row "
        "that contributes to an aggregate, and cite a recap or injury note together with its game. "
        "Use team mascots and exact roster player names. Format scores as winner-points then loser-points, "
        "for example 117-102.\n\n"
        f"Sources:\n{context}\n\nQuestion: {question}\nJSON:"
    )


def answer_question(cx: Connection, question: dict, teams: list[dict] | None = None, players: list[dict] | None = None) -> dict:
    """Run one shared retrieve, join, reason, format, and citation-check flow."""
    return_spec = question["return"]
    teams = teams if teams is not None else load_teams(cx)
    players = players if players is not None else load_players(cx)
    sources, _ = _prepare_sources(cx, question["question"], return_spec, teams, players)
    return _answer_from_sources(question, sources, teams, players)


def _answer_from_sources(
    question: dict,
    sources: dict[str, list[dict]],
    teams: list[dict],
    players: list[dict],
    *,
    require_all_evidence: bool = True,
) -> dict:
    """Shared reasoning and citation validation for batch answers and live chat."""
    return_spec = question["return"]
    context = build_context(sources, question["question"])
    if context == "[No matching source rows found.]":
        return abstain_result(return_spec)

    declared_shapes = return_spec.get("evidence", [])
    # A free-form chat fallback can cite any retrieved source; it need not use
    # every corpus. Typed questions retain their requested evidence requirements.
    required_shapes = declared_shapes if require_all_evidence else [
        shape for shape in declared_shapes if shape.get("table") == "game_details"
    ]
    if any(not sources.get(shape.get("table"), []) for shape in required_shapes):
        return abstain_result(return_spec)
    team_ids, player_ids = resolve_mentions(question["question"], teams, players)
    derived = derive_structured_answer(
        question["question"], return_spec, sources, teams, players, team_ids, player_ids
    )

    raw = ollama_generate(LLM_MODEL, _answer_prompt(question["question"], return_spec, context))
    parsed = extract_json(raw)
    if parsed is None:
        return derived if derived is not None else abstain_result(return_spec)

    result = {}
    for key, type_name in return_spec.items():
        if key == "evidence":
            continue
        if key not in parsed:
            return derived if derived is not None else abstain_result(return_spec)
        try:
            result[key] = coerce_type(parsed[key], type_name)
        except (TypeError, ValueError):
            return derived if derived is not None else abstain_result(return_spec)
    result = normalize_team_names(result, teams)
    if isinstance(result.get("player_name"), str):
        result["player_name"] = normalize_player_name(result["player_name"], players)

    if derived is not None:
        return derived

    # Availability state is a categorical table field, so retain its canonical value.
    if note_rows := sources.get("injury_notes", []):
        if "status" in result:
            result["status"] = str(note_rows[0]["status"])

    grounded = ground_evidence(parsed.get("evidence"), declared_shapes, sources)
    grounded = _complete_player_aggregate_evidence(
        question["question"], result, sources, declared_shapes, players, teams,
        team_ids, player_ids, grounded,
    )
    grounded = _pair_text_evidence(grounded, sources, declared_shapes)
    if not result.get("answerable", False) or not evidence_fully_grounded(required_shapes, grounded):
        return abstain_result(return_spec)
    result["evidence"] = grounded
    return result


def _chat_return_spec(question: str, teams: list[dict], players: list[dict]) -> dict:
    """Infer requested fields for chat, which has no caller-supplied return block.

    Only general intent words and catalog mentions are used. Batch questions
    continue to use their own runtime return blocks without inference.
    """
    q = question.casefold()
    _, player_ids = resolve_mentions(question, teams, players)
    fields: dict[str, str] = {"answer": "str"}
    tables = set(EVIDENCE_TABLES)
    scoring = re.search(r"\b(?:points?|scor(?:ed|ing|er|ers))\b", q)
    if player_ids and re.search(r"\bscor(?:e|es)\b", q):
        scoring = True
    player_lookup = player_ids or re.search(r"\b(?:who|player|players|starter|starters)\b", q)

    if re.search(r"\b(?:injur\w*|availab\w*|unavailab\w*|ruled out|sit out|sat out|miss\w*|absen\w*|status)\b", q):
        fields = {"status": "str", "reason": "str"}
        tables = {"game_details", "injury_notes"}
    elif re.search(r"go[- ]ahead|game[- ]winning|deciding", q):
        fields = {"player_name": "str"}
        tables = {"game_details", "game_recaps", "player_box_scores"}
    elif re.search(r"\b(?:deficit|comeback)\b", q):
        fields = {"opponent": "str", "deficit": "int", "score": "str"}
        tables = {"game_details", "game_recaps"}
    elif re.search(r"\b(?:largest|biggest|greatest|maximum|worst)\b", q) and re.search(
        r"\b(?:loss|defeat|margin|lost)\b", q
    ) and not (player_ids or re.search(r"\b(?:player|players|starter|starters)\b", q)):
        fields = {"opponent": "str", "margin": "int", "score": "str"}
        tables = {"game_details"}
    elif scoring and player_lookup:
        tables = {"game_details", "player_box_scores"}
        if re.search(r"\bmost\b.{0,35}\b(?:games?|appearances?)\b", q):
            fields = {"player_name": "str", "games": "int"}
        elif not player_ids:
            fields = {"player_name": "str", "points": "int"}
        else:
            fields = {"points": "int"}
            start, end = date_window(question)
            if (start and start != end) or re.search(r"\b(?:total|sum|season|games)\b", q):
                # An opener is a single-game lookup, even though it says season.
                if not re.search(r"\b(?:season opener|opening game|first game)\b", q):
                    fields["games"] = "int"
    elif re.search(r"\b(?:led|leading|leader|top scorer)\b", q) and scoring:
        fields = {"player_name": "str", "points": "int"}
        tables = {"game_details", "player_box_scores"}
    elif re.search(r"\b(?:score|winner|won|beat|victor)\b", q):
        fields = {"winner": "str", "score": "str"}
        tables = {"game_details"}

    shapes = [
        {"table": "game_details", "id": "int"},
        {"table": "player_box_scores", "game_id": "int", "person_id": "int"},
        {"table": "game_recaps", "id": "int"},
        {"table": "injury_notes", "id": "int"},
    ]
    return {
        "answerable": "bool", **fields,
        "evidence": [shape for shape in shapes if shape["table"] in tables],
    }


def _format_chat_answer(result: dict) -> str:
    """Render verified values without asking the model to recalculate them."""
    if not result.get("answerable"):
        return "The provided data does not contain enough evidence to answer this question."
    if "answer" in result:
        return result["answer"]
    if "player_name" in result:
        if "points" in result:
            return f"{result['player_name']} scored {result['points']} points."
        if "games" in result:
            return f"{result['player_name']} had {result['games']} qualifying games."
        return f"{result['player_name']}."
    if "points" in result:
        if "games" in result:
            return f"{result['points']} points across {result['games']} games."
        return f"{result['points']} points."
    if "winner" in result:
        return f"{result['winner']} won {result['score']}."
    return "\n".join(
        f"{key.replace('_', ' ').capitalize()}: {value}"
        for key, value in result.items() if key not in {"answerable", "evidence"}
    )


def answer_chat(cx: Connection, question: str) -> dict:
    """Use the batch reasoning path, then attach only the cited rows for the UI."""
    teams = load_teams(cx)
    players = load_players(cx)
    return_spec = _chat_return_spec(question, teams, players)
    sources, _ = _prepare_sources(cx, question, return_spec, teams, players)
    result = _answer_from_sources(
        {"question": question, "return": return_spec}, sources, teams, players,
        require_all_evidence="answer" not in return_spec,
    )
    docs = {
        _identity(table, row): row.get("doc", "")
        for table, rows in sources.items() for row in rows
    }
    evidence = [
        {**item, "doc": docs.get(_identity(item["table"], item), "")}
        for item in result["evidence"]
    ]
    return {"answer": _format_chat_answer(result), "evidence": evidence}


def main() -> None:
    """Read runtime questions and write one shaped result for each question."""
    print("Starting RAG Answering")
    with open(QUESTIONS_PATH, encoding="utf-8") as handle:
        questions = json.load(handle)
    engine = sa.create_engine(DB_DSN)
    with engine.begin() as cx:
        teams = load_teams(cx)
        players = load_players(cx)
        outputs = []
        for question in questions:
            result = answer_question(cx, question, teams, players)
            outputs.append({"id": question["id"], "result": result})
            print(f"Q{question['id']}: {json.dumps(result, ensure_ascii=False)}")

    with open(ANSWERS_PATH, "w", encoding="utf-8") as handle:
        json.dump(outputs, handle, ensure_ascii=False, indent=2)
    print(f"Finished RAG Answering: wrote {len(outputs)} answers to {ANSWERS_PATH}")


if __name__ == "__main__":
    main()
