from llm_kg.methods.lightrag.parser import parse_extraction

TD = "<|#|>"
CD = "<|COMPLETE|>"


def test_parses_one_entity_and_one_relation() -> None:
    text = (
        f"entity{TD}Alice{TD}person{TD}A person.\n"
        f"entity{TD}Bob{TD}person{TD}Another person.\n"
        f"relation{TD}Alice{TD}Bob{TD}friendship, trust{TD}They know each other.\n"
        f"{CD}\n"
    )
    r = parse_extraction(text, TD, CD)
    assert len(r.entities) == 2
    assert r.entities[0].name == "Alice"
    assert r.entities[0].type == "person"
    assert r.entities[0].description == "A person."
    assert len(r.relations) == 1
    rel = r.relations[0]
    assert rel.src == "Alice"
    assert rel.dst == "Bob"
    assert rel.keywords == ["friendship", "trust"]
    assert rel.weight == 1.0


def test_handles_relationship_keyword_alias() -> None:
    """LightRAG accepts both `relation` and `relationship` as the record-type literal."""
    text = f"relationship{TD}A{TD}B{TD}kw{TD}d\n"
    r = parse_extraction(text, TD, CD)
    assert len(r.relations) == 1


def test_drops_self_loops() -> None:
    text = f"relation{TD}A{TD}A{TD}kw{TD}d\n"
    assert parse_extraction(text, TD, CD).relations == []


def test_normalizes_entity_type_to_lowercase() -> None:
    text = f"entity{TD}X{TD}PERSON{TD}d\n"
    r = parse_extraction(text, TD, CD)
    assert r.entities[0].type == "person"


def test_entity_type_with_comma_takes_first_token() -> None:
    text = f"entity{TD}X{TD}person, leader{TD}d\n"
    r = parse_extraction(text, TD, CD)
    assert r.entities[0].type == "person"


def test_skips_malformed_records() -> None:
    """Wrong field count → silently dropped."""
    text = (
        f"entity{TD}A{TD}person\n"  # only 3 fields, need 4
        f"entity{TD}B{TD}person{TD}good\n"
        f"relation{TD}A{TD}B{TD}kw\n"  # only 4 fields, need 5
        f"relation{TD}A{TD}B{TD}kw{TD}good\n"
    )
    r = parse_extraction(text, TD, CD)
    assert len(r.entities) == 1 and r.entities[0].name == "B"
    assert len(r.relations) == 1 and r.relations[0].keywords == ["kw"]


def test_skips_invalid_entity_type_chars() -> None:
    """Types containing `'()<>|/\\` are rejected (matches LightRAG)."""
    text = f"entity{TD}A{TD}<weird>{TD}d\n"
    assert parse_extraction(text, TD, CD).entities == []


def test_handles_records_glued_with_tuple_delim_instead_of_newline() -> None:
    """LLMs sometimes output `entity<|#|>A<|#|>...<|#|>entity<|#|>B<|#|>...` on one line."""
    text = (
        f"entity{TD}A{TD}person{TD}desc-a"
        f"{TD}entity{TD}B{TD}person{TD}desc-b"
    )
    r = parse_extraction(text, TD, CD)
    assert {e.name for e in r.entities} == {"A", "B"}


def test_completion_delim_is_stripped() -> None:
    text = f"entity{TD}A{TD}person{TD}d{CD}"
    r = parse_extraction(text, TD, CD)
    assert len(r.entities) == 1


def test_empty_input_returns_empty_result() -> None:
    r = parse_extraction("", TD, CD)
    assert r.entities == [] and r.relations == []


def test_strips_outer_quotes_from_fields() -> None:
    text = f'entity{TD}"Alice"{TD}"person"{TD}"A person."\n'
    r = parse_extraction(text, TD, CD)
    assert r.entities[0].name == "Alice"
    assert r.entities[0].type == "person"
    assert r.entities[0].description == "A person."


def test_truncates_long_entity_names() -> None:
    long_name = "x" * 500
    text = f"entity{TD}{long_name}{TD}person{TD}d\n"
    r = parse_extraction(text, TD, CD)
    assert len(r.entities[0].name) == 256
