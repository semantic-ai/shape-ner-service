"""
Tests of LLMSegmentorWithShape itself, without the LLM: its configuration, the prompt it builds from the shape and the
examples, the numbered-document index its spans are resolved from, and finding the expressions related to a task. What
a segmentation must produce is tested per case in tests/test_segmentation_cases.py.

The shape, examples and vocabularies come from segmentation.targets in config/config.json (shapes in config/shapes/ and
examples in config/examples/, mounted at /config), so run it inside the service container:
    docker compose exec shape-ner-service sh -c 'cd /app && uv run --with pytest pytest'
To debug, start it under debugpy and attach with "Attach to pytest (Docker)" in .vscode/launch.json:
    docker compose exec shape-ner-service sh -c 'cd /app && uv run --with pytest python -m debugpy --listen 0.0.0.0:5679 --wait-for-client -m pytest'
"""
import pytest
from bs4 import BeautifulSoup
from rdflib import Graph, Namespace, URIRef
from rdflib.namespace import PROV, RDF, SH

from src.config import get_config
from tests.support import case_folders, default_target

CASES = case_folders()
BESLUIT = Namespace("http://data.vlaanderen.be/ns/besluit#")
ELI = Namespace("http://data.europa.eu/eli/ontology#")


def test_shape_input_comes_from_config(segmentor):
    shape_config = get_config().segmentation.targets[default_target()]
    assert [definition["uri"] for definition in segmentor.definitions] == shape_config.node_shapes
    # The vocabulary-specific settings belong to the shape, not to the segmentor
    assert str(segmentor.usage_note_predicate) == shape_config.usage_note_predicate
    assert str(segmentor.text_predicate) == shape_config.text_predicate
    # Every target class gets its superclasses from the configured vocabularies
    for definition in segmentor.definitions:
        superclasses = segmentor._superclasses(definition["target_class"]) - {definition["target_class"]}
        assert superclasses, f"no superclasses of {definition['target_class']} in {shape_config.ontologies}"


def test_prompt_has_the_examples_of_the_usage_notes(segmentor):
    # A property's usage note is the id of its example in the example HTML. A value only in a content attribute (other
    # than a string) or in a resource attribute is hidden metadata: it is not in the visible text
    example = BeautifulSoup(
        segmentor._resolve(get_config().segmentation.targets[default_target()].html_example_path)
        .read_text(encoding="utf-8"), "html.parser")
    checked = 0
    for definition in segmentor.definitions:
        for prop in definition["properties"]:
            element = example.find(id=prop["usage_note"]) if prop["usage_note"] else None
            if element is None:
                continue
            visible = " ".join(element.get_text(" ").split())
            if element.get("content"):
                value, hidden = " ".join(element["content"].split()), prop["datatype"] not in segmentor.STRING_DATATYPES
            elif element.get("resource") and not visible:
                value, hidden = f"<{element['resource']}>", True
            else:
                value, hidden = visible, False
            if len(value) > segmentor.EXAMPLE_MAX_LENGTH:
                value = value[:segmentor.EXAMPLE_MAX_LENGTH].rstrip() + "…"
            # A relation's example is the occurrence it refers to
            # A relation's example says what it can refer to; a relation that nothing in scope can be has none
            hint = segmentor._relation_hint(prop) if prop["class"] else None
            if prop["class"] and not hint:
                assert f"        Example (hidden metadata, not visible text): {value}\n" not in segmentor.system_prompt
                assert f"        Example: {value}\n" not in segmentor.system_prompt
                continue
            label = ("Example (hidden metadata, not visible text)" if hidden else
                     f"Example of what it refers to ({hint})" if hint else "Example")
            assert f"        {label}: {value}\n" in segmentor.system_prompt, f"no example for {prop['key']} {prop['name']}"
            checked += 1
    assert checked, "no property has an example"


def test_occurrence_examples_mark_every_part_of_a_relation(offline_segmentor):
    segmentor = offline_segmentor
    # The decision in the example has two articles: both are marked as parts, each one referred to with a ref
    relations = [(definition, prop) for definition in segmentor.definitions for prop in definition["properties"]
                 if prop.get("example", {}).get("ref") and definition.get("example")]
    assert relations, "no occurrence example has a relation"
    for definition, prop in relations:
        marker = f"[{prop['key']} → {prop['example']['ref']}]"
        assert definition["example"]["value"].count(marker) == 2, f"{marker} in {definition['example']['value']!r}"
        assert f"[/{prop['key']}]" in definition["example"]["value"]


@pytest.fixture(scope="module")
def offline_segmentor():
    """A segmentor that never calls the LLM, so it needs no API key."""
    from tests.support import case_segmentor
    return case_segmentor(record=False)


# The document the span-resolution tests resolve spans in: an uittreksel like the recorded case's, with blank lines
SPANS_TEXT = ("Uittreksel\n\n1. Invoeren van een parkeerverbod.\n\nVerantwoording\nDe straat is te smal.\n\nBESLIST:\n"
              "Artikel 1. Invoeren van een parkeerverbod in de Kerkstraat.\n\n"
              "Artikel 2. Dit besluit treedt in werking.")


def test_the_document_is_indexed_line_by_line(offline_segmentor):
    # Every line is rendered with its 1-based number, and line_offsets[n - 1] is where line n starts in the source
    # the LLM saw; blank lines are numbered too, and the newline at the end adds no empty line
    rendered, line_offsets, source = offline_segmentor._index_document(SPANS_TEXT)
    assert source == SPANS_TEXT
    assert rendered == "\n".join(f"{number:>4}| {line}" for number, line in enumerate(SPANS_TEXT.split("\n"), 1))
    assert line_offsets == [0, 11, 12, 47, 48, 63, 85, 86, 95, 155, 156]
    assert rendered.startswith("   1| Uittreksel\n   2| \n   3| 1. Invoeren van een parkeerverbod.")


def test_the_indexed_document_survives_line_endings_and_truncation(offline_segmentor, monkeypatch):
    s = offline_segmentor
    # A \r stays part of its line, and the final newline does not add an empty line
    rendered, line_offsets, source = s._index_document("één\r\ntwee\r\n")
    assert source == "één\r\ntwee\r\n"
    assert rendered == "   1| één\r\n   2| twee\r"
    assert line_offsets == [0, 5]
    # Truncation happens at the last line boundary <= TEXT_LIMIT, so no partial line is ever indexed
    monkeypatch.setattr(s, "TEXT_LIMIT", 9)
    rendered, line_offsets, source = s._index_document("één\ntwee\ndrie\n")
    assert source == "één\ntwee\n"
    assert rendered == "   1| één\n   2| twee"
    assert line_offsets == [0, 4]
    # Without a line boundary within the limit, the text up to the limit is the one line there is
    monkeypatch.setattr(s, "TEXT_LIMIT", 5)
    rendered, line_offsets, source = s._index_document("zonder regeleinde")
    assert source == "zonde" and rendered == "   1| zonde"
    assert line_offsets == [0]


def test_a_span_is_resolved_from_line_numbers_and_anchors(offline_segmentor):
    # The line numbers slice the document's lines; the verbatim words at the start and end of the span ("first",
    # "last") pin down where in the first and last non-blank line it starts and ends, so the offsets are exact
    s = offline_segmentor
    index = s._index_document(SPANS_TEXT)
    # One whole line
    found = s._resolve_span(index, {"start": 9, "end": 9, "first": "Artikel 1.", "last": "Kerkstraat."}, "N3")
    assert found == (SPANS_TEXT.index("Artikel 1."), SPANS_TEXT.index("Kerkstraat.") + len("Kerkstraat."))
    assert SPANS_TEXT[found[0]:found[1]] == "Artikel 1. Invoeren van een parkeerverbod in de Kerkstraat."
    # Several lines, blank ones included; a range that starts on a blank line starts at its first non-blank line
    found = s._resolve_span(index, {"start": 2, "end": 8, "first": "1.", "last": "BESLIST:"}, "N1")
    assert SPANS_TEXT[found[0]:found[1]] == ("1. Invoeren van een parkeerverbod.\n\nVerantwoording\n"
                                             "De straat is te smal.\n\nBESLIST:")
    # Line numbers that arrive as strings are coerced
    found = s._resolve_span(index, {"start": "9", "end": "9", "first": "Artikel 1.", "last": "Kerkstraat."}, "N3")
    assert SPANS_TEXT[found[0]:found[1]] == "Artikel 1. Invoeren van een parkeerverbod in de Kerkstraat."


def test_anchors_pin_a_span_down_inside_its_lines(offline_segmentor):
    # A value that starts or ends in the middle of a line, one that runs into the next line, and a one-word value
    # ("first" == "last")
    s = offline_segmentor
    index = s._index_document(SPANS_TEXT)
    found = s._resolve_span(index, {"start": 9, "end": 9, "first": "Invoeren van", "last": "Kerkstraat."}, "N3")
    assert SPANS_TEXT[found[0]:found[1]] == "Invoeren van een parkeerverbod in de Kerkstraat."
    found = s._resolve_span(index, {"start": 5, "end": 6, "first": "Verantwoording", "last": "smal."}, "N1")
    assert SPANS_TEXT[found[0]:found[1]] == "Verantwoording\nDe straat is te smal."
    found = s._resolve_span(index, {"start": 9, "end": 9, "first": "1", "last": "1"}, "N3")
    assert SPANS_TEXT[found[0]:found[1]] == "1"


def test_anchors_are_matched_as_literal_text(offline_segmentor):
    # The anchors are matched as the words they are: regex characters in them are literal, and a non-ASCII symbol
    # (see _token_pattern) matches any symbol
    s = offline_segmentor
    text = "Zoek (alle) [drie] woorden.\nVerder."
    found = s._resolve_span(s._index_document(text),
                            {"start": 1, "end": 1, "first": "Zoek (alle)", "last": "[drie] woorden."}, "N1")
    assert text[found[0]:found[1]] == "Zoek (alle) [drie] woorden."
    text = "• Punt één.\nVerder."
    found = s._resolve_span(s._index_document(text), {"start": 1, "end": 1, "first": "· Punt", "last": "één."}, "N1")
    assert text[found[0]:found[1]] == "• Punt één."


def test_line_numbers_that_do_not_pin_a_span_down_are_a_problem(offline_segmentor):
    # Every problem is a string, phrased with the key the LLM knows the span by, so that the feedback loop can
    # correct the answer
    s = offline_segmentor
    index = s._index_document(SPANS_TEXT)
    problems = [
        s._resolve_span(index, {"start": 0, "end": 9, "first": "Artikel", "last": "Kerkstraat."}, "N1"),
        s._resolve_span(index, {"start": 9, "end": 12, "first": "Artikel", "last": "Kerkstraat."}, "N1"),
        s._resolve_span(index, {"start": 9, "end": 3, "first": "Artikel", "last": "Kerkstraat."}, "N1"),
        s._resolve_span(index, {"start": "nine", "end": 9, "first": "Artikel", "last": "Kerkstraat."}, "N1"),
        s._resolve_span(index, {"start": 9, "end": None, "first": "Artikel", "last": "Kerkstraat."}, "N1"),
        s._resolve_span(index, {"start": 10, "end": 10, "first": "Artikel", "last": "Kerkstraat."}, "N1"),
        s._resolve_span(index, {"start": 9, "end": 9}, "N1"),
        s._resolve_span(index, {"start": 9, "end": 9, "first": "Artikel 1.", "last": ""}, "N1"),
        s._resolve_span(index, {"start": 9, "end": 9, "first": 1, "last": "Kerkstraat."}, "N1"),
        s._resolve_span(index, {"start": 9, "end": 9, "first": "Uittreksel", "last": "Kerkstraat."}, "N1"),
        s._resolve_span(index, {"start": 9, "end": 9, "first": "Artikel 1.", "last": "werking"}, "N1"),
        # Anchors that overlap the wrong way around would resolve to a span that contradicts them: it would not
        # start with "first" or not end with "last"
        s._resolve_span(index, {"start": 9, "end": 9, "first": "Artikel 1.", "last": "Artikel"}, "N1"),
        s._resolve_span(s._index_document("ab"), {"start": 1, "end": 1, "first": "b", "last": "ab"}, "N1"),
        # An anchor is matched in its own line: words that run over a line break do not match
        s._resolve_span(index, {"start": 8, "end": 9, "first": "BESLIST: Artikel", "last": "Kerkstraat."}, "N1"),
    ]
    assert all(isinstance(problem, str) and problem.startswith("N1") for problem in problems)
    assert any("outside the document" in problem for problem in problems)
    assert any("reversed" in problem for problem in problems)
    assert any("must be the numbers of the lines" in problem for problem in problems)
    assert any("are blank" in problem for problem in problems)
    assert any("must be a few verbatim words" in problem for problem in problems)
    assert sum("does not start with" in problem for problem in problems) == 2
    assert sum("does not end with" in problem for problem in problems) == 1
    assert sum("does not come before" in problem for problem in problems) == 2
    # A property's key phrases the problem too
    problem = s._resolve_span(index, {"start": 99, "end": 99, "first": "a", "last": "a"}, "N1, property S1.P4")
    assert problem.startswith("N1, property S1.P4")


def test_a_decision_split_in_two_is_one_decision(offline_segmentor):
    # The LLM may split an uittreksel's decision at "BESLIST:" into its motivation and its content, with the relations
    # to the articles on the second part. That part continues the first: there is one decision, a new entity, with the
    # content and the relations of both parts
    s = offline_segmentor
    besluit, artikel = (s.definitions_by_class[str(BESLUIT[name])] for name in ("Besluit", "Artikel"))
    key = {(definition["key"], prop["path"]): prop["key"]
           for definition in (besluit, artikel) for prop in definition["properties"]}
    title, motivation, content, has_part = (key[besluit["key"], str(path)] for path in
                                            (ELI.title, BESLUIT.motivering, PROV.value, ELI.has_part))
    number, article_content = key[artikel["key"], str(ELI.number)], key[artikel["key"], str(PROV.value)]
    text = ("Uittreksel\n\n1. Invoeren van een parkeerverbod.\n\nVerantwoording\nDe straat is te smal.\n\nBESLIST:\n"
            "Artikel 1. Invoeren van een parkeerverbod in de Kerkstraat.\n\nArtikel 2. Dit besluit treedt in werking.")
    # The numbered lines of the document: 3 "1. Invoeren van een parkeerverbod.", 5 "Verantwoording",
    # 6 "De straat is te smal.", 8 "BESLIST:", 9 "Artikel 1. ...", 11 "Artikel 2. ..."
    answer = {"segments": [
        {"id": "N1", "shape": besluit["key"], "start": 3, "end": 8, "first": "1.", "last": "BESLIST:",
         "properties": [{"property": title, "start": 3, "end": 3, "first": "1.", "last": "parkeerverbod."},
                        {"property": motivation, "start": 5, "end": 6, "first": "Verantwoording", "last": "smal."},
                        {"property": content, "start": 8, "end": 8, "first": "BESLIST:", "last": "BESLIST:"}]},
        {"id": "N2", "shape": besluit["key"], "start": 8, "end": 11, "first": "BESLIST:", "last": "werking.",
         "properties": [{"property": content, "start": 9, "end": 11, "first": "Artikel 1.", "last": "werking."},
                        {"property": has_part, "ref": "N3"}, {"property": has_part, "ref": "N4"}]},
        {"id": "N3", "shape": artikel["key"], "start": 9, "end": 9, "first": "Artikel 1.", "last": "Kerkstraat.",
         "properties": [{"property": number, "start": 9, "end": 9, "first": "1", "last": "1"},
                        {"property": article_content, "start": 9, "end": 9,
                         "first": "Invoeren", "last": "Kerkstraat."}]},
        {"id": "N4", "shape": artikel["key"], "start": 11, "end": 11, "first": "Artikel 2.", "last": "werking.",
         "properties": [{"property": number, "start": 11, "end": 11, "first": "2", "last": "2"},
                        {"property": article_content, "start": 11, "end": 11, "first": "Dit", "last": "werking."}]},
    ]}

    problems = []
    segments = s._segments_from_answer(s._index_document(text), answer, problems,
                                       {"graph": Graph(), "entities": {}, "language": "nl"})

    assert problems == []
    decisions = [segment for segment in segments if segment["shape"] == besluit["uri"]]
    assert len(decisions) == 1 and decisions[0]["uri"].startswith("http://data.lblod.info/id/besluit/")
    values = {prop["path"]: prop["text"] for prop in decisions[0]["properties"] if "ref" not in prop}
    assert values[str(PROV.value)].startswith("Artikel 1.")
    articles = [segment["uri"] for segment in segments if segment["shape"] == artikel["uri"]]
    assert len(articles) == 2
    assert sorted(prop["ref"] for prop in decisions[0]["properties"] if "ref" in prop) == sorted(articles)


def test_properties_that_need_an_iri_are_left_out(segmentor):
    # An IRI value that is not a relation to an entity (no sh:class) cannot be taken from the text
    graph = segmentor.shape_graph
    for definition in segmentor.definitions:
        for prop_node in graph.objects(URIRef(definition["uri"]), SH.property):
            path = graph.value(prop_node, SH.path)
            if graph.value(prop_node, SH.nodeKind) in segmentor.IRI_NODE_KINDS and graph.value(prop_node, SH["class"]) is None:
                assert str(path) not in {p["path"] for p in definition["properties"]}
                assert f"path <{path}>" not in segmentor.system_prompt


def test_a_target_without_tuning_divides_documents_by_the_shape_only():
    # The rules on numbered sub-parts and headings belong to the decisions target: without them, the prompt has no
    # such rules and occurrences are not merged
    from src.config import get_config
    from src.library.segmentors import LLMSegmentorWithShape
    target = get_config().segmentation.targets[default_target()].model_copy(
        update={"prompt_rules": [], "numbered_parts": False})
    s = LLMSegmentorWithShape(target, provider="ollama", model_name="replay")
    assert "<TARGET RULES>" not in s.system_prompt
    assert "numbered sub-part" not in s.system_prompt
    # The heading rule is for every target, among the rules on spans
    assert "\n    - A heading is a line of at most 4 words that ends with a colon" in s.system_prompt
    assert "\n    SPANS\n" in s.system_prompt
    segments = [{"uri": "http://example.org/1", "shape": "S", "start": 0, "end": 9,
                 "text": "Artikel 1", "properties": []},
                {"uri": "http://example.org/1.1", "shape": "S", "start": 10, "end": 21,
                 "text": "Artikel 1.1", "properties": []}]
    assert s._merge_sub_parts(segments, "Artikel 1\nArtikel 1.1") == segments


def test_new_entities_can_refer_to_existing_ones_of_the_right_class(offline_segmentor):
    # A related entity is never returned or changed, but a relation of a new entity can refer to it, when it is an
    # instance of the class the relation needs
    s = offline_segmentor
    besluit, artikel = (s.definitions_by_class[str(BESLUIT[name])] for name in ("Besluit", "Artikel"))
    has_part = next(prop["key"] for prop in besluit["properties"] if prop["path"] == str(ELI.has_part))
    text = "1. Een besluit.\n\nArtikel 1. Een artikel."
    article, other = URIRef("http://example.org/artikel"), URIRef("http://example.org/iets")
    graph = Graph()
    graph.add((article, RDF.type, BESLUIT.Artikel))
    graph.add((other, RDF.type, URIRef("http://example.org/Iets")))
    context = {"graph": graph, "language": "nl", "entities": {
        "E1": {"uri": str(article), "types": {str(BESLUIT.Artikel)}},
        "E2": {"uri": str(other), "types": {"http://example.org/Iets"}}}}
    # A ref to an existing entity also has the span where the document mentions it
    answer = {"segments": [
        {"id": "N1", "shape": besluit["key"], "start": 1, "end": 3, "first": "1.", "last": "artikel.",
         "properties": [{"property": has_part, "ref": "E1", "start": 3, "end": 3,
                         "first": "Artikel 1", "last": "Artikel 1"},
                        {"property": has_part, "ref": "E2"}]},
        {"entity": "E1", "shape": artikel["key"], "properties": []},
    ]}

    problems = []
    segments = s._segments_from_answer(s._index_document(text), answer, problems, context)

    assert [segment["key"] for segment in segments] == ["N1"]
    assert [prop["ref"] for prop in segments[0]["properties"] if "ref" in prop] == [str(article)]
    assert any("E2 is not an instance of" in problem for problem in problems)
    assert any("E1 is an existing entity" in problem for problem in problems)
    mention = next(prop for prop in segments[0]["properties"] if "ref" in prop)
    assert (mention["text"], text[mention["start"]:mention["end"]]) == ("Artikel 1", "Artikel 1")
    assert article not in set(Graph().parse(data=s.to_turtle(segments), format="turtle").subjects())


def test_unresolvable_spans_in_an_answer_are_problems(offline_segmentor):
    # A segment without anchors, and a property value whose anchors are not in its lines, are problems phrased with
    # the keys the LLM knows them by; a property that is only a ref needs no span
    s = offline_segmentor
    besluit, artikel = (s.definitions_by_class[str(BESLUIT[name])] for name in ("Besluit", "Artikel"))
    key = {(definition["key"], prop["path"]): prop["key"]
           for definition in (besluit, artikel) for prop in definition["properties"]}
    content, has_part = key[besluit["key"], str(PROV.value)], key[besluit["key"], str(ELI.has_part)]
    text = "Besluit\n\nArtikel 1. Een artikel."
    answer = {"segments": [
        {"id": "N1", "shape": besluit["key"], "start": 1, "end": 1, "first": "Besluit", "last": "Besluit",
         "properties": [{"property": content, "start": 1, "end": 1, "first": "Artikel", "last": "artikel."},
                        {"property": has_part, "ref": "N2"}]},
        {"id": "N2", "shape": artikel["key"], "start": 3, "end": 3, "first": "Artikel 1.", "last": "artikel."},
        {"id": "N3", "shape": artikel["key"], "start": 3, "end": 3},
    ]}

    problems = []
    segments = s._segments_from_answer(s._index_document(text), answer, problems,
                                       {"graph": Graph(), "entities": {}, "language": "nl"})

    assert [segment["key"] for segment in segments] == ["N1", "N2"]
    assert any(problem.startswith(f"N1, property {content}") and "does not start with" in problem
               for problem in problems)
    assert any(problem.startswith("N3") and '"first" and "last"' in problem for problem in problems)
    # The ref-only property still refers to the occurrence it names, and the unresolvable value is left out
    assert [(prop["path"], prop.get("ref")) for prop in segments[0]["properties"]] == \
        [(str(ELI.has_part), segments[1]["uri"])]


def test_each_entity_is_linked_to_its_input_by_an_annotation(monkeypatch):
    # An oa:Annotation per new entity: the entity as body, the input with the entity's position in its text as target
    import src.task.entity_extraction as task_module
    created = []

    class Annotation:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def add_to_triplestore_if_not_exists(self):
            created.append(self.kwargs)
            return f"http://example.org/annotation/{len(created)}"

    monkeypatch.setattr(task_module, "NERAnnotation", Annotation)
    monkeypatch.setattr(task_module, "get_agent_uri", lambda component: f"{component}/agent")
    task = task_module.EntityExtractionTask.__new__(task_module.EntityExtractionTask)
    task.task_uri = "http://example.org/task"
    segments = [{"uri": "http://example.org/besluit/1", "start": 0, "end": 40},
                {"uri": "http://example.org/artikel/1", "start": 20, "end": 40}]

    assert task.annotate("http://example.org/expression", segments) == [
        "http://example.org/annotation/1", "http://example.org/annotation/2"]
    assert [(a["source_uri"], a["class_uri"], a["start"], a["end"]) for a in created] == [
        ("http://example.org/expression", "http://example.org/besluit/1", 0, 40),
        ("http://example.org/expression", "http://example.org/artikel/1", 20, 40)]
    assert {a["activity_id"] for a in created} == {"http://example.org/task"}
    assert {a["agent"] for a in created} == {f"{task_module.EXTRACTOR_COMPONENT}/agent"}


def test_typed_values_and_values_outside_their_occurrence():
    # A typed property (e.g. a date and time) gets its value in the form of its datatype, besides the text it is read
    # from; a value only elsewhere in the document (e.g. a location in a page footer) is found there
    from tests.support import case_segmentor
    s = case_segmentor(record=False, target="uittreksel-zitting")
    zitting = s.definitions_by_class[str(BESLUIT.Zitting)]
    key = {prop["path"]: prop["key"] for prop in zitting["properties"]}
    planned = key["http://data.vlaanderen.be/ns/besluit#geplandeStart"]
    location = key["http://www.w3.org/ns/prov#atLocation"]
    text = "Uittreksel\n\n22 juni 2026\n\nAanwezig: iedereen\n\n__________\n\nStad Zottegem"
    index = s._index_document(text)
    # The numbered lines: 1 "Uittreksel", 3 "22 juni 2026", 5 "Aanwezig: iedereen", 9 "Stad Zottegem"
    date = {"property": planned, "start": 3, "end": 3, "first": "22 juni", "last": "2026"}
    session = {"id": "N1", "shape": zitting["key"], "start": 1, "end": 5, "first": "Uittreksel", "last": "iedereen",
               "properties": [{**date, "value": "2026-06-22T00:00:00"},
                              {"property": location, "start": 9, "end": 9, "first": "Stad", "last": "Zottegem"}]}
    context = {"graph": Graph(), "entities": {}, "language": "nl"}

    problems = []
    segments = s._segments_from_answer(index, {"segments": [session]}, problems, context)
    assert problems == []
    assert [prop["text"] for prop in segments[0]["properties"]] == ["22 juni 2026", "Stad Zottegem"]
    graph = Graph().parse(data=s.to_turtle(segments), format="turtle")
    entity = URIRef(segments[0]["uri"])
    value = graph.value(entity, URIRef("http://data.vlaanderen.be/ns/besluit#geplandeStart"))
    assert (str(value), str(value.datatype)) == ("2026-06-22T00:00:00", "http://www.w3.org/2001/XMLSchema#dateTime")
    assert str(graph.value(entity, URIRef("http://www.w3.org/ns/prov#atLocation"))) == "Stad Zottegem"

    for wrong in ({**date, "value": "22 juni 2026"}, date):
        problems = []
        s._segments_from_answer(index, {"segments": [{**session, "properties": [wrong]}]}, problems, context)
        assert any(f"property {planned}: needs a \"value\"" in problem for problem in problems)


def test_each_usage_note_is_one_element_of_the_example(segmentor):
    # A usage note is the id of the element that is the example: when two elements have it, the first one is taken,
    # whichever it is
    example = BeautifulSoup(
        segmentor._resolve(get_config().segmentation.targets[default_target()].html_example_path)
        .read_text(encoding="utf-8"), "html.parser")
    notes = {segmentor._usage_note(node) for node in segmentor.shape_graph.subjects(SH.path, None)}
    notes |= {segmentor._usage_note(node) for node in segmentor.shape_graph.subjects(RDF.type, SH.NodeShape)}
    duplicated = sorted(note for note in notes - {None} if len(example.find_all(id=note)) > 1)
    assert not duplicated, f"usage notes with more than one element in the example: {duplicated}"


def test_a_relation_out_of_scope_can_be_a_placeholder_per_passage():
    # With placeholders, a relation to a class no structure in scope is of (e.g. the session's minutes, a foaf:Document)
    # can be given as the span of the passage that mentions it: a new entity of that class, one per distinct passage
    from src.config import get_config
    from src.library.segmentors import LLMSegmentorWithShape
    from tests.support import case_segmentor
    s = case_segmentor(record=False, target="uittreksel-zitting")
    assert s.placeholders and "give the span of the text that mentions it" in s.system_prompt
    # The output contract shows such a relation: a span without "ref"
    passage = '{"property": "<relation property key>", "start": 20, "end": 20,'
    assert passage in s.system_prompt and "<PLACEHOLDER CONTRACT>" not in s.system_prompt
    zitting = s.definitions_by_class[str(BESLUIT.Zitting)]
    key = {prop["path"]: prop["key"] for prop in zitting["properties"]}
    minutes, agenda = key[str(BESLUIT.heeftNotulen)], key[str(BESLUIT.heeftAgenda)]
    text = "Uittreksel\n\nuit de notulen van de Gemeenteraad\n\nde agenda van de Gemeenteraad"
    index = s._index_document(text)
    # The numbered lines: 1 "Uittreksel", 3 "uit de notulen ...", 5 "de agenda ..."
    notulen = {"property": minutes, "start": 3, "end": 3, "first": "uit de notulen", "last": "Gemeenteraad"}
    session = {"id": "N1", "shape": zitting["key"], "start": 1, "end": 5, "first": "Uittreksel", "last": "Gemeenteraad",
               "properties": [notulen,
                              # The LLM may also give "ref": "span" with the passage
                              {**notulen, "ref": "span"},
                              {"property": agenda, "start": 5, "end": 5, "first": "de agenda", "last": "Gemeenteraad"}]}
    context = {"graph": Graph(), "entities": {}, "language": "nl"}

    problems = []
    segments = s._segments_from_answer(index, {"segments": [session]}, problems, context)
    assert problems == []
    placeholders = [segment for segment in segments if not segment["shape"]]
    assert [p["text"] for p in placeholders] == ["uit de notulen van de Gemeenteraad", "de agenda van de Gemeenteraad"]
    graph = Graph().parse(data=s.to_turtle(segments), format="turtle")
    for placeholder in placeholders:
        assert (URIRef(placeholder["uri"]), RDF.type, URIRef("http://xmlns.com/foaf/0.1/Document")) in graph
        assert placeholder["uri"].startswith("http://data.lblod.info/id/document/")
    session_uri = URIRef(next(segment["uri"] for segment in segments if segment["shape"]))
    assert len(set(graph.objects(session_uri, BESLUIT.heeftNotulen))) == 1

    # A passage whose span does not resolve is a problem, and no placeholder
    problems = []
    segments = s._segments_from_answer(
        index, {"segments": [{**session, "properties": [{**notulen, "first": "de agenda"}]}]}, problems, context)
    assert [segment for segment in segments if not segment["shape"]] == []
    assert any(problem.startswith(f"N1, property {minutes}") and "does not start with" in problem
               for problem in problems)

    # Without placeholders, the relation needs a ref
    target = get_config().segmentation.targets["uittreksel-zitting"].model_copy(update={"placeholders": False})
    without = LLMSegmentorWithShape(target, provider="ollama", model_name="replay")
    assert passage not in without.system_prompt and "<PLACEHOLDER CONTRACT>" not in without.system_prompt
    problems = []
    without._segments_from_answer(index, {"segments": [session]}, problems, context)
    assert any(f"property {minutes}: relation needs a \"ref\"" in problem for problem in problems)


def test_a_relation_says_what_it_can_refer_to_and_is_checked_on_it(offline_segmentor):
    # A relation says which structure in scope, or which existing entities, can be what it refers to; a ref to an
    # occurrence of another class, or to the occurrence itself, is a problem that says it again
    s = offline_segmentor
    besluit, artikel = (s.definitions_by_class[str(BESLUIT[name])] for name in ("Besluit", "Artikel"))
    has_part = next(prop for prop in besluit["properties"] if prop["path"] == str(ELI.has_part))
    assert f'relation to an instance of <{BESLUIT.Artikel}>: give the "ref" of an {artikel["key"]} occurrence' in s.system_prompt
    context = {"graph": Graph(), "language": "nl", "entities": {
        "E1": {"uri": "http://example.org/artikel", "types": {str(BESLUIT.Artikel)}}}}
    assert s._relation_hint(has_part, context) == \
        f'give the "ref" of an {artikel["key"]} occurrence or the "ref" of an existing entity of that class'

    text = "1. Een besluit.\n\n2. Een ander besluit."
    answer = {"segments": [
        {"id": "N1", "shape": besluit["key"], "start": 1, "end": 1, "first": "1.", "last": "besluit.",
         "properties": [{"property": has_part["key"], "ref": "N1"}]},
        {"id": "N2", "shape": besluit["key"], "start": 3, "end": 3, "first": "2.", "last": "besluit.",
         "properties": [{"property": has_part["key"], "ref": "N1"}]},
    ]}
    problems = []
    segments = s._segments_from_answer(s._index_document(text), answer, problems,
                                       {"graph": Graph(), "entities": {}, "language": "nl"})
    assert all("ref" not in prop for segment in segments for prop in segment["properties"])
    assert f"N1, property {has_part['key']}: a relation never refers to the occurrence itself; give the \"ref\" of an " \
           f"{artikel['key']} occurrence" in problems
    assert f"N2, property {has_part['key']}: N1 is not an instance of <{BESLUIT.Artikel}>; give the \"ref\" of an " \
           f"{artikel['key']} occurrence" in problems
