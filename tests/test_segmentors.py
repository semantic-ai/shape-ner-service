"""
Tests of LLMSegmentorWithShape itself, without the LLM: its configuration, the prompt it builds from the shape and the
examples, and finding the expressions related to a task. What a segmentation must produce is tested per case in
tests/test_segmentation_cases.py.

The shape, examples and vocabularies come from segmentation.targets in config.json (shape files in shapes/), so run it
inside the service container:
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
            ref = segmentor._structure_of(prop["class"]) if prop["class"] else None
            label = ("Example (hidden metadata, not visible text)" if hidden else
                     f'Example of the {ref} occurrence it refers to (give its "ref")' if ref else "Example")
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
    answer = {"segments": [
        {"id": "N1", "shape": besluit["key"],
         "text": "1. Invoeren van een parkeerverbod.\n\nVerantwoording\nDe straat is te smal.\n\nBESLIST:",
         "properties": [{"property": title, "text": "1. Invoeren van een parkeerverbod."},
                        {"property": motivation, "text": "Verantwoording\nDe straat is te smal."},
                        {"property": content, "text": "BESLIST:"}]},
        {"id": "N2", "shape": besluit["key"], "text": text[text.index("BESLIST:"):],
         "properties": [{"property": content, "text": text[text.index("Artikel 1."):]},
                        {"property": has_part, "ref": "N3"}, {"property": has_part, "ref": "N4"}]},
        {"id": "N3", "shape": artikel["key"], "text": "Artikel 1. Invoeren van een parkeerverbod in de Kerkstraat.",
         "properties": [{"property": number, "text": "1"},
                        {"property": article_content, "text": "Invoeren van een parkeerverbod in de Kerkstraat."}]},
        {"id": "N4", "shape": artikel["key"], "text": "Artikel 2. Dit besluit treedt in werking.",
         "properties": [{"property": number, "text": "2"},
                        {"property": article_content, "text": "Dit besluit treedt in werking."}]},
    ]}

    problems = []
    segments = s._segments_from_answer(text, answer, problems, {"graph": Graph(), "entities": {}, "language": "nl"})

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
    # The heading rule is for every target
    assert "\n    - A heading is a line of at most 4 words that ends with a colon" in s.system_prompt
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
    answer = {"segments": [
        {"id": "N1", "shape": besluit["key"], "text": text,
         "properties": [{"property": has_part, "ref": "E1"}, {"property": has_part, "ref": "E2"}]},
        {"entity": "E1", "shape": artikel["key"], "properties": []},
    ]}

    problems = []
    segments = s._segments_from_answer(text, answer, problems, context)

    assert [segment["key"] for segment in segments] == ["N1"]
    assert [prop["ref"] for prop in segments[0]["properties"] if "ref" in prop] == [str(article)]
    assert any("E2 is not an instance of" in problem for problem in problems)
    assert any("E1 is an existing entity" in problem for problem in problems)
    assert article not in set(Graph().parse(data=s.to_turtle(segments), format="turtle").subjects())


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
