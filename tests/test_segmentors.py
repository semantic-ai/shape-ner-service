"""
Tests of LLMSegmentorWithShape itself, without the LLM: its configuration, the prompt it builds from the shape and the
examples, and finding the expressions related to a task. What a segmentation must produce is tested per case in
tests/test_segmentation_cases.py.

The shape, examples and vocabularies come from segmentation.shape in config.json (shape files in shapes/), so run it
inside the service container:
    docker compose exec shape-ner-service sh -c 'cd /app && uv run --with pytest pytest'
To debug, start it under debugpy and attach with "Attach to pytest (Docker)" in .vscode/launch.json:
    docker compose exec shape-ner-service sh -c 'cd /app && uv run --with pytest python -m debugpy --listen 0.0.0.0:5679 --wait-for-client -m pytest'
"""
import pytest
from bs4 import BeautifulSoup
from rdflib import URIRef
from rdflib.namespace import SH

from src.config import get_config
from tests.support import case_folders, load_case, temporary_task

CASES = case_folders()


def test_shape_input_comes_from_config(segmentor):
    shape_config = get_config().segmentation.shape
    assert [definition["uri"] for definition in segmentor.definitions] == shape_config.node_shapes
    # The vocabulary-specific settings belong to the shape, not to the segmentor
    assert str(segmentor.usage_note_predicate) == shape_config.usage_note_predicate
    assert str(segmentor.text_predicate) == shape_config.text_predicate
    assert str(segmentor.same_as_predicate) == shape_config.same_as_predicate
    assert segmentor.same_as_class == shape_config.same_as_class
    # Every target class gets its superclasses from the configured vocabularies
    for definition in segmentor.definitions:
        superclasses = segmentor._superclasses(definition["target_class"]) - {definition["target_class"]}
        assert superclasses, f"no superclasses of {definition['target_class']} in {shape_config.ontologies}"


def test_prompt_has_the_examples_of_the_usage_notes(segmentor):
    # A property's usage note is the id of its example in the example HTML. A value only in a content attribute (other
    # than a string) or in a resource attribute is hidden metadata: it is not in the visible text
    examples = [BeautifulSoup(segmentor._resolve(path).read_text(encoding="utf-8"), "html.parser")
                for path in get_config().segmentation.shape.example_paths]
    checked = 0
    for definition in segmentor.definitions:
        for prop in definition["properties"]:
            element = next((e for soup in examples for e in [soup.find(id=prop["usage_note"])] if e is not None), None) \
                if prop["usage_note"] else None
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
            label = "Example (hidden metadata, not visible text)" if hidden else "Example"
            assert f"        {label}: {value}\n" in segmentor.system_prompt, f"no example for {prop['key']} {prop['name']}"
            checked += 1
    assert checked, "no property has an example"


def test_properties_that_need_an_iri_are_left_out(segmentor):
    # An IRI value that is not a relation to an entity (no sh:class) cannot be taken from the text
    graph = segmentor.shape_graph
    for definition in segmentor.definitions:
        for prop_node in graph.objects(URIRef(definition["uri"]), SH.property):
            path = graph.value(prop_node, SH.path)
            if graph.value(prop_node, SH.nodeKind) in segmentor.IRI_NODE_KINDS and graph.value(prop_node, SH["class"]) is None:
                assert str(path) not in {p["path"] for p in definition["properties"]}
                assert f"path <{path}>" not in segmentor.system_prompt


def test_inputs_are_resources_the_shapes_fit(segmentor):
    # A resource is input when one of the selected shapes fits its types: their target class, or a superclass of it
    # (e.g. an eli:Expression can be enriched as a besluit:Besluit)
    for definition in segmentor.definitions:
        target_class = definition["target_class"]
        for type_ in segmentor._superclasses(target_class):
            assert segmentor._is_input({type_}), f"{type_} fits {target_class}"
    assert not segmentor._is_input({"http://example.org/UnrelatedClass"})
    assert not segmentor._is_input(set())


@pytest.mark.parametrize("folder", CASES, ids=[folder.name for folder in CASES])
def test_related_inputs_come_from_the_task(segmentor, folder):
    # Needs the triplestore: a temporary task whose input container holds the case's input
    case = load_case(folder)
    with temporary_task(case["input"]) as task:
        related = segmentor.fetch_related_entities(task.task_uri, case["text"])
    if not related:
        pytest.skip(f"{case['input']} is not in the triplestore")
    assert related == [case["input"]]
