"""
Segmentation test cases in case folders (tests/cases/<id>/, after the RML test cases; see tests/support.py).

A normal run replays each case's recorded LLM answers, so it needs neither the LLM nor the triplestore. Per case it
checks the recording (prompts, answers, output.ttl) and the case's own patterns.ttl, and then the general checks below,
which hold for every segmentation. These are written in terms of the shape and the segmentor's settings, so anything
specific to a document belongs in its patterns.ttl. To add a case, see tests/cases/README.md.

Record a case (calls the LLM and writes prompts/, answers.json and output.ttl) with:
    docker compose exec shape-ner-service sh -c 'cd /app && SEGMENTATION_RECORD=1 uv run --with pytest pytest tests/test_segmentation_cases.py -k SEGTC0001'
When a code change changes the output but not the prompts, SEGMENTATION_RECORD=output rewrites only output.ttl from
the recorded answers, without the LLM.
"""
import os
import re
import uuid
from types import SimpleNamespace

import pytest
from rdflib import Graph, URIRef
from rdflib.namespace import RDF

from tests.support import (RecordingExhausted, case_folders, check_patterns, compare_output,
                           compare_prompts, load_case, run_case, usage_report, write_output, write_recording,
                           write_result)

RECORD = os.environ.get("SEGMENTATION_RECORD") == "1"
UPDATE_OUTPUT = os.environ.get("SEGMENTATION_RECORD") == "output"
CASES = case_folders()


@pytest.fixture(scope="module", params=CASES, ids=[folder.name for folder in CASES])
def run(request, report_lines: list[str]) -> SimpleNamespace:
    """The case, segmented once: replayed from its recording, or recorded with SEGMENTATION_RECORD=1."""
    folder = request.param
    case = load_case(folder)
    if not RECORD and case["answers"] is None:
        pytest.skip(f"{folder.name} has not been recorded yet: run it with SEGMENTATION_RECORD=1")
    try:
        result, calls, segmentor, task = run_case(case, record=RECORD)
    except RecordingExhausted as e:
        pytest.fail(f"{folder.name}: the segmentation asks the LLM more often than recorded ({e}); record it again")
    write_result(case, result)
    if RECORD:
        write_recording(case, result, calls)
        report_lines.extend(usage_report(segmentor.model_name, result["usage"], label=f"{folder.name}: "))
    elif UPDATE_OUTPUT:
        write_output(case, result)
    return SimpleNamespace(case=case, result=result, calls=calls, segmentor=segmentor, task=task,
                           graph=Graph().parse(data=result["turtle"], format="turtle"))


def new_by_shape(run) -> dict[str, list[dict]]:
    """New entities per node shape, in document order."""
    shapes: dict[str, list[dict]] = {}
    for segment in sorted((s for s in run.result["segments"] if not s["existing"]), key=lambda s: s["start"]):
        shapes.setdefault(segment["shape"], []).append(segment)
    return shapes


# The case's own expectations
def test_recording(run):
    if RECORD or UPDATE_OUTPUT:
        pytest.skip("the recording was just written")
    failures = compare_prompts(run.case, run.calls)
    if len(run.calls) < len(run.case["answers"]):
        failures.append(f"only {len(run.calls)} of the {len(run.case['answers'])} recorded answers were used; "
                        f"record the case again")
    failures += compare_output(run.case, run.result)
    assert not failures, "\n".join(failures)


def test_patterns(run):
    failures = check_patterns(run.case, run.result, run.segmentor)
    assert not failures, f"{len(failures)} pattern(s) of {run.case['name']}/patterns.ttl not met:\n" + "\n".join(failures)


# General checks
def test_conforms_within_the_attempts(run):
    assert 1 <= run.result["attempts"] <= run.segmentor.MAX_ATTEMPTS
    assert run.result["conforms"], "\n".join(run.segmentor._describe_violation(v) for v in run.result["violations"])


def test_texts_point_to_the_source_text(run):
    text = run.case["text"]
    for segment in run.result["segments"]:
        if not segment["existing"]:
            assert text[segment["start"]:segment["end"]] == segment["text"]
        for prop in segment["properties"]:
            if "ref" in prop:
                continue
            assert text[prop["start"]:prop["end"]] == prop["text"]
            if not segment["existing"]:
                assert segment["start"] <= prop["start"] <= prop["end"] <= segment["end"], \
                    f"{prop['path']} of {segment['key']} lies outside it"


def test_occurrences_of_a_structure_do_not_overlap(run):
    for shape, segments in new_by_shape(run).items():
        for segment, following in zip(segments, segments[1:]):
            assert segment["end"] <= following["start"], \
                f"{shape}: {segment['text'][:40]!r} [{segment['start']}:{segment['end']}] overlaps " \
                f"{following['text'][:40]!r} [{following['start']}:{following['end']}]"


def test_numbered_parts_are_whole(run):
    # When the occurrences of a structure are numbered, each starts with its own number: a sub-part (e.g. 1.1 after 1)
    # or an unnumbered continuation (e.g. after a page break) is part of the occurrence before it
    if not run.segmentor.numbered_parts:
        pytest.skip("the case's target has no numbered parts")
    for shape, segments in new_by_shape(run).items():
        labels = [run.segmentor._numbered_label(s["text"]) for s in segments]
        if not any(labels):
            continue
        unnumbered = [s["text"][:40] for s, label in zip(segments, labels) if label is None]
        assert not unnumbered, f"{shape}: occurrences without a number: {unnumbered}"
        numbers = {label for label in labels}
        sub_parts = [f"{word} {number}" for word, number in numbers
                     if any(word == w and number.startswith(n + ".") for w, n in numbers)]
        assert not sub_parts, f"{shape}: sub-parts returned as occurrences of their own: {sub_parts}"


def test_values_do_not_start_or_end_with_a_heading(run):
    # A heading line (e.g. "BESLIST:" between a motivation and the decision itself) belongs to no property
    for segment in run.result["segments"]:
        for prop in segment["properties"]:
            lines = [line for line in prop.get("text", "").split("\n") if line.strip()]
            if len(lines) > 1:
                assert not run.segmentor._is_heading_line(lines[0]), f"{prop['path']} of {segment['key']} starts with {lines[0]!r}"
                assert not run.segmentor._is_heading_line(lines[-1]), f"{prop['path']} of {segment['key']} ends with {lines[-1]!r}"


def test_new_entities_have_a_minted_uri_and_uuid(run):
    uuids = []
    for segments in new_by_shape(run).values():
        for segment in segments:
            target_class = run.segmentor.target_classes.get(segment["shape"]) or segment["shape"]
            class_name = re.split(r"[#/]", target_class.rstrip("#/"))[-1].lower()
            assert segment["uri"] == run.segmentor.URI_TEMPLATE.format(class_name=class_name, uuid=uuid.UUID(segment["uuid"]))
            assert (URIRef(segment["uri"]), RDF.type, URIRef(target_class)) in run.graph
            assert str(run.graph.value(URIRef(segment["uri"]), run.segmentor.MU_UUID)) == segment["uuid"]
            uuids.append(segment["uuid"])
    assert len(set(uuids)) == len(uuids)


def test_ai_calls_are_recorded_on_the_task(run):
    # Only a recording calls the LLM, and so records the calls on the task
    if not RECORD:
        pytest.skip("only checked when recording")
    assert len(run.task.ai_calls) == len(run.calls)
