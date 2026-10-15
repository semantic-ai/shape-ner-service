"""
Shared helpers for the segmentation tests: a stand-in task, a temporary task in the triplestore, the LLM usage report,
and test cases in case folders (tests/cases/<id>/, after the RML test cases), with their files:

    README.md       title and description
    input.ttl       input: the resource in the task's input container, with the text to segment (the target's
                    text_predicate)
    context.ttl     input (optional): what the triplestore holds besides it that matters to the segmentation
    prompts/        snapshot of the prompts per attempt (attempt-1-system.txt, attempt-1-user.txt, ...)
    answers.json    the recorded LLM answers, one per attempt
    output.ttl      expected output: new entities as blank nodes, without their mu:uuid
    patterns.ttl    patterns the output must match (see below)
    out/            written by every run, not in git: segments.ttl, the segmentation result as the service writes it

A normal run replays answers.json without the LLM or the triplestore: the segmentor's lookups read input.ttl and
context.ttl. It fails when the prompts differ from the snapshot, the output is not isomorphic to output.ttl, or a
pattern does not match. With SEGMENTATION_RECORD=1 the LLM is called and prompts/, answers.json and output.ttl are
written (review them with git diff); the patterns are still checked. With SEGMENTATION_RECORD=output only output.ttl
is written from a replay: for code changes that change the output but not the prompts.

patterns.ttl has one test:Case node, and everything else in it is a pattern:

    @prefix test: <http://example.org/segmentation-test#> .

    [] a test:Case ;
        test:target "uittreksel" ;                        # target of the app config (optional, default the first)
        test:count [ test:class besluit:Artikel ; test:number 6 ] .   # segmented entities per class

    <...expression> a besluit:Besluit .                   # triples with URIs must be in the result as they are
    [] a besluit:Artikel ;                                # a blank node is any entity: its new URI is not known
        eli:number "1" ;
        prov:value "Invoeren van éénrichtingsverkeer"^^test:startsWith .

Literals are compared on their text, ignoring whitespace differences and language tags; ^^test:startsWith and
^^test:contains compare part of it. Blank nodes linked to each other are matched together.
"""
import copy
import difflib
import json
import uuid
from contextlib import contextmanager
from pathlib import Path

import pytest
from rdflib import BNode, Graph, Literal, Namespace, URIRef
from rdflib.compare import graph_diff, isomorphic, to_isomorphic
from rdflib.namespace import RDF

# Namespace is a str: use TEST["count"], as TEST.count would be str.count
TEST = Namespace("http://example.org/segmentation-test#")


class TaskStub:
    """Stands in for the DecisionTask: only its URI, and the AI calls it would record in the triplestore."""

    def __init__(self, task_uri: str):
        self.task_uri = task_uri
        self.ai_calls: list[dict] = []

    def record_ai_call(self, **call):
        self.ai_calls.append(call)


def default_target() -> str:
    """The first target of the app config."""
    from src.config import get_config
    return next(iter(get_config().segmentation.targets))


def create_segmentor(target: str | None = None):
    """The segmentor EntityExtractionTask uses for the target (default the first), with the app config's settings."""
    from src.config import get_config
    from src.task.entity_extraction import create_segmentor as create_task_segmentor

    llm_config = get_config().segmentation.llm
    if llm_config.provider != "ollama" and not llm_config.api_key:
        pytest.skip(f"No API key configured for segmentation provider {llm_config.provider}")
    return create_task_segmentor(target or default_target())


@contextmanager
def temporary_task(expression: str):
    """A task with one input container holding the expression, like a shape-ner task; removed afterwards."""
    from helpers import update
    from escape_helpers import sparql_escape_uri
    from decide_ai_service_base.sparql_config import GRAPHS

    task_uri = f"http://redpencil.data.gift/id/task/test-{uuid.uuid4()}"
    container = f"http://data.lblod.info/id/data-container/test-{uuid.uuid4()}"
    data = f"""
      GRAPH {sparql_escape_uri(GRAPHS["jobs"])} {{
        {sparql_escape_uri(task_uri)} a <http://redpencil.data.gift/vocabularies/tasks/Task> ;
          <http://redpencil.data.gift/vocabularies/tasks/inputContainer> {sparql_escape_uri(container)} .
      }}
      GRAPH {sparql_escape_uri(GRAPHS["data_containers"])} {{
        {sparql_escape_uri(container)} a <http://www.semanticdesktop.org/ontologies/2007/03/22/nfo#DataContainer> ;
          <http://redpencil.data.gift/vocabularies/tasks/hasResource> {sparql_escape_uri(expression)} .
      }}
    """
    try:
        update(f"INSERT DATA {{ {data} }}", sudo=True)
    except Exception as e:
        pytest.skip(f"Triplestore not reachable: {e}")
    try:
        yield TaskStub(task_uri)
    finally:
        update(f"DELETE DATA {{ {data} }}", sudo=True)


def usage_report(model_name: str, usage: dict, label: str = "") -> list[str]:
    lines = [f"{label}{model_name}: {usage['calls']} call(s), {usage['input_tokens']} input tokens, "
             f"{usage['output_tokens']} output tokens"]
    try:
        # Same OpenRouter pricing as the cost recorded for tasks; needs the triplestore for the model's ext:openrouterId
        from decide_ai_service_base.ai_logging import model_name_to_uri
        from decide_ai_service_base.pricing import calculate_cost
        cost = calculate_cost(model_name_to_uri(model_name), usage["input_tokens"], usage["output_tokens"])
        lines.append(f"{label}Cost: ${cost:.4f}" if cost else f"{label}Cost: unknown (no pricing found for this model)")
    except Exception as e:
        lines.append(f"{label}Cost: unknown ({e})")
    return lines


# Test cases in case folders
CASES_DIR = Path(__file__).parent / "cases"
MU_UUID = URIRef("http://mu.semte.ch/vocabularies/core/uuid")
PREFIXES = {
    "besluit": "http://data.vlaanderen.be/ns/besluit#", "eli": "http://data.europa.eu/eli/ontology#",
    "prov": "http://www.w3.org/ns/prov#", "mu": "http://mu.semte.ch/vocabularies/core/",
    "oa": "http://www.w3.org/ns/oa#", "nif": "http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#",
}


class RecordingExhausted(Exception):
    """The replayed segmentation asks the LLM more often than was recorded."""


def case_folders() -> list[Path]:
    return sorted(folder for folder in CASES_DIR.iterdir() if (folder / "input.ttl").exists())


def load_case(folder: Path) -> dict:
    from src.config import get_config
    patterns = Graph().parse(folder / "patterns.ttl", format="turtle")
    case = next(patterns.subjects(RDF.type, TEST["Case"]), None)
    target = patterns.value(case, TEST["target"]) if case is not None else None
    target = str(target) if target is not None else default_target()
    targets = get_config().segmentation.targets
    if target not in targets:
        raise ValueError(f"{folder.name}/patterns.ttl: unknown target {target!r}, the app config has {list(targets)}")
    text_predicate = URIRef(targets[target].text_predicate)
    input_graph = Graph().parse(folder / "input.ttl", format="turtle")
    inputs = sorted({str(s) for s in input_graph.subjects(text_predicate, None)})
    if len(inputs) != 1:
        raise ValueError(f"{folder.name}/input.ttl must hold one resource with the text to segment ({text_predicate}), "
                         f"found {inputs}")
    context = Graph()
    if (folder / "context.ttl").exists():
        context.parse(folder / "context.ttl", format="turtle")

    counts = list(patterns.objects(case, TEST["count"])) if case is not None else []
    description = {case, *counts}
    expected = Graph(namespace_manager=patterns.namespace_manager)
    for triple in patterns:
        if triple[0] not in description:
            expected.add(triple)

    answers = folder / "answers.json"
    return {
        "name": folder.name,
        "folder": folder,
        "target": target,
        "input": inputs[0],
        "text": str(input_graph.value(URIRef(inputs[0]), text_predicate)),
        "store": input_graph + context,
        "counts": {str(patterns.value(c, TEST["class"])): int(patterns.value(c, TEST["number"])) for c in counts},
        "expected": expected,
        "answers": json.loads(answers.read_text(encoding="utf-8")) if answers.exists() else None,
    }


def use_case_store(segmentor, case: dict) -> None:
    """Let the segmentor's triplestore lookups read the case's input.ttl and context.ttl instead."""
    store = case["store"]

    def fetch_related_entities(task_uri, text=None):
        # As the live lookup: the input counts when its types (from input.ttl and context.ttl) fit the shapes
        types = {str(t) for t in store.objects(URIRef(case["input"]), RDF.type)}
        return [case["input"]] if segmentor._is_input(types) else []

    def fetch_entities(uris):
        graph = Graph()
        for uri in uris:
            for p, o in store.predicate_objects(URIRef(uri)):
                graph.add((URIRef(uri), p, o))
        return graph

    segmentor.fetch_related_entities = fetch_related_entities
    segmentor.fetch_entities = fetch_entities


def capture_llm_calls(segmentor, answers: list | None = None) -> list[dict]:
    """
    Record every LLM call of the segmentor (prompts and answer). With answers, replay those instead of calling the
    LLM: the answers are returned in order, and RecordingExhausted is raised when more are asked.
    """
    calls: list[dict] = []
    call_llm = segmentor.analyzer.analyze_single_entry

    def analyze_single_entry(text, system_prompt, user_prompt_template, expected_schema, text_limit=8000, **kwargs):
        if answers is None:
            answer = call_llm(text, system_prompt, user_prompt_template, expected_schema, text_limit=text_limit, **kwargs)
        elif len(calls) < len(answers):
            answer = copy.deepcopy(answers[len(calls)])
        else:
            raise RecordingExhausted(f"attempt {len(calls) + 1} was not recorded")
        calls.append({"system": system_prompt, "user": user_prompt_template.format(text=text[:text_limit]),
                      "answer": copy.deepcopy(answer)})
        return answer

    segmentor.analyzer.analyze_single_entry = analyze_single_entry
    return calls


def case_segmentor(record: bool, target: str | None = None):
    """A segmentor of its own for a case's target (default the first): replaying needs no LLM, so no API key either."""
    if record:
        return create_segmentor(target)
    from src.config import get_config
    from src.library.segmentors import LLMSegmentorWithShape
    return LLMSegmentorWithShape(get_config().segmentation.targets[target or default_target()],
                                 provider="ollama", model_name="replay")


def run_case(case: dict, record: bool) -> tuple[dict, list[dict], object, TaskStub]:
    """Segment the case's text, replaying its recorded answers or (record) calling the LLM."""
    segmentor = case_segmentor(record, case["target"])
    use_case_store(segmentor, case)
    calls = capture_llm_calls(segmentor, None if record else case["answers"])
    task = TaskStub(f"http://example.org/task/{case['name']}")
    try:
        result = segmentor.segment_and_validate(case["text"], task=task)
    except RuntimeError as e:
        # The segmentor wraps errors of the LLM call, the replay's included
        if isinstance(e.__cause__, RecordingExhausted):
            raise e.__cause__ from None
        raise
    return result, calls, segmentor, task


def normalized_output(result: dict) -> Graph:
    """The output with new entities as blank nodes and without their mu:uuid, which differ per run."""
    graph = Graph().parse(data=result["turtle"], format="turtle")
    new = sorted(URIRef(s["uri"]) for s in result["segments"] if not s["existing"])
    bnodes = {uri: BNode(f"new{i + 1}") for i, uri in enumerate(new)}
    output = Graph()
    for prefix, namespace in PREFIXES.items():
        output.bind(prefix, namespace)
    for s, p, o in graph:
        if s in bnodes and p == MU_UUID:
            continue
        output.add((bnodes.get(s, s), p, bnodes.get(o, o)))
    return output


def write_result(case: dict, result: dict) -> None:
    out = case["folder"] / "out"
    out.mkdir(exist_ok=True)
    (out / "segments.ttl").write_text(result["turtle"], encoding="utf-8")


def write_output(case: dict, result: dict) -> None:
    (case["folder"] / "output.ttl").write_text(normalized_output(result).serialize(format="turtle"), encoding="utf-8")


def write_recording(case: dict, result: dict, calls: list[dict]) -> None:
    folder = case["folder"]
    prompts = folder / "prompts"
    prompts.mkdir(exist_ok=True)
    for old in prompts.glob("attempt-*.txt"):
        old.unlink()
    for i, call in enumerate(calls, start=1):
        (prompts / f"attempt-{i}-system.txt").write_text(call["system"], encoding="utf-8")
        (prompts / f"attempt-{i}-user.txt").write_text(call["user"], encoding="utf-8")
    (folder / "answers.json").write_text(
        json.dumps([call["answer"] for call in calls], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_output(case, result)


def compare_prompts(case: dict, calls: list[dict]) -> list[str]:
    failures = []
    prompts = case["folder"] / "prompts"
    for i, call in enumerate(calls, start=1):
        for part in ("system", "user"):
            path = prompts / f"attempt-{i}-{part}.txt"
            recorded = path.read_text(encoding="utf-8") if path.exists() else ""
            if call[part] != recorded:
                diff = list(difflib.unified_diff(recorded.splitlines(), call[part].splitlines(),
                                                 f"recorded {path.name}", "now", n=1, lineterm=""))
                failures.append(f"{part} prompt of attempt {i} changed; record the case again if that is intended:\n"
                                + "\n".join("    " + line for line in diff[:20]))
    return failures


def compare_output(case: dict, result: dict) -> list[str]:
    path = case["folder"] / "output.ttl"
    if not path.exists():
        return [f"{case['name']} has no output.ttl: record the case"]
    expected = Graph().parse(path, format="turtle")
    actual = normalized_output(result)
    if isomorphic(actual, expected):
        return []
    _, only_actual, only_expected = graph_diff(to_isomorphic(actual), to_isomorphic(expected))
    lines = ["output is not isomorphic to output.ttl:"]
    for label, graph in (("only in the output", only_actual), ("only in output.ttl", only_expected)):
        triples = sorted(graph, key=lambda t: tuple(map(str, t)))
        lines.append(f"  {label} ({len(triples)} triples):")
        lines += [f"    {s.n3()} {p.n3()} {' '.join(o.n3().split())[:100]}" for s, p, o in triples[:10]]
    return ["\n".join(lines)]


def check_patterns(case: dict, result: dict, segmentor) -> list[str]:
    """Everything in patterns.ttl that the segmentation result does not satisfy, one line (or block) per failure."""
    failures = []

    # Segmented entities: new ones and enriched existing ones
    actual_counts: dict[str, int] = {}
    for segment in result["segments"]:
        target_class = segmentor.target_classes.get(segment["shape"])
        actual_counts[target_class] = actual_counts.get(target_class, 0) + 1
    for target_class, number in case["counts"].items():
        if actual_counts.get(target_class, 0) != number:
            failures.append(f"expected {number} entities of {target_class}, found {actual_counts.get(target_class, 0)}")

    graph = Graph().parse(data=result["turtle"], format="turtle")
    for group in _groups(case["expected"]):
        if not graph.query(_ask(group)).askAnswer:
            failures.append("pattern not in the output:\n" + _describe(group, case["expected"]))
    return failures


def _groups(expected: Graph) -> list[list[tuple]]:
    """The expected triples, grouped so that triples sharing a blank node are matched together."""
    groups: list[tuple[set, list]] = []  # (blank nodes, triples)
    for triple in sorted(expected, key=lambda t: tuple(map(str, t))):
        bnodes = {term for term in (triple[0], triple[2]) if isinstance(term, BNode)}
        joined = [g for g in groups if bnodes & g[0]]
        merged = (set(bnodes), [triple])
        for g in joined:
            merged[0].update(g[0])
            merged[1].extend(g[1])
            groups.remove(g)
        groups.append(merged)
    return [sorted(triples, key=lambda t: tuple(map(str, t))) for _, triples in groups]


def _normalize(text: str) -> str:
    return " ".join(text.split())


def _ask(triples: list[tuple]) -> str:
    variables: dict = {}

    def term(node) -> str:
        if isinstance(node, BNode):
            return variables.setdefault(node, f"?b{len(variables)}")
        return node.n3()

    patterns, filters = [], []
    for i, (s, p, o) in enumerate(triples):
        if not isinstance(o, Literal):
            patterns.append(f"{term(s)} {term(p)} {term(o)} .")
            continue
        value = f"?v{i}"
        patterns.append(f"{term(s)} {term(p)} {value} .")
        actual = f'REPLACE(STR({value}), "\\\\s+", " ")'
        expected = Literal(_normalize(str(o))).n3()
        if o.datatype == TEST["startsWith"]:
            filters.append(f"STRSTARTS({actual}, {expected})")
        elif o.datatype == TEST["contains"]:
            filters.append(f"CONTAINS({actual}, {expected})")
        else:
            filters.append(f"{actual} = {expected}")
    return "ASK { " + " ".join(patterns) + " " + " ".join(f"FILTER({f})" for f in filters) + " }"


def _describe(triples: list[tuple], expected: Graph) -> str:
    graph = Graph(namespace_manager=expected.namespace_manager)
    for triple in triples:
        graph.add(triple)
    lines = graph.serialize(format="turtle").splitlines()
    return "\n".join("    " + line for line in lines if line.strip() and not line.startswith("@prefix"))
