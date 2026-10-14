"""
Write the inputs of a segmentation test case from the triplestore: input.ttl with the resource to segment and its
text (the shape config's text_predicate), and context.ttl with the other entities the segmentor should find there
(e.g. an original expression with the same title).

Run it in the service container, e.g.:
    docker compose exec shape-ner-service sh -c 'cd /app && uv run python -m tests.export_case \\
        tests/cases/SEGTC0002-... --input <input URI> --context <original URI>'
Then write patterns.ttl and README.md, and record the case (see tests/test_segmentation_cases.py).
"""
import argparse
from pathlib import Path

from rdflib import BNode, Graph, Literal, URIRef

PREFIXES = {
    "besluit": "http://data.vlaanderen.be/ns/besluit#", 
    "eli": "http://data.europa.eu/eli/ontology#",
    "epvoc": "https://data.europarl.europa.eu/def/epvoc#", 
    "prov": "http://www.w3.org/ns/prov#",
    "dct": "http://purl.org/dc/terms/", 
    "mu": "http://mu.semte.ch/vocabularies/core/",
    "ext": "http://mu.semte.ch/vocabularies/ext/",
}


def fetch(uris: list[str]) -> Graph:
    """All triples of the entities, from any graph."""
    from helpers import query
    from escape_helpers import sparql_escape_uri

    def term(binding):
        if binding["type"] == "uri":
            return URIRef(binding["value"])
        if binding["type"] == "bnode":
            return BNode(binding["value"])
        return Literal(binding["value"], lang=binding.get("xml:lang"),
                       datatype=URIRef(binding["datatype"]) if binding.get("datatype") else None)

    graph = Graph()
    for prefix, namespace in PREFIXES.items():
        graph.bind(prefix, namespace)
    if not uris:
        return graph
    q = ("SELECT DISTINCT ?s ?p ?o WHERE { VALUES ?s { " + " ".join(sparql_escape_uri(u) for u in uris)
         + " } GRAPH ?g { ?s ?p ?o } }")
    for b in query(q, sudo=True)["results"]["bindings"]:
        graph.add((term(b["s"]), term(b["p"]), term(b["o"])))
    return graph


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("folder", type=Path, help="case folder, e.g. tests/cases/SEGTC0002-...")
    parser.add_argument("--input", required=True, help="resource to segment (in the task's input container)")
    parser.add_argument("--context", nargs="*", default=[], help="other entities in the triplestore, e.g. an original")
    args = parser.parse_args()

    args.folder.mkdir(parents=True, exist_ok=True)
    from src.config import get_config
    text_predicate = get_config().segmentation.shape.text_predicate
    resource = fetch([args.input])
    if (URIRef(args.input), URIRef(text_predicate), None) not in resource:
        raise SystemExit(f"{args.input} has no {text_predicate}")
    (args.folder / "input.ttl").write_text(
        f"# Input: the resource in the task's input container, with the text to segment ({text_predicate}).\n"
        + resource.serialize(format="turtle"), encoding="utf-8")
    if args.context:
        (args.folder / "context.ttl").write_text(
            "# Input: entities the triplestore holds besides the expression.\n"
            + fetch(args.context).serialize(format="turtle"), encoding="utf-8")
    print(f"Wrote the inputs of {args.folder}: {len(resource)} input triples, "
          f"{len(args.context)} context entities")


if __name__ == "__main__":
    main()
