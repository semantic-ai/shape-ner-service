"""
Helper to detect inputs that were already processed by a previous task run: a task checks for its own existing
outputs in the triplestore, so re-running on already-processed expressions skips the expensive model work while
still passing the expressions downstream.
"""

from helpers import query
from escape_helpers import sparql_escape_uri

from decide_ai_service_base.sparql_config import get_prefixes_for_query, GRAPHS


def get_existing_outputs(
    expression_uris: list[str],
    where_pattern: str,
    prefixes: str = "",
    batch_size: int = 20,
) -> dict[str, list[str]]:
    """
    Return the existing outputs in the triplestore for each given expression.

    Args:
        expression_uris: Expression URIs to check.
        where_pattern: SPARQL graph pattern binding ?expression (a checked
            input) to ?result (an existing output for that input).
        prefixes: PREFIX declarations used by where_pattern, if any.
        batch_size: Maximum number of URIs per SPARQL query.

    Returns:
        Dict mapping each already-processed expression URI to the list of its
        existing output URIs. Expressions without outputs are absent.
    """
    existing: dict[str, list[str]] = {}

    for i in range(0, len(expression_uris), batch_size):
        batch = expression_uris[i:i + batch_size]
        values_clause = " ".join(sparql_escape_uri(u) for u in batch)

        q = f"""
            {prefixes}
            SELECT DISTINCT ?expression ?result WHERE {{
                VALUES ?expression {{ {values_clause} }}
                {where_pattern}
            }}
            """

        for b in query(q, sudo=True).get("results", {}).get("bindings", []):
            expression = b.get("expression", {}).get("value")
            result = b.get("result", {}).get("value")
            if expression and result:
                existing.setdefault(expression, []).append(result)

    return existing


def get_existing_annotations(
    expression_uris: list[str],
    agent_uri: str,
    batch_size: int = 20,
) -> dict[str, list[str]]:
    """
    Return the existing oa:Annotations created by the given AI agent that
    target each given expression. For tasks whose output is annotations
    (e.g. entity extraction), the presence of such annotations means
    the expression was already processed by that task before.

    Args:
        expression_uris: Expression URIs to check.
        agent_uri: URI of the AI component that created the annotations
            (see get_agent_uri).
        batch_size: Maximum number of URIs per SPARQL query.

    Returns:
        Dict mapping each already-annotated expression URI to the list of its
        existing annotation URIs. Expressions without annotations are absent.
    """
    where_pattern = f"""
        GRAPH {sparql_escape_uri(GRAPHS["ai"])} {{
            ?result a oa:Annotation ;
                    oa:hasTarget ?target .
            ?target oa:hasSource ?expression .
            ?activity a prov:Activity ;
                      prov:generated ?result ;
                      prov:wasAssociatedWith {sparql_escape_uri(agent_uri)} .
        }}
    """

    return get_existing_outputs(
        expression_uris,
        where_pattern,
        prefixes=get_prefixes_for_query("oa", "prov"),
        batch_size=batch_size,
    )
