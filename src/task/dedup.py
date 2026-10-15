"""
Helper to detect inputs that were already processed by a previous task run: a task checks for its own existing
outputs in the triplestore, so re-running on already-processed expressions skips the expensive model work while
still passing the expressions downstream.
"""

from helpers import query
from escape_helpers import sparql_escape_uri


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
