import uuid

from helpers import query, update, logger
from string import Template
from escape_helpers import sparql_escape_uri, sparql_escape_string
from rdflib import Graph

from decide_ai_service_base.task import DecisionTask
from decide_ai_service_base.sparql_config import get_prefixes_for_query, GRAPHS

from ..config import get_config
from .dedup import get_existing_outputs

EXTRACTOR_COMPONENT = "http://lblod.data.gift/id/components/entity-extraction/v1.0.0"
SHAPE_NER_OPERATION = "http://lblod.data.gift/id/jobs/concept/TaskOperation/shape-ner"
# Triples per INSERT DATA when writing a segmentation to the triplestore
INSERT_BATCH_SIZE = 100


def create_segmentor():
    """The entity extraction's segmentor, LLMSegmentorWithShape, with the segmentation settings from the app config."""
    from ..library.segmentors import LLMSegmentorWithShape
    seg_config = get_config().segmentation
    return LLMSegmentorWithShape(
        provider=seg_config.llm.provider,
        model_name=seg_config.llm.model_name,
        api_key=seg_config.llm.api_key.get_secret_value() if seg_config.llm.api_key else None,
        base_url=seg_config.llm.base_url,
        temperature=seg_config.llm.temperature,
        max_retries=seg_config.llm.max_retries,
        retry_delay=seg_config.llm.retry_delay,
    )


class EntityExtractionTask(DecisionTask):
    """
    Extract the entities of each input expression: segment it along the SHACL shape (see LLMSegmentorWithShape) and
    write the result, the expression with its segments, to the AI graph.
    """

    __task_type__ = SHAPE_NER_OPERATION

    def fetch_eli_expressions(self) -> dict[str, list[str]]:
        """
        Retrieve the ELI expressions and their contents from the task's input container.

        Returns:
            Dictionary containing:
                - "expression_uris": list containing the expression URIs
                - "expression_contents": list containing the expression contents
        """
        q = Template(
            get_prefixes_for_query("task", "epvoc", "eli") +
            f"""
            SELECT ?expression ?content WHERE {{
            GRAPH {sparql_escape_uri(GRAPHS["jobs"])} {{
                $task task:inputContainer ?container .
            }}

            GRAPH {sparql_escape_uri(GRAPHS["data_containers"])} {{
                ?container task:hasResource ?expression .
            }}

            GRAPH {sparql_escape_uri(GRAPHS["expressions"])} {{
                ?expression a eli:Expression ;
                            epvoc:expressionContent ?content .
            }}
            }}
            """
        ).substitute(task=sparql_escape_uri(self.task_uri))

        bindings = query(q, sudo=True).get("results", {}).get("bindings", [])
        if not bindings:
            logger.warning(
                f"No expressions found in input container for task {self.task_uri}")
            return {
                "expression_uris": [],
                "expression_contents": []
            }

        expression_uris = [b["expression"]["value"] for b in bindings]
        expression_contents = [b["content"]["value"] for b in bindings]

        return {
            "expression_uris": expression_uris,
            "expression_contents": expression_contents
        }

    def fetch_segmented(self, expression_uris: list[str], segmentor) -> set[str]:
        """
        The expressions segmented before: the AI graph already holds a value of one of the shape's properties on
        them, e.g. their parts (eli:has_part).
        """
        paths = sorted({prop["path"] for definition in segmentor.definitions for prop in definition["properties"]})
        existing = get_existing_outputs(
            expression_uris,
            f"""
            VALUES ?path {{ {" ".join(sparql_escape_uri(path) for path in paths)} }}
            GRAPH {sparql_escape_uri(GRAPHS["ai"])} {{
                ?expression ?path ?result .
            }}
            """,
        )
        return set(existing)

    def insert_segmentation(self, turtle: str) -> None:
        """Write the segmentation (Turtle from the segmentor) to the AI graph."""
        triples = Graph().parse(data=turtle, format="turtle").serialize(format="nt").splitlines()
        triples = [triple for triple in triples if triple.strip()]
        for i in range(0, len(triples), INSERT_BATCH_SIZE):
            update(
                f"INSERT DATA {{ GRAPH {sparql_escape_uri(GRAPHS['ai'])} {{\n"
                + "\n".join(triples[i:i + INSERT_BATCH_SIZE])
                + "\n} }",
                sudo=True,
            )
        logger.info(f"Wrote {len(triples)} triples of the segmentation to {GRAPHS['ai']}")

    def create_output_container(self, resource: str) -> str:
        """
        Function to create an output data container for the translated ELI expression.

        Args:
            resource: String containing the URI of the translated ELI expression.

        Returns:
            String containing the URI of the output data container
        """
        container_id = str(uuid.uuid4())
        container_uri = f"http://data.lblod.info/id/data-container/{container_id}"

        q = Template(
            get_prefixes_for_query("task", "nfo", "mu") +
            f"""
            INSERT DATA {{
            GRAPH {sparql_escape_uri(GRAPHS["data_containers"])} {{
                $container a nfo:DataContainer ;
                    mu:uuid $uuid ;
                    task:hasResource $resource .
            }}
            }}
            """
        ).substitute(
            container=sparql_escape_uri(container_uri),
            uuid=sparql_escape_string(container_id),
            resource=sparql_escape_uri(resource)
        )

        update(q, sudo=True)
        return container_uri

    def process(self):
        """
        Extract the entities of each input expression and write the result to the AI graph; the expression, with its
        segments, is passed forward.
        """
        eli_expressions = self.fetch_eli_expressions()
        expression_uris = eli_expressions["expression_uris"]
        if not expression_uris:
            return
        skipped_empty = 0

        # The shape, examples and vocabularies are loaded once for all expressions
        segmentor = create_segmentor()

        # Expressions segmented before are passed forward without segmenting them again
        already_segmented = self.fetch_segmented(expression_uris, segmentor)
        if already_segmented:
            logger.info(
                f"{len(already_segmented)} of {len(expression_uris)} "
                f"expressions were already segmented, skipping those")

        for expression_uri, text in zip(expression_uris, eli_expressions["expression_contents"]):
            if expression_uri in already_segmented:
                logger.info(
                    f"Expression {expression_uri} was already "
                    f"segmented, passing it forward without re-segmenting")
                self.results_container_uris.append(
                    self.create_output_container(expression_uri))
                continue

            if not text or not text.strip():
                logger.warning(
                    f"No content available for segmentation on {expression_uri}"
                )
                skipped_empty += 1
                continue

            logger.info(f"Extracting entities of {expression_uri}")
            # The segmentor finds the expression through the task's input container, and enriches it
            result = segmentor.segment_and_validate(text, task=self)
            if not result["conforms"] or result["unlocated"]:
                logger.warning(
                    f"Writing the segmentation of {expression_uri} with problems: "
                    f"{len(result['violations'])} violations, {len(result['unlocated'])} problems in the answer")
            self.insert_segmentation(result["turtle"])

            self.results_container_uris.append(
                self.create_output_container(expression_uri))

        if skipped_empty == len(expression_uris):
            raise RuntimeError(
                f"Entity extraction task {self.task_uri}: "
                f"all {len(expression_uris)} input expressions had empty content, "
                f"nothing was processed"
            )
