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


def create_segmentor(target_name: str):
    """The entity extraction's segmentor, LLMSegmentorWithShape, for a target of the app config."""
    from ..library.segmentors import LLMSegmentorWithShape
    seg_config = get_config().segmentation
    return LLMSegmentorWithShape(
        seg_config.targets[target_name],
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
    Extract the entities of each input of the task: segment it along the SHACL node shapes of its target (see
    segmentation.targets in the app config, and LLMSegmentorWithShape) and write the result, the input with its
    segments, to the AI graph.
    """

    __task_type__ = SHAPE_NER_OPERATION

    def fetch_inputs(self) -> list[str]:
        """The resources in the task's input container."""
        q = Template(
            get_prefixes_for_query("task") +
            f"""
            SELECT DISTINCT ?resource WHERE {{
            GRAPH {sparql_escape_uri(GRAPHS["jobs"])} {{
                $task task:inputContainer ?container .
            }}

            GRAPH {sparql_escape_uri(GRAPHS["data_containers"])} {{
                ?container task:hasResource ?resource .
            }}
            }}
            """
        ).substitute(task=sparql_escape_uri(self.task_uri))

        bindings = query(q, sudo=True).get("results", {}).get("bindings", [])
        return [b["resource"]["value"] for b in bindings]

    def match_targets(self, inputs: list[str]) -> dict[str, str]:
        """
        The target of each input: the first target, in the order of the app config, whose target_filter matches it.
        Inputs no target matches are left out.
        """
        targets: dict[str, str] = {}
        for name, target in get_config().segmentation.targets.items():
            remaining = [resource for resource in inputs if resource not in targets]
            if not remaining:
                break
            q = (
                f"{target.prefixes}\n"
                f"SELECT DISTINCT ?target WHERE {{\n"
                f"  VALUES ?target {{ {' '.join(sparql_escape_uri(resource) for resource in remaining)} }}\n"
                f"  {target.target_filter}\n"
                f"}}"
            )
            for b in query(q, sudo=True).get("results", {}).get("bindings", []):
                targets.setdefault(b["target"]["value"], name)
        return targets

    def fetch_texts(self, inputs: list[str], text_predicate: str) -> dict[str, str]:
        """The text to segment of each input: its value of the target's text_predicate, in any graph."""
        q = f"""
            SELECT ?resource ?text WHERE {{
            VALUES ?resource {{ {" ".join(sparql_escape_uri(resource) for resource in inputs)} }}
            GRAPH ?g {{
                ?resource {sparql_escape_uri(text_predicate)} ?text .
            }}
            }}
            """
        texts: dict[str, str] = {}
        for b in query(q, sudo=True).get("results", {}).get("bindings", []):
            texts.setdefault(b["resource"]["value"], b["text"]["value"])
        return texts

    def fetch_segmented(self, inputs: list[str], segmentor) -> set[str]:
        """
        The inputs segmented before: the AI graph already holds a value of one of the shape's properties on them,
        e.g. their parts (eli:has_part).
        """
        paths = sorted({prop["path"] for definition in segmentor.definitions for prop in definition["properties"]})
        existing = get_existing_outputs(
            inputs,
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
        """Create an output data container holding the resource; returns its URI."""
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
        Extract the entities of each input that a target applies to, and write the result to the AI graph; the input,
        with its segments, is passed forward.
        """
        inputs = self.fetch_inputs()
        targets = self.match_targets(inputs)
        for resource in inputs:
            if resource not in targets:
                logger.info(f"No target applies to input {resource} of task {self.task_uri}, skipping it")
        if not targets:
            logger.warning(f"No target applies to any input of task {self.task_uri}")
            return
        skipped_empty = 0

        for name in dict.fromkeys(targets.values()):
            resources = [resource for resource in inputs if targets.get(resource) == name]
            # The shape, examples and vocabularies are loaded once per target
            segmentor = create_segmentor(name)
            texts = self.fetch_texts(resources, str(segmentor.text_predicate))

            # Inputs segmented before are passed forward without segmenting them again
            already_segmented = self.fetch_segmented(resources, segmentor)
            if already_segmented:
                logger.info(
                    f"{len(already_segmented)} of {len(resources)} "
                    f"inputs of target {name} were already segmented, skipping those")

            for resource in resources:
                if resource in already_segmented:
                    logger.info(
                        f"Input {resource} was already "
                        f"segmented, passing it forward without re-segmenting")
                    self.results_container_uris.append(
                        self.create_output_container(resource))
                    continue

                text = texts.get(resource)
                if not text or not text.strip():
                    logger.warning(
                        f"No content available for segmentation on {resource}"
                    )
                    skipped_empty += 1
                    continue

                logger.info(f"Extracting entities of {resource} (target {name})")
                # The segmentor finds the input through the task's input container, and enriches it
                result = segmentor.segment_and_validate(text, task=self)
                if not result["conforms"] or result["unlocated"]:
                    logger.warning(
                        f"Writing the segmentation of {resource} with problems: "
                        f"{len(result['violations'])} violations, {len(result['unlocated'])} problems in the answer")
                self.insert_segmentation(result["turtle"])

                self.results_container_uris.append(
                    self.create_output_container(resource))

        if skipped_empty == len(targets):
            raise RuntimeError(
                f"Entity extraction task {self.task_uri}: "
                f"all {len(targets)} inputs had empty content, "
                f"nothing was processed"
            )
