import re
import json
import uuid
import difflib
import unicodedata
from pathlib import Path
from string import Template
import requests
from typing import Optional, Any, List, Dict
from helpers import logger
from rdflib import Graph, URIRef, Literal, BNode, Namespace
from rdflib.namespace import SH, RDF, RDFS
from rdflib.collection import Collection
from pyshacl import validate as shacl_validate
from bs4 import BeautifulSoup

from .LLMAnalyzer import LLMAnalyzer


# Shape-driven LLM segmentor: what to look for is read from a SHACL shape, the LLM only returns
# the verbatim text it found and the character offsets are located deterministically in the source text.
class LLMSegmentorWithShape:
    """
    Segmentor using LLMAnalyzer, driven by a SHACL shape.

    Every selected sh:NodeShape (and its sh:property shapes) is described to the LLM, together with the
    existing entities related to the document (fetched from the SPARQL endpoint). The result enriches
    those entities and adds the entities found in the text, linked through the shape's own properties.

    segment_and_validate() returns, besides the validation, Turtle with only the new triples:
        <new entity> a <sh:targetClass> ;
            mu:uuid "{uuid}" ;
            <sh:path> "value found in the text" .
        <existing entity> a <sh:targetClass of the shape it plays> ;
            <relation sh:path> <new entity> .
    A new entity's URI is uri_template filled with a new UUID and {class_name}, the lowercased local name of the
    shape's sh:targetClass: e.g. http://data.lblod.info/id/artikel/{uuid} for a besluit:Artikel.
    and the same entities with their character offsets ('segments'):
        {'key': 'N1' | 'E1', 'shape': <node shape URI>, 'uri': str, 'uuid': str | None, 'existing': bool,
         'text': str | None, 'start': int | None, 'end': int | None, 'add_type': str | None,
         'properties': [{'path': <sh:path URI>, 'text': str, 'start': int, 'end': int}
                        | {'path': <sh:path URI>, 'ref': <entity URI>}]}
    The Turtle is validated against the SHACL shape, and the LLM is asked again with the violations, up to
    MAX_ATTEMPTS times.
    """

    SYSTEM_PROMPT = """
    You are a precise document-segmentation assistant.

    You receive a document and a description of one or more STRUCTURES, derived from
    a SHACL shape. Each structure has a key (e.g. "S1") and may have PROPERTIES with
    their own keys (e.g. "S1.P1"). You may also receive EXISTING ENTITIES (keys
    "E1", "E2", ...) that are already known for this document.

    TASK
    - Find every occurrence of each structure in the document.
    - For each occurrence, also find the value of each of its properties that is
      present inside that occurrence.
    - Only use the keys listed under STRUCTURES and EXISTING ENTITIES. Omit
      structures and properties that are absent; never invent content.
    - Cardinalities (min/max) are only a suggestion: omit a property when its
      value is not in the document.
    - Examples are taken from another document: they show what an occurrence or
      value looks like, never copy them. A value whose example is hidden
      metadata is usually not in the document text.
    - An occurrence's example marks its properties with [key] ... [/key]: divide
      occurrences in the document the same way. A property's value runs from
      where its part starts to where it ends, including any headings and
      sections inside it. Only text between two parts, outside all markers,
      belongs to no property.
    - A relation's parts are marked [key → structure key] ... [/key]: each is an
      occurrence of that structure of its own, which the relation refers to
      with "ref", one ref per part.

    EXISTING ENTITIES
    - When an occurrence is one of the existing entities, enrich that entity
      instead of creating a new occurrence: refer to it with "entity" and give the
      structure it plays in "shape" (one of its "can be enriched as" structures).
      It needs no "text". Only give the properties it does not have yet.
    - An existing entity marked "its text is the text below" holds the whole text.
      When the text holds one occurrence of a structure that entity can be
      enriched as, enrich the entity. When it holds several, they are new
      occurrences, not that entity.

    RELATIONS
    - A property marked as a relation refers to another occurrence or existing
      entity: give its key in "ref" ("N..." id of an occurrence or "E..." key of
      an existing entity) instead of "text".
    - The relation is given on the occurrence or entity that has the property.
    - Always give every relation the document supports between the occurrences
      and existing entities you return.

    COPYING TEXT
    - "text": the complete text of the occurrence or property value, copied
      verbatim from the document.
    - Copy character for character: no paraphrasing, translation, summarizing,
      normalization or added punctuation.
    - An occurrence includes all of its content: every line, list, bullet point
      and continuation (also across page breaks), up to where the next part of
      the document starts, such as the next occurrence. Do not stop after the
      first sentence or line.
    - Within one structure, a numbered sub-part (such as 1.1 or 2.1 under 1 or
      2) is part of the occurrence it belongs to, not a separate occurrence of
      that structure: the occurrence includes its sub-parts and ends where the
      next part at the same level starts (2 after 1).
    - One occurrence holds all of its properties: never split it into several
      occurrences of the same structure, e.g. at a heading between two of its
      properties.
    - Occurrences of different structures are always returned separately, also
      when one contains the other.
    - Copy everything between the start and the end of the text, including page
      headers, footers and page numbers that interrupt it: the text must be one
      contiguous piece of the document.

    OUTPUT CONTRACT
    Respond with STRICT JSON only, no markdown fences or commentary, exactly:
    {"segments": [
      {"id": "N1", "shape": "<structure key>", "text": "...",
       "properties": [{"property": "<property key>", "text": "..."},
                      {"property": "<relation property key>", "ref": "<N or E key>"}]},
      {"entity": "<E key>", "shape": "<structure key>",
       "properties": [...]}
    ]}
    Every new occurrence has a unique "id" (N1, N2, ...). List new occurrences in
    document order. Return {"segments": []} if nothing is found.

    STRUCTURES
    """

    USER_PROMPT_TEMPLATE = """
    Find all STRUCTURES described in the system instructions in the document below
    and return STRICT JSON as specified.

    Document:
    {text}
    """

    FEEDBACK_TEMPLATE = """
    PREVIOUS ATTEMPT
    Your previous answer was converted to RDF and validated against the SHACL shape.
    Correct the problems below where the document supports it. Keep every occurrence
    that was correct: do not drop occurrences to avoid problems, and never invent
    content that is not in the document. Texts that could not be found must be
    copied again, character for character, from the document.

    Previous answer:
    {previous_answer}

    Problems:
    {problems}
    """

    # Cardinalities are only a suggestion: properties that are not in the text (e.g. a language) stay unannotated
    IGNORED_CONSTRAINTS = {str(SH.MinCountConstraintComponent), str(SH.MaxCountConstraintComponent)}

    MU_UUID = URIRef("http://mu.semte.ch/vocabularies/core/uuid")
    # Annotation linking a segmented decision to its original, as the other annotations of the service
    OA = Namespace("http://www.w3.org/ns/oa#")
    NIF_CONFIDENCE = URIRef("http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#confidence")
    ANNOTATION_TEMPLATE = "http://data.lblod.info/id/annotations/{uuid}"
    IRI_NODE_KINDS = {SH.IRI, SH.BlankNode, SH.BlankNodeOrIRI}
    STRING_DATATYPES = {"http://www.w3.org/2001/XMLSchema#string", str(RDF.langString)}

    # Vocabularies are published as Turtle (e.g. OSLO) or RDF/XML only (e.g. ELI)
    ONTOLOGY_ACCEPT = "text/turtle, application/rdf+xml;q=0.9, */*;q=0.1"
    RDF_FORMATS = {"text/turtle": "turtle", "application/rdf+xml": "xml", "application/xml": "xml",
                   "application/ld+json": "json-ld", "application/n-triples": "nt", "text/n3": "n3"}
    _ontology_cache: Dict[str, Graph] = {}

    RESULTS_SCHEMA = {
        "segments": {"default": [], "type": list},
    }

    # Settings; a subclass can override them
    TEXT_LIMIT = 28000
    MAX_ATTEMPTS = 3
    # {class_name} is the lowercased local name of the shape's sh:targetClass, e.g. "artikel"
    URI_TEMPLATE = "http://data.lblod.info/id/{class_name}/{uuid}"
    EXAMPLE_MAX_LENGTH = 400
    # An occurrence's example shows how it is divided into its properties, so it gets more room
    OCCURRENCE_EXAMPLE_MAX_LENGTH = 2000
    # Language tag for rdf:langString values; detected from the text when None
    LANGUAGE: Optional[str] = None
    # Text the LLM may leave out between two lines it copied, e.g. a page footer
    MAX_SKIPPED_CHARS = 300
    # Minimal similarity of the titles of a segmented entity and an original (shape config: same_as_predicate)
    TITLE_SIMILARITY = 0.9
    # Relative paths in the shape config are resolved against the project root
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    def __init__(self, api_key: str = None, base_url: str = None, model_name: str = "mistral-nemo", temperature: float = 0.0, provider: str = "ollama", max_retries: int = 3, retry_delay: float = 15.0):
        self.model_name = model_name

        # Shape, node shapes, examples, vocabularies and the vocabulary-specific settings come from the app config
        # (segmentation.shape)
        shape_config = self._load_shape_config()
        self.usage_note_predicate = URIRef(shape_config.usage_note_predicate)
        self.text_predicate = URIRef(shape_config.text_predicate)
        self.same_as_predicate = URIRef(shape_config.same_as_predicate)
        self.same_as_class = shape_config.same_as_class
        self.shape_graph = Graph().parse(self._resolve(shape_config.shape_path), format="turtle")
        # rdfs:subClassOf hierarchy, e.g. besluit:Besluit -> eli:LegalExpression -> eli:Expression
        self.ontology = self._load_ontologies(shape_config.ontologies)
        self.definitions = self._load_shape_definitions(shape_config.node_shapes)
        if not self.definitions:
            raise ValueError("No sh:NodeShape found in the given SHACL shape.")
        self._load_examples([self._resolve(path).read_text(encoding="utf-8") for path in shape_config.example_paths])

        # Key used in the prompt -> {'shape': node shape URI, 'path': property path URI or None}
        self.key_lookup: Dict[str, Dict[str, Optional[str]]] = {}
        self.target_classes = {definition["uri"]: definition["target_class"] for definition in self.definitions}
        self.definitions_by_class = {definition["target_class"]: definition for definition in self.definitions
                                     if definition["target_class"]}
        self.properties_by_key: Dict[str, Dict[str, Any]] = {}
        for definition in self.definitions:
            self.key_lookup[definition["key"]] = {"shape": definition["uri"], "path": None}
            for prop in definition["properties"]:
                self.key_lookup[prop["key"]] = {"shape": definition["uri"], "path": prop["path"]}
                self.properties_by_key[prop["key"]] = prop

        self.system_prompt = self.SYSTEM_PROMPT + self._render_shape_definitions()

        self.analyzer = LLMAnalyzer(
            provider=provider,
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )

    def _load_shape_config(self):
        from ..config import get_config
        return get_config().segmentation.shape

    def _resolve(self, path: str) -> Path:
        return Path(path) if Path(path).is_absolute() else self.PROJECT_ROOT / path

    # Class hierarchy
    def _load_ontologies(self, ontologies: Optional[List[str | Graph]]) -> Graph:
        graph = Graph()
        for ontology in ontologies or []:
            graph += ontology if isinstance(ontology, Graph) else self._fetch_ontology(ontology)
        return graph

    def _fetch_ontology(self, url: str) -> Graph:
        """Fetch a vocabulary by URL with content negotiation, cached per process."""
        if url not in self._ontology_cache:
            response = requests.get(url, headers={"Accept": self.ONTOLOGY_ACCEPT}, timeout=60)
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "").split(";")[0].strip()
            rdf_format = self.RDF_FORMATS.get(content_type)
            if rdf_format is None:
                raise ValueError(f"Unsupported content type {content_type!r} for ontology {url}")
            self._ontology_cache[url] = Graph().parse(data=response.content, format=rdf_format)
            logger.info(f"Loaded ontology {url}: {len(self._ontology_cache[url])} triples")
        return self._ontology_cache[url]

    def _superclasses(self, class_uri: str) -> set[str]:
        """The class itself and all its (transitive) rdfs:subClassOf superclasses."""
        return {str(c) for c in self.ontology.transitive_objects(URIRef(class_uri), RDFS.subClassOf)}

    def _is_input(self, types: set[str]) -> bool:
        """A resource with these types is something to segment: it can be (enriched as) one of the selected shapes."""
        return any(self._can_play(types, definition["target_class"]) for definition in self.definitions)

    def _can_play(self, types: set[str], target_class: Optional[str]) -> bool:
        """An entity of the given types can be (enriched as) an instance of target_class."""
        return bool(target_class) and bool(types & self._superclasses(target_class))

    # Shape reading
    def _value(self, subject, predicate) -> Optional[str]:
        value = self.shape_graph.value(subject, predicate)
        return str(value) if value is not None else None

    def _local_name(self, node, fallback_uri: str) -> str:
        """Name of a shape: its sh:name (or rdfs:label), otherwise the last segment of fallback_uri."""
        name = self._value(node, SH.name) or self._value(node, RDFS.label)
        if name:
            return name
        return re.split(r"[#/]", fallback_uri.rstrip("#/"))[-1] or fallback_uri

    def _expects_literal(self, prop_node) -> bool:
        """The property shape has a sh:datatype, or sh:or alternatives with one (e.g. string or langString)."""
        if self.shape_graph.value(prop_node, SH.datatype) is not None:
            return True
        alternatives = self.shape_graph.value(prop_node, SH["or"])
        return alternatives is not None and any(self.shape_graph.value(alternative, SH.datatype) is not None
                                                for alternative in Collection(self.shape_graph, alternatives))

    def _defined_paths(self, graph: Graph, subject, shape_uri: str) -> set[str]:
        """
        Paths for which the entity already has a value that counts: for a property that expects a literal, an IRI
        (e.g. prov:value holding a link to the PDF publication) does not, so the text can still be added.
        """
        definition = next((d for d in self.definitions if d["uri"] == shape_uri), None)
        literal_paths = {p["path"] for p in definition["properties"] if p["literal"]} if definition else set()
        return {str(p) for p, o in graph.predicate_objects(subject)
                if not (str(p) in literal_paths and not isinstance(o, Literal))}

    def _load_shape_definitions(self, node_shapes: Optional[List[str]]) -> List[Dict[str, Any]]:
        """Read the selected node shapes and their property shapes from the shape graph."""
        g = self.shape_graph
        if node_shapes:
            shape_nodes = [URIRef(uri) for uri in node_shapes]
            missing = [str(s) for s in shape_nodes if (s, RDF.type, SH.NodeShape) not in g]
            if missing:
                raise ValueError(f"Node shapes not found in SHACL shape: {missing}")
        else:
            shape_nodes = sorted(set(g.subjects(RDF.type, SH.NodeShape)), key=str)

        definitions = []
        for i, shape_node in enumerate(shape_nodes, start=1):
            key = f"S{i}"
            target_class = self._value(shape_node, SH.targetClass)
            properties = []
            for prop_node in g.objects(shape_node, SH.property):
                path = g.value(prop_node, SH.path)
                if not isinstance(path, URIRef):
                    # Complex property paths (sequences, inverses, ...) do not map to a single URI
                    logger.warning(f"Skipping property shape without a simple sh:path on {shape_node}")
                    continue
                if g.value(prop_node, SH.nodeKind) in self.IRI_NODE_KINDS and g.value(prop_node, SH["class"]) is None:
                    # Needs an IRI that is neither in the text nor an entity it can refer to (e.g. a language code list)
                    logger.info(f"Skipping property {path} of {shape_node}: its value must be an IRI that cannot be taken from the text")
                    continue
                properties.append({
                    "path": str(path),
                    "name": self._local_name(prop_node, str(path)),
                    "description": self._value(prop_node, SH.description),
                    "datatype": self._value(prop_node, SH.datatype),
                    "class": self._value(prop_node, SH["class"]),
                    "min_count": self._value(prop_node, SH.minCount),
                    "max_count": self._value(prop_node, SH.maxCount),
                    "order": g.value(prop_node, SH.order),
                    "usage_note": self._value(prop_node, self.usage_note_predicate),
                    "literal": self._expects_literal(prop_node),
                })
            properties.sort(key=lambda p: (p["order"] is None, float(p["order"] or 0), p["name"]))
            for j, prop in enumerate(properties, start=1):
                prop["key"] = f"{key}.P{j}"

            definitions.append({
                "key": key,
                "uri": str(shape_node),
                "target_class": target_class,
                "name": self._local_name(shape_node, target_class or str(shape_node)),
                "description": self._value(shape_node, SH.description) or self._value(shape_node, RDFS.comment),
                "usage_note": self._value(shape_node, self.usage_note_predicate),
                "properties": properties,
            })
        return definitions

    # Examples: the shape's usage notes refer to element ids in example HTML documents
    def _load_examples(self, examples: str | List[str] | None) -> None:
        """Attach an example to each node and property shape whose usage note matches an element id."""
        if not examples:
            return
        soups = [BeautifulSoup(html, "html.parser") for html in ([examples] if isinstance(examples, str) else examples)]

        def element_for(usage_note: Optional[str]):
            if not usage_note:
                return None
            return next((el for soup in soups for el in [soup.find(id=usage_note)] if el is not None), None)

        for definition in self.definitions:
            occurrence = element_for(definition["usage_note"])
            for prop in definition["properties"]:
                element = element_for(prop["usage_note"])
                if element is None:
                    continue
                prop["example"] = self._example_value(element)
                # A string in a content attribute (e.g. an article's title) can still be written from the text;
                # only typed values (e.g. dates) and IRIs are hidden metadata the text does not contain as such
                if prop["example"]["source"] == "content" and prop["datatype"] in self.STRING_DATATYPES:
                    prop["example"]["hidden"] = False
                # A relation's example is the text of the occurrence it refers to, given as a ref
                if prop["class"]:
                    prop["example"]["ref"] = self._structure_of(prop["class"])
                # Without a usage note of its own, the occurrence is the closest typed element around its properties
                if occurrence is None:
                    occurrence = element.find_parent(attrs={"typeof": True})
            if occurrence is not None:
                definition["example"] = self._annotated_example(occurrence, definition)

    def _annotated_example(self, occurrence, definition: Dict[str, Any]) -> Dict[str, Any]:
        """
        The visible text of an occurrence's example, with the parts that are examples of its properties marked
        [key] ... [/key]: it shows how an occurrence is divided into its properties, and what belongs to none.
        """
        copy = BeautifulSoup(str(occurrence), "html.parser")
        for prop in definition["properties"]:
            if not prop["usage_note"] or prop.get("example", {}).get("source") != "text":
                continue
            element = copy.find(id=prop["usage_note"])
            if element is None:
                continue
            ref = prop["example"].get("ref")
            # Every value of the property in the occurrence is marked, e.g. each article of a decision
            for part in self._same_property(copy, element):
                part.insert_before(f" [{prop['key']}{f' → {ref}' if ref else ''}] ")
                part.insert_after(f" [/{prop['key']}] ")
        value = " ".join(copy.get_text(" ").split())
        if len(value) > self.OCCURRENCE_EXAMPLE_MAX_LENGTH:
            value = value[:self.OCCURRENCE_EXAMPLE_MAX_LENGTH].rstrip() + "…"
        return {"value": value, "hidden": False, "source": "text", "annotated": True}

    def _same_property(self, root, element) -> List[Any]:
        """The elements in root with the element's RDFa property on the same resource, not on a resource inside it."""
        if not element.get("property"):
            return [element]
        owner = element.find_parent(attrs={"typeof": True})
        return [e for e in root.find_all(attrs={"property": element["property"]})
                if e.find_parent(attrs={"typeof": True}) is owner]

    def _structure_of(self, class_uri: str) -> Optional[str]:
        """Key of the selected structure whose instances are instances of the class, e.g. "S2" for besluit:Artikel."""
        return next((d["key"] for d in self.definitions
                     if d["target_class"] and class_uri in self._superclasses(d["target_class"])), None)

    def _example_value(self, element) -> Dict[str, Any]:
        """Visible text of an example element, or its RDFa content/resource when the value is hidden metadata."""
        visible = " ".join(element.get_text(" ").split())
        if element.get("content"):
            value, hidden, source = " ".join(element["content"].split()), True, "content"
        elif element.get("resource") and not visible:
            value, hidden, source = f"<{element['resource']}>", True, "resource"
        else:
            value, hidden, source = visible, False, "text"
        if len(value) > self.EXAMPLE_MAX_LENGTH:
            value = value[:self.EXAMPLE_MAX_LENGTH].rstrip() + "…"
        return {"value": value, "hidden": hidden, "source": source}

    def _render_example(self, example: Optional[Dict[str, Any]], indent: str) -> List[str]:
        if not example or not example["value"]:
            return []
        label = ("Example, with its properties marked [key] ... [/key]" if example.get("annotated") else
                 "Example (hidden metadata, not visible text)" if example["hidden"] else
                 f"Example of the {example['ref']} occurrence it refers to (give its \"ref\")" if example.get("ref") else
                 "Example")
        return [f"{indent}{label}: {example['value']}"]

    def _render_shape_definitions(self) -> str:
        """Describe the node shapes and their properties as plain text for the prompt."""
        lines = []
        for definition in self.definitions:
            header = f"[{definition['key']}] {definition['name']}"
            if definition["target_class"]:
                header += f" (instance of <{definition['target_class']}>)"
            lines.append(header)
            if definition["description"]:
                lines.append(f"  Description: {definition['description']}")
            lines += self._render_example(definition.get("example"), "  ")
            if definition["properties"]:
                lines.append("  Properties:")
            for prop in definition["properties"]:
                line = f"    [{prop['key']}] {prop['name']}"
                if prop["description"]:
                    line += f": {prop['description']}"
                details = [f"path <{prop['path']}>"]
                if prop["datatype"]:
                    details.append(f"datatype <{prop['datatype']}>")
                if prop["class"]:
                    details.append(f"relation to an instance of <{prop['class']}>, give a ref")
                if prop["min_count"] is not None:
                    details.append(f"min {prop['min_count']}")
                if prop["max_count"] is not None:
                    details.append(f"max {prop['max_count']}")
                lines.append(f"{line} ({', '.join(details)})")
                lines += self._render_example(prop.get("example"), "        ")
            lines.append("")
        return "\n".join(lines)

    # Existing entities related to the document: the inputs of the task (see _is_input; shape config: text_predicate)
    RELATED_ENTITIES_QUERY = """
    PREFIX task: <http://redpencil.data.gift/vocabularies/tasks/>
    SELECT DISTINCT ?input ?type ?content WHERE {
      GRAPH ?taskGraph { $task task:inputContainer ?container . }
      GRAPH ?containerGraph { ?container task:hasResource ?input . }
      OPTIONAL { GRAPH ?typeGraph { ?input a ?type . } }
      OPTIONAL { GRAPH ?contentGraph { ?input $text_predicate ?content . } }
    }
    """

    def fetch_related_entities(self, task_uri: str, text: Optional[str] = None) -> List[str]:
        """
        URIs of the inputs in the task's input container: resources whose types fit one of the selected shapes'
        target classes, through the class hierarchy (see _is_input). With text, only the input whose text_predicate
        value is that text counts (the first by URI when several are, e.g. duplicate
        harvests of one document); when none is, e.g. for a translation, all inputs count.
        """
        from helpers import query
        from escape_helpers import sparql_escape_uri

        q = Template(self.RELATED_ENTITIES_QUERY).substitute(
            task=sparql_escape_uri(task_uri),
            text_predicate=sparql_escape_uri(str(self.text_predicate)))
        types: Dict[str, set[str]] = {}
        contents: Dict[str, List[str]] = {}
        for binding in query(q, sudo=True).get("results", {}).get("bindings", []):
            resource = binding["input"]["value"]
            types.setdefault(resource, set())
            if "type" in binding:
                types[resource].add(binding["type"]["value"])
            if "content" in binding:
                contents.setdefault(resource, []).append(binding["content"]["value"])
        inputs = [resource for resource in types if self._is_input(types[resource])]
        if not inputs:
            logger.warning(f"No input for the shapes found in the input container of task {task_uri}")
            return []

        if text is not None:
            holding = sorted(i for i in inputs if any(self._same_text(c, text) for c in contents.get(i, [])))
            if len(holding) > 1:
                logger.warning(f"{len(holding)} inputs of task {task_uri} hold the text, using the first: {holding}")
            if holding:
                inputs = holding[:1]
        return inputs

    # Originals: the same decision already in the triplestore, e.g. harvested from another publication
    def find_similar_entities(self, title: str, class_uri: str, exclude: set[str]) -> List[Dict[str, Any]]:
        """
        Entities of the class whose same_as_predicate value is (almost) the given title: candidates from the
        triplestore's full-text index (Virtuoso bif:contains on the title's longest words), kept when their
        similarity is at least title_similarity.
        """
        from helpers import query
        from escape_helpers import sparql_escape_uri

        words = sorted(dict.fromkeys(re.findall(r"[^\W\d_]{4,}", title.lower())), key=len, reverse=True)[:6]
        if not words or not class_uri:
            return []
        q = f"""
        SELECT DISTINCT ?entity ?title WHERE {{
          GRAPH ?titleGraph {{
            ?entity {sparql_escape_uri(str(self.same_as_predicate))} ?title .
            ?title bif:contains '{" AND ".join(f'"{word}"' for word in words)}' .
          }}
          GRAPH ?typeGraph {{ ?entity a {sparql_escape_uri(class_uri)} . }}
        }} LIMIT 50
        """
        try:
            bindings = query(q, sudo=True).get("results", {}).get("bindings", [])
        except Exception as e:
            logger.warning(f"Could not search for entities with title {title!r}: {e}")
            return []
        matches = []
        for binding in bindings:
            uri, candidate = binding["entity"]["value"], binding["title"]["value"]
            score = self._title_similarity(title, candidate)
            if uri not in exclude and score >= self.TITLE_SIMILARITY and uri not in {m["uri"] for m in matches}:
                matches.append({"uri": uri, "title": candidate, "similarity": round(score, 3)})
        return sorted(matches, key=lambda m: -m["similarity"])

    def _title_similarity(self, a: str, b: str) -> float:
        normalize = lambda t: " ".join(re.findall(r"\w+", t.lower()))
        return difflib.SequenceMatcher(None, normalize(a), normalize(b)).ratio()

    def _link_originals(self, result: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        For each segmented entity of the same_as_class, find originals in the triplestore with (almost) the
        same title, and link each with an oa:Annotation: oa:hasTarget the segmented entity, oa:hasBody the original,
        oa:motivatedBy oa:linking, nif:confidence the title similarity. Without an original there is no annotation.
        Being the same decision, the segmented entity does not get what an original already defines (e.g. its
        description). The Turtle is rebuilt and validated again; result["originals"] lists the links.
        """
        result["originals"] = []
        if not self.same_as_class:
            return result
        segments = result["segments"]
        known = {s["uri"] for s in segments} | {e["uri"] for e in context["entities"].values()}
        for segment in segments:
            if self.target_classes.get(segment["shape"]) != self.same_as_class:
                continue
            title = next((p["text"] for p in segment["properties"]
                          if p["path"] == str(self.same_as_predicate) and "text" in p), None)
            if title is None and segment["existing"]:
                title = context["graph"].value(URIRef(segment["uri"]), self.same_as_predicate)
            if not title:
                continue
            for match in self.find_similar_entities(str(title), self.same_as_class, known):
                annotation_uuid = str(uuid.uuid4())
                result["originals"].append({**match, "of": segment["key"], "target": segment["uri"],
                                            "annotation": self.ANNOTATION_TEMPLATE.format(uuid=annotation_uuid),
                                            "annotation_uuid": annotation_uuid})
                known.add(match["uri"])
                defined = self._defined_paths(self.fetch_entities([match["uri"]]), URIRef(match["uri"]), segment["shape"])
                dropped = sorted({p["path"] for p in segment["properties"] if p["path"] in defined})
                if dropped:
                    logger.info(f"Not adding {dropped} to {segment['key']}: original {match['uri']} defines them")
                segment["properties"] = [p for p in segment["properties"] if p["path"] not in defined]
                logger.info(f"Linking {segment['key']} to original {match['uri']} (title similarity {match['similarity']})")
        if result["originals"]:
            result["turtle"] = self.to_turtle(segments, result["originals"])
            result["conforms"], result["violations"] = self.validate(
                result["turtle"], context["graph"], {s["uri"]: s["key"] for s in segments})
        return result

    def fetch_entities(self, uris: List[str]) -> Graph:
        """Fetch all triples of the given entities, from any graph, from the SPARQL endpoint."""
        from helpers import query
        from escape_helpers import sparql_escape_uri

        graph = Graph()
        if not uris:
            return graph
        q = ("SELECT DISTINCT ?s ?p ?o WHERE { VALUES ?s { "
             + " ".join(sparql_escape_uri(uri) for uri in uris)
             + " } GRAPH ?g { ?s ?p ?o } }")
        for binding in query(q, sudo=True).get("results", {}).get("bindings", []):
            graph.add((self._term(binding["s"]), self._term(binding["p"]), self._term(binding["o"])))
        return graph

    def _term(self, binding: Dict[str, str]):
        """rdflib term from a SPARQL JSON result binding."""
        if binding["type"] == "uri":
            return URIRef(binding["value"])
        if binding["type"] == "bnode":
            return BNode(binding["value"])
        return Literal(binding["value"], lang=binding.get("xml:lang"),
                       datatype=URIRef(binding["datatype"]) if binding.get("datatype") else None)

    def _build_context(self, task: Any, text: Optional[str] = None) -> Dict[str, Any]:
        """
        Existing entities related to the task (keys E1, E2, ...) with their types, known properties and the shapes
        they can play.
        """
        task_uri = getattr(task, "task_uri", None)
        uris = self.fetch_related_entities(task_uri, text) if task_uri else []
        graph = self.fetch_entities(uris)

        entities = {}
        for i, uri in enumerate(uris, start=1):
            subject = URIRef(uri)
            types = {str(t) for t in graph.objects(subject, RDF.type)}
            entities[f"E{i}"] = {
                "uri": uri,
                "types": types,
                "roles": [d["key"] for d in self.definitions if self._can_play(types, d["target_class"])],
            }
            if not types:
                logger.warning(f"Related entity {uri} has no rdf:type in the triplestore")
        return {"graph": graph, "entities": entities}

    def _detect_language(self, text: str) -> Optional[str]:
        if self.LANGUAGE:
            return self.LANGUAGE
        try:
            from langdetect import DetectorFactory, detect
            DetectorFactory.seed = 0
            return detect(text[:self.TEXT_LIMIT])
        except Exception as e:
            logger.warning(f"Could not detect the language of the text: {e}")
            return None

    def _same_text(self, value: str, text: str) -> bool:
        """Compared on the first 200 characters, ignoring whitespace differences."""
        start = " ".join(text.split())[:200]
        return bool(start) and " ".join(str(value).split())[:200] == start

    def _holds_text(self, graph: Graph, subject, text: str) -> bool:
        """The entity has a literal value holding the input text, e.g. an expression's content."""
        return any(isinstance(o, Literal) and self._same_text(o, text) for o in graph.objects(subject, None))

    def _render_context(self, context: Dict[str, Any], text: str) -> str:
        if not context["entities"]:
            return ""
        document_start = " ".join(text.split())[:200]
        lines = ["", "    EXISTING ENTITIES"]
        for key, entity in context["entities"].items():
            subject = URIRef(entity["uri"])
            lines.append(f"[{key}] <{entity['uri']}>"
                         + (" (its text is the text below)" if self._holds_text(context["graph"], subject, text) else ""))
            lines.append(f"  Types: {', '.join(f'<{t}>' for t in sorted(entity['types'])) or 'unknown'}")
            if entity["roles"]:
                names = {d["key"]: d["name"] for d in self.definitions}
                lines.append(f"  Can be enriched as: {', '.join(f'{role} ({names[role]})' for role in entity['roles'])}")
            known = []
            for p in sorted({p for p in context["graph"].predicates(subject) if p not in (RDF.type, self.MU_UUID)}, key=str):
                values = [f"<{o}>" if isinstance(o, URIRef) else
                          "the text below" if document_start and " ".join(str(o).split())[:200] == document_start else
                          f'"{" ".join(str(o).split())[:60]}"'
                          for o in list(context["graph"].objects(subject, p))[:3]]
                known.append(f"    <{p}>: {', '.join(values)}")
            if known:
                lines.append("  Known values:")
                lines += known
        return "\n".join(lines) + "\n"

    # Locating the LLM's text in the source text
    def _locate(self, text: str, found_text: Any, lo: int, hi: int) -> Optional[tuple[int, int]]:
        """Return (start, end) of found_text within text[lo:hi], ignoring whitespace differences (LLMs tend to collapse newlines)."""
        if not isinstance(found_text, str) or not found_text.split():
            return None
        match = self._pattern(found_text).search(text, lo, hi)
        if match:
            return match.start(), match.end()
        return self._locate_lines(text, found_text, lo, hi)

    def _pattern(self, found_text: str) -> re.Pattern:
        return re.compile(r"\s+".join(self._token_pattern(token) for token in found_text.split()))

    def _locate_lines(self, text: str, found_text: str, lo: int, hi: int) -> Optional[tuple[int, int]]:
        """
        Fallback when found_text is not one contiguous piece of the text: its lines in order, with at most
        max_skipped_chars of left-out text between two lines (e.g. a page footer). The span runs from the first to
        the last line, so the left-out text is part of it.
        """
        lines = [line for line in found_text.split("\n") if line.strip()]
        if len(lines) < 2:
            return None
        patterns = [self._pattern(line) for line in lines]
        for first in patterns[0].finditer(text, lo, hi):
            end = first.end()
            for pattern in patterns[1:]:
                match = pattern.search(text, end, hi)
                if match is None or len("".join(text[end:match.start()].split())) > self.MAX_SKIPPED_CHARS:
                    break
                end = match.end()
            else:
                logger.info(f"Located text with left-out lines: {' '.join(text[first.start():end].split())[:60]!r}...")
                return first.start(), end
        return None

    def _token_pattern(self, token: str) -> str:
        """Exact characters, except that a non-ASCII symbol (e.g. a bullet "•") matches any single symbol."""
        return "".join(r"[^\w\s]" if ord(c) > 127 and not unicodedata.category(c).startswith(("L", "N")) else re.escape(c)
                       for c in token)

    def _locate_from(self, text: str, found_text: Any, cursor: int, lo: int, hi: int) -> Optional[tuple[int, int]]:
        """Locate after the cursor first (document order, repeated phrases), then anywhere in range."""
        found = self._locate(text, found_text, max(cursor, lo), hi)
        if found is None and cursor > lo:
            found = self._locate(text, found_text, lo, hi)
        return found

    def segment_and_validate(self, text: str, task: Any = None) -> Dict[str, Any]:
        """
        Segment, serialize to Turtle and validate against the SHACL shape. While the result does not
        conform or has texts that cannot be found, ask the LLM again with its previous answer and
        the problems, up to max_attempts. Returns the attempt with the fewest problems per entity:
            {'turtle': str, 'segments': [...], 'conforms': bool, 'violations': [...],
             'unlocated': [...], 'attempts': int,
             'usage': {'calls': int, 'input_tokens': int, 'output_tokens': int},  # summed over all attempts
             'originals': [{'uri', 'title', 'similarity', 'of', 'target', 'annotation', 'annotation_uuid'}]}
                                                                    # see _link_originals
        task: the task being processed (with task_uri). The inputs in its input container (see
        fetch_related_entities) are enriched and linked to. A segmented decision is linked with an oa:Annotation to an
        original with (almost) the same title.
        """
        context = self._build_context(task, text)
        context["language"] = self._detect_language(text)
        system_prompt = self.system_prompt + self._render_context(context, text)
        best = None
        feedback = None
        usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0}
        for attempt in range(1, self.MAX_ATTEMPTS + 1):
            answer = self._ask(text, task=task, feedback=feedback, system_prompt=system_prompt)
            tokens = getattr(self.analyzer.last_response, "usage_metadata", None) or {}
            usage["calls"] += 1
            usage["input_tokens"] += tokens.get("input_tokens", 0) or 0
            usage["output_tokens"] += tokens.get("output_tokens", 0) or 0
            unlocated: List[str] = []
            segments = self._segments_from_answer(text, answer, unlocated, context)
            turtle = self.to_turtle(segments)
            try:
                conforms, violations = self.validate(turtle, context["graph"], {s["uri"]: s["key"] for s in segments})
            except Exception as e:
                # The Turtle is generated here, so this points to a bug rather than the LLM's answer
                logger.exception("Generated Turtle could not be parsed")
                conforms, violations = False, [{"entity": None, "path": None, "message": f"Invalid Turtle: {e}"}]

            logger.info(f"Shape validation attempt {attempt}/{self.MAX_ATTEMPTS}: {len(segments)} entities, "
                        f"{'conforms' if conforms else f'{len(violations)} violations'}, {len(unlocated)} problems in the answer")
            current = {"turtle": turtle, "segments": segments, "conforms": conforms,
                       "violations": violations, "unlocated": unlocated}
            # Problems per entity, so that dropping entities never looks like an improvement
            if best is None or self._score(current) < self._score(best):
                best = current
            best["attempts"] = attempt
            best["usage"] = usage
            if conforms and not unlocated:
                break
            feedback = self.FEEDBACK_TEMPLATE.format(
                previous_answer=json.dumps(answer, ensure_ascii=False),
                problems="\n".join([f"- {problem}" for problem in unlocated] +
                                   [f"- {self._describe_violation(v)}" for v in violations]),
            )

        best = self._link_originals(best, context)
        if not best["conforms"] or best["unlocated"]:
            logger.warning(f"Segmentation still has problems after {best['attempts']} attempts, returning the attempt "
                           f"with {len(best['segments'])} entities, {len(best['violations'])} violations "
                           f"and {len(best['unlocated'])} problems in the answer")
        return best

    def _score(self, attempt: Dict[str, Any]) -> tuple[float, int]:
        problems = len(attempt["violations"]) + len(attempt["unlocated"])
        return problems / max(len(attempt["segments"]), 1), -len(attempt["segments"])

    def validate(self, turtle: str, context_graph: Optional[Graph] = None, keys: Optional[Dict[str, str]] = None) -> tuple[bool, List[Dict[str, Any]]]:
        """
        Parse the Turtle (raises when invalid) and validate it, together with the existing entities' triples,
        against the SHACL shape. Cardinalities are ignored, and only violations on what the Turtle adds count: on a
        value the Turtle adds, on a property the Turtle sets for that entity, or on an entity in the Turtle as a
        whole. Problems in the existing data (e.g. other values of an existing entity) cannot be fixed by the LLM.
        """
        output_graph = Graph().parse(data=turtle, format="turtle")
        data_graph = output_graph + context_graph if context_graph is not None else output_graph
        output_subjects = set(output_graph.subjects())
        output_properties = set(output_graph.subject_predicates())
        _, report_graph, _ = shacl_validate(
            data_graph, shacl_graph=self.shape_graph, allow_warnings=True, allow_infos=True)
        violations = []
        for result in report_graph.subjects(RDF.type, SH.ValidationResult):
            if self._value_of(report_graph, result, SH.sourceConstraintComponent) in self.IGNORED_CONSTRAINTS:
                continue
            focus = report_graph.value(result, SH.focusNode)
            path = report_graph.value(result, SH.resultPath)
            value = report_graph.value(result, SH.value)
            if path is not None and value is not None:
                added = (focus, path, value) in output_graph
            elif path is not None:
                added = (focus, path) in output_properties
            else:
                added = focus in output_subjects
            if not added:
                continue
            violations.append({
                "entity": self._describe_entity(output_graph, focus, keys or {}),
                "path": str(path) if path is not None else None,
                "constraint": self._value_of(report_graph, result, SH.sourceConstraintComponent),
                "value": self._value_of(report_graph, result, SH.value),
                "message": self._value_of(report_graph, result, SH.resultMessage),
                "severity": self._value_of(report_graph, result, SH.resultSeverity),
            })
        violations.sort(key=lambda v: (v["entity"] or "", v["path"] or "", v["message"] or ""))
        return not violations, violations

    def _value_of(self, graph: Graph, subject, predicate) -> Optional[str]:
        value = graph.value(subject, predicate)
        return str(value) if value is not None else None

    def _describe_entity(self, data_graph: Graph, focus, keys: Dict[str, str]) -> Optional[str]:
        """Describe a focus node by its key, structure and property values, as the LLM knows them."""
        if focus is None:
            return None
        definition = next((self.definitions_by_class[str(t)] for t in data_graph.objects(focus, RDF.type)
                           if str(t) in self.definitions_by_class), None)
        key = keys.get(str(focus))
        if definition is None:
            return f"{key} <{focus}>" if key else str(focus)
        prop_keys = {prop["path"]: prop["key"] for prop in definition["properties"]}
        values = [f'{prop_keys.get(str(p), p)}="{" ".join(str(o).split())[:60]}"'
                  for p, o in data_graph.predicate_objects(focus)
                  if p not in (RDF.type, self.MU_UUID)]
        return (f"{key + ' ' if key else ''}{definition['key']} ({definition['name']})"
                + (f" with {', '.join(sorted(values))}" if values else ""))

    def _describe_violation(self, violation: Dict[str, Any]) -> str:
        """One line for the LLM, in terms of the structure and property keys from the prompt."""
        props = {prop["path"]: prop for definition in self.definitions for prop in definition["properties"]}
        line = violation["entity"] or "document"
        if violation["path"]:
            prop = props.get(violation["path"])
            line += f", property {prop['key']} ({prop['name']})" if prop else f", property <{violation['path']}>"
        if violation.get("constraint"):
            line += f": violates {self._local_name(URIRef(violation['constraint']), violation['constraint']).removesuffix('ConstraintComponent')}"
            if violation.get("value"):
                line += f' for value "{" ".join(violation["value"].split())[:60]}"'
        else:
            line += f": {violation['message'] or 'violates the shape'}"
        return line

    def _mint_uri(self, shape_uri: str) -> tuple[str, str]:
        """New (uri, uuid) for an entity of the given node shape."""
        entity_uuid = str(uuid.uuid4())
        class_uri = self.target_classes.get(shape_uri) or shape_uri
        class_name = re.split(r"[#/]", class_uri.rstrip("#/"))[-1].lower()
        return self.URI_TEMPLATE.format(class_name=class_name, uuid=entity_uuid), entity_uuid

    def to_turtle(self, segments: List[Dict[str, Any]], originals: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Serialize the new triples: new entities, added types and properties of existing entities, relations, and the
        annotations linking segmented entities to their originals.
        """
        blocks = []
        for segment in segments:
            if segment["existing"]:
                statements = [f"a {URIRef(segment['add_type']).n3()}"] if segment["add_type"] else []
            else:
                target_class = self.target_classes.get(segment["shape"])
                statements = [f"a {URIRef(target_class).n3()}"] if target_class else []
                statements.append(f"{self.MU_UUID.n3()} {Literal(segment['uuid']).n3()}")
            for prop in segment["properties"]:
                value = URIRef(prop["ref"]) if "ref" in prop else Literal(prop["text"], lang=prop.get("lang"))
                statements.append(f"{URIRef(prop['path']).n3()} {value.n3()}")
            if statements:
                blocks.append(f"{URIRef(segment['uri']).n3()} " + " ;\n    ".join(statements) + " .")
        for original in originals or []:
            statements = [
                f"a {self.OA['Annotation'].n3()}",
                f"{self.MU_UUID.n3()} {Literal(original['annotation_uuid']).n3()}",
                f"{self.OA['hasTarget'].n3()} {URIRef(original['target']).n3()}",
                f"{self.OA['hasBody'].n3()} {URIRef(original['uri']).n3()}",
                f"{self.OA['motivatedBy'].n3()} {self.OA['linking'].n3()}",
                f"{self.NIF_CONFIDENCE.n3()} {Literal(original['similarity']).n3()}",
            ]
            blocks.append(f"{URIRef(original['annotation']).n3()} " + " ;\n    ".join(statements) + " .")
        return "\n\n".join(blocks) + ("\n" if blocks else "")

    def _ask(self, text: str, task: Any = None, feedback: Optional[str] = None, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        logger.info(f"Running shape-driven LLM segmentation with {self.analyzer.model_name}...")

        # Attach the current task so the analyzer logs the AI call (see record_llm_call).
        self.analyzer._task = task

        try:
            result = self.analyzer.analyze_single_entry(
                text=text,
                system_prompt=system_prompt or self.system_prompt,
                # The template is str.format()-ed with the text, so braces in the feedback are escaped
                user_prompt_template=self.USER_PROMPT_TEMPLATE + (
                    feedback.replace("{", "{{").replace("}", "}}") if feedback else ""),
                expected_schema=self.RESULTS_SCHEMA,
                text_limit=self.TEXT_LIMIT,
            )
        except Exception as e:
            logger.exception("Shape-driven LLM segmentation failed")
            raise RuntimeError(
                f"Shape-driven LLM segmentation failed ({self.analyzer.model_name}): {e}"
            ) from e

        return result

    def _merge_into_text_holder(self, segments: List[Dict[str, Any]], text: str,
                                context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        An existing entity holding the whole text is the one occurrence of a structure it can be enriched as
        (e.g. an uittreksel's expression is its one decision). When there is exactly one new occurrence of such a
        structure, that occurrence is the entity: it becomes the entity, or, when the answer also enriches the entity
        as that structure, its properties are added to that. Several new occurrences (e.g. the decisions in notulen)
        stay new. Runs after _merge_sub_parts, so that an occurrence the LLM split in parts counts once.
        """
        holder = next((key for key, entity in context["entities"].items()
                       if self._holds_text(context["graph"], URIRef(entity["uri"]), text)), None)
        if holder is None:
            return segments
        entity = context["entities"][holder]
        enriched = [s for s in segments if s["existing"] and s["key"] == holder]
        for role in entity["roles"]:
            shape = self.key_lookup[role]["shape"]
            new = [s for s in segments if not s["existing"] and s["shape"] == shape]
            if len(new) != 1:
                continue
            occurrence = new[0]
            target = next((s for s in enriched if s["shape"] == shape), None)
            if target is None and enriched:
                continue  # the answer enriches the entity as another structure
            if target is None:
                target_class = self.target_classes.get(shape)
                target = {"key": holder, "shape": shape, "uri": entity["uri"], "uuid": None, "existing": True,
                          "text": None, "start": None, "end": None,
                          "add_type": target_class if target_class and target_class not in entity["types"] else None,
                          "properties": []}
                segments.append(target)
            # As for an enriched entity, only the properties it does not have yet
            existing_paths = self._defined_paths(context["graph"], URIRef(entity["uri"]), shape)
            target["properties"] += [p for p in occurrence["properties"] if p["path"] not in existing_paths]
            logger.info(f"Merged the only new {role} occurrence {occurrence['key']} into {holder}, which holds the text")
            return self._redirect(segments, {occurrence["uri"]: target})
        return segments

    # A heading line: at most this many words, ending with a colon, e.g. "BESLIST:" between a motivation and the
    # decision itself; a longer line ending with a colon introduces what follows, e.g. "... worden door:"
    HEADING_MAX_WORDS = 4

    def _is_heading_line(self, line: str) -> bool:
        line = line.strip()
        return line.endswith(":") and ":" not in line[:-1] and 1 <= len(line.split()) <= self.HEADING_MAX_WORDS

    def _without_heading_lines(self, source: str, start: int, end: int) -> tuple[int, int]:
        """
        A property value does not start or end with a heading line: such a heading introduces or follows the value,
        and belongs to no property. A value that is a single line is kept as it is.
        """
        while True:
            text = source[start:end]
            first_end = text.find("\n")
            last_start = text.rfind("\n") + 1
            if first_end > 0 and self._is_heading_line(text[:first_end]):
                start += first_end
            elif 0 < last_start < len(text) and self._is_heading_line(text[last_start:]):
                end = start + last_start
            else:
                return start, end
            while start < end and source[start].isspace():
                start += 1
            while end > start and source[end - 1].isspace():
                end -= 1

    _NUMBERED_LABEL = re.compile(r"\s*(?:([^\W\d_]+)\s+)?(\d+(?:\.\d+)*)\b")

    def _numbered_label(self, text: Optional[str]) -> Optional[tuple[str, str]]:
        """("Artikel", "1.1") for a text starting with "Artikel 1.1. ...", None for a text without a leading number."""
        match = self._NUMBERED_LABEL.match(text or "")
        return ((match.group(1) or "").lower(), match.group(2)) if match else None

    def _merge_sub_parts(self, segments: List[Dict[str, Any]], source: str) -> List[Dict[str, Any]]:
        """
        Within one structure, a numbered sub-part (e.g. "Artikel 1.1" after "Artikel 1") is part of the occurrence
        it belongs to, not an occurrence of its own: the parent is extended over it, and relations to the sub-part
        refer to the parent. So is an occurrence without a number of its own that follows a numbered one: it continues
        it (e.g. the rest of an article after a page break, or a decision's content after its motivation). A parent
        property value that ran to the end of the parent (e.g. its content) is extended as well. The sub-part's
        relations, and its values of properties the parent has no value of, are the parent's.
        """
        merged: Dict[str, Dict[str, Any]] = {}  # sub-part URI -> parent
        new = sorted((s for s in segments if not s["existing"]), key=lambda s: s["start"])
        parents: Dict[str, Dict[str, Any]] = {}  # shape -> current parent
        for segment in new:
            label = self._numbered_label(segment["text"])
            parent = parents.get(segment["shape"])
            parent_label = self._numbered_label(parent["text"]) if parent else None
            is_sub_part = label and parent_label and label[0] == parent_label[0] and label[1].startswith(parent_label[1] + ".")
            is_continuation = label is None and parent_label is not None
            if is_sub_part or is_continuation:
                old_end = parent["end"]
                parent["end"] = max(parent["end"], segment["end"])
                parent["text"] = source[parent["start"]:parent["end"]]
                for prop in parent["properties"]:
                    if "ref" not in prop and prop["end"] == old_end:
                        # e.g. a content that was only the heading "BESLIST:" now runs on without it
                        prop["start"], prop["end"] = self._without_heading_lines(source, prop["start"], parent["end"])
                        prop["text"] = source[prop["start"]:prop["end"]]
                values = {prop["path"] for prop in parent["properties"] if "ref" not in prop}
                parent["properties"] += [prop for prop in segment["properties"]
                                         if "ref" in prop or prop["path"] not in values]
                parent["properties"].sort(key=self._property_order)
                merged[segment["uri"]] = parent
                logger.info(f"Merged {'sub-part' if is_sub_part else 'continuation'} {segment['key']}"
                            f"{f' ({label[1]})' if label else ''} into {parent['key']} ({parent_label[1]})")
            else:
                parents[segment["shape"]] = segment
        return self._redirect(segments, merged) if merged else segments

    @staticmethod
    def _property_order(prop: Dict[str, Any]) -> tuple:
        """Values in document order, relations last."""
        return prop.get("start") is None, prop.get("start") or 0, -(prop.get("end") or 0)

    def _redirect(self, segments: List[Dict[str, Any]], merged: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Leave out the merged entities (URI -> the entity they were merged into) and redirect relations to them."""
        kept = [s for s in segments if s["uri"] not in merged]
        for segment in kept:
            properties, seen = [], set()
            for prop in segment["properties"]:
                if "ref" in prop:
                    target = merged.get(prop["ref"])
                    prop = {**prop, "ref": target["uri"]} if target else prop
                    if prop["ref"] == segment["uri"] or (prop["path"], prop["ref"]) in seen:
                        continue
                    seen.add((prop["path"], prop["ref"]))
                properties.append(prop)
            segment["properties"] = properties
        return kept

    def _unlinked(self, segments: List[Dict[str, Any]]) -> List[str]:
        """New entities that another returned entity could refer to through a relation property, but none does."""
        referenced = {prop["ref"] for segment in segments for prop in segment["properties"] if "ref" in prop}
        definitions = {d["uri"]: d for d in self.definitions}
        problems = []
        for segment in segments:
            if segment["existing"] or segment["uri"] in referenced:
                continue
            classes = self._superclasses(self.target_classes.get(segment["shape"]) or "")
            candidates = [f"{owner['key']} via {prop['key']} ({prop['name']})"
                          for owner in segments if owner is not segment
                          for prop in definitions[owner["shape"]]["properties"] if prop["class"] in classes]
            if candidates:
                problems.append(f"{segment['key']} is not referred to by any relation; it could be referred to by "
                                f"{', '.join(candidates)}")
        return problems

    def _segments_from_answer(self, text: str, result: Dict[str, Any], problems: Optional[List[str]] = None,
                              context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Build entities from the LLM's answer: locate the texts of new occurrences in the source text, resolve
        existing entities and relation refs. Anything that cannot be resolved is added to problems.
        """
        problems = problems if problems is not None else []
        context = context or {"graph": Graph(), "entities": {}}
        entities = context["entities"]
        # Offsets must refer to exactly the text the LLM saw
        source = text[:self.TEXT_LIMIT]
        segments: List[Dict[str, Any]] = []
        by_key: Dict[str, Dict[str, Any]] = {}
        items = [dict(item) for item in result.get("segments", []) if isinstance(item, dict)]
        cursors: Dict[str, int] = {}

        # First pass: the entities themselves, so that relations can refer to any of them
        resolved_items = []
        for item in items:
            shape_key = str(item.get("shape", "")).strip()
            target = self.key_lookup.get(shape_key)
            if target is None or target["path"] is not None:
                logger.warning(f"Skipping segment with unknown structure key {shape_key!r}")
                problems.append(f"unknown structure key {shape_key!r}")
                continue

            entity_key = str(item.get("entity") or "").strip()
            if entity_key:
                entity = entities.get(entity_key)
                if entity is None:
                    problems.append(f"unknown existing entity {entity_key!r}")
                    continue
                if shape_key not in entity["roles"]:
                    problems.append(f"{entity_key} cannot be enriched as {shape_key}; "
                                    f"it can be enriched as: {', '.join(entity['roles']) or 'nothing'}")
                    continue
                segment = by_key.get(entity_key)
                if segment is None:
                    target_class = self.target_classes.get(target["shape"])
                    segment = {"key": entity_key, "shape": target["shape"], "uri": entity["uri"], "uuid": None,
                               "existing": True, "text": None, "start": None, "end": None,
                               "add_type": target_class if target_class and target_class not in entity["types"] else None,
                               "properties": []}
                    by_key[entity_key] = segment
                    segments.append(segment)
                resolved_items.append((item, segment, shape_key))
                continue

            found = self._locate_from(source, item.get("text"), cursors.get(shape_key, 0), 0, len(source))
            if found is None:
                logger.warning(f"Could not locate {shape_key} in source: {str(item.get('text'))[:80]!r}")
                problems.append(f"{shape_key}: text not found verbatim in the document: {item.get('text')!r}")
                continue
            start, end = found
            cursors[shape_key] = end
            uri, entity_uuid = self._mint_uri(target["shape"])
            occurrence_key = str(item.get("id") or "").strip() or f"N?{len(segments) + 1}"
            if occurrence_key in by_key or occurrence_key in entities:
                problems.append(f"id {occurrence_key!r} is used more than once")
                occurrence_key = f"{occurrence_key}?{len(segments) + 1}"
            segment = {"key": occurrence_key, "shape": target["shape"], "uri": uri, "uuid": entity_uuid,
                       "existing": False, "text": source[start:end], "start": start, "end": end,
                       "add_type": None, "properties": []}
            by_key[occurrence_key] = segment
            segments.append(segment)
            resolved_items.append((item, segment, shape_key))

        # Second pass: property values and relations
        for item, segment, shape_key in resolved_items:
            existing_paths = (self._defined_paths(context["graph"], URIRef(segment["uri"]), segment["shape"])
                              if segment["existing"] else set())
            # Values of existing entities can be anywhere in the document, new ones only inside their occurrence
            lo, hi = (0, len(source)) if segment["existing"] else (segment["start"], segment["end"])
            property_cursors: Dict[str, int] = {}
            properties = item.get("properties") or []
            for prop in properties if isinstance(properties, list) else []:
                if not isinstance(prop, dict):
                    continue
                prop_key = str(prop.get("property", "")).strip()
                prop_target = self.key_lookup.get(prop_key)
                if prop_target is None or prop_target["shape"] != segment["shape"] or prop_target["path"] is None:
                    logger.warning(f"Skipping property with unknown key {prop_key!r} for {segment['key']}")
                    problems.append(f"{segment['key']}: unknown property key {prop_key!r} for {shape_key}")
                    continue
                if prop_target["path"] in existing_paths:
                    logger.info(f"Skipping {prop_key} of {segment['key']}: the existing entity already has a value")
                    continue

                if self.properties_by_key[prop_key]["class"]:
                    ref = str(prop.get("ref") or "").strip()
                    ref_uri = by_key[ref]["uri"] if ref in by_key else entities[ref]["uri"] if ref in entities else None
                    if ref_uri is None:
                        problems.append(f"{segment['key']}, property {prop_key}: relation needs a \"ref\" to an "
                                        f"occurrence id or existing entity key, got {prop.get('ref') or prop.get('text')!r}")
                        continue
                    segment["properties"].append({"path": prop_target["path"], "ref": ref_uri})
                    continue

                prop_found = self._locate_from(source, prop.get("text"), property_cursors.get(prop_key, lo), lo, hi)
                if prop_found is None:
                    where = "in the document" if segment["existing"] else \
                        f"inside {segment['key']} starting with {source[lo:lo + 60]!r}"
                    logger.warning(f"Could not locate {prop_key} of {segment['key']}: {str(prop.get('text'))[:80]!r}")
                    problems.append(f"{segment['key']}, property {prop_key}: text not found verbatim {where}: {prop.get('text')!r}")
                    continue
                prop_start, prop_end = self._without_heading_lines(source, *prop_found)
                property_cursors[prop_key] = prop_end
                segment["properties"].append({
                    "path": prop_target["path"],
                    "text": source[prop_start:prop_end],
                    "start": prop_start,
                    "end": prop_end,
                    "lang": context.get("language") if self.properties_by_key[prop_key]["datatype"] == str(RDF.langString) else None,
                })

            segment["properties"].sort(key=self._property_order)

        segments = self._merge_sub_parts(segments, source)
        segments = self._merge_into_text_holder(segments, text, context)
        problems += self._unlinked(segments)

        # Existing entities first, then new occurrences in document order
        segments.sort(key=lambda s: (not s["existing"], s["start"] or 0, -(s["end"] or 0)))
        logger.info(f"Located {len(segments)} shape segments")
        return segments
