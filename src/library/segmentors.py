import re
import json
import uuid
import textwrap
import unicodedata
from pathlib import Path
import requests
from typing import Optional, Any, List, Dict
from helpers import logger
from rdflib import Graph, URIRef, Literal, BNode
from rdflib.namespace import SH, RDF, RDFS
from rdflib.collection import Collection
from pyshacl import validate as shacl_validate
from bs4 import BeautifulSoup

from .LLMAnalyzer import LLMAnalyzer


# Shape-driven LLM segmentor: what to look for is read from a SHACL shape, the LLM only returns the lines each
# finding runs from and to with the words at their start and end, and the character offsets are resolved
# deterministically in the source text.
class LLMSegmentorWithShape:
    """
    Segmentor using LLMAnalyzer, driven by a SHACL shape.

    Every selected sh:NodeShape (and its sh:property shapes) is described to the LLM, together with the existing
    entities related to the target (see the target's related_entities). Every occurrence the LLM finds in the text is
    a new entity, linked to the other new entities, and to the existing ones, through the shape's own properties. The
    existing entities themselves are never changed. The document is presented numbered line by line: the LLM answers
    with, per occurrence and property value, the 1-based lines the span runs from and to with a few verbatim words at
    its start and end, which segment_and_validate() resolves to exact character offsets in the text.

    segment_and_validate() returns, besides the validation, the Turtle of the new entities:
        <new entity> a <sh:targetClass> ;
            mu:uuid "{uuid}" ;
            <sh:path> "value found in the text" ;
            <relation sh:path> <other new entity or existing entity> .
    A new entity's URI is URI_TEMPLATE filled with a new UUID and {class_name}, the lowercased local name of the
    shape's sh:targetClass: e.g. http://data.lblod.info/id/artikel/{uuid} for a besluit:Artikel.
    and the same entities with their character offsets in the text ('segments'):
        {'key': 'N1', 'shape': <node shape URI>, 'uri': str, 'uuid': str, 'text': str, 'start': int, 'end': int,
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
    "E1", "E2", ...): entities that are already known, which occurrences can refer to.

    TASK
    - Find every occurrence of each structure in the document.
    - For each occurrence, also find the value of each of its properties: inside
      that occurrence, or elsewhere in the document when it is only there (e.g.
      a location in a page footer).
    - A value belongs to the occurrence it describes. A value, such as a date,
      of something that is not one of the STRUCTURES (e.g. the document itself,
      like the date of a certified copy) is left out: never put it on an
      occurrence instead.
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
    - Existing entities are never occurrences: do not return them, and do not
      give them properties.
    - An occurrence's relation can refer to an existing entity of the class the
      relation needs: give its key ("E...") in "ref".

    RELATIONS
    - A property marked as a relation refers to another occurrence or existing
      entity: give its key in "ref" ("N..." id of an occurrence or "E..." key of
      an existing entity) instead of a span.
    - The relation is given on the occurrence that has the property.
    - A ref to an existing entity also gets the span where the document
      mentions it (e.g. a person's name as written), see SPANS. Go through the
      document and give a ref for every mention of an existing entity.
    - Always give every relation the document supports from the occurrences you
      return, to each other and to existing entities.
    - Leave a relation out when what it refers to is neither an occurrence of
      one of the STRUCTURES nor one of the EXISTING ENTITIES (e.g. a document
      that is not in scope): never refer to another occurrence, or to the
      occurrence itself, instead.

    SPANS
    - The document is presented with a number on every line, e.g. "  15| ".
      Give every span (an occurrence, a property value or a mention) as
      "start" and "end": the numbers of the lines it runs from and to, both
      included.
    - Every span also has "first" and "last": a few words copied verbatim from
      the document at the start and at the end of the span, without the line
      numbers; for a one-word span they are the same word. Copy them character
      for character: no paraphrasing, translation, summarizing, normalization
      or added punctuation.
    - An occurrence includes all of its content: every line, list, bullet point
      and continuation (also across page breaks), up to where the next part of
      the document starts, such as the next occurrence. Do not stop after the
      first sentence or line.
<TARGET RULES>
    - Occurrences of different structures are always returned separately, also
      when one contains the other.
    - A span covers everything between its first and its last line, including
      page headers, footers and page numbers that interrupt the content: those
      lines belong to the span.
    - A property with a datatype other than a string (e.g. a date, a date and
      time, or a number) also gets "value": the value in the form of that
      datatype, e.g. 2026-06-22 for a date and 2026-06-22T19:00:00 for a date and
      time. Its span is still the text the value is read from.

    OUTPUT CONTRACT
    Respond with STRICT JSON only, no markdown fences or commentary, exactly:
    {"segments": [
      {"id": "N1", "shape": "<structure key>", "start": 12, "end": 40,
       "first": "...", "last": "...",
       "properties": [{"property": "<property key>", "start": 12, "end": 13,
                       "first": "...", "last": "..."},
                      {"property": "<typed property key>", "start": 14, "end": 14,
                       "first": "...", "last": "...", "value": "..."},
                      {"property": "<relation property key>", "ref": "<N key>"},
                      {"property": "<relation property key>", "ref": "<E key>",
                       "start": 15, "end": 15, "first": "...", "last": "..."}]}
    ]}
    Every new occurrence has a unique "id" (N1, N2, ...). List new occurrences in
    document order. Return {"segments": []} if nothing is found.

    For example, the line
  15| Artikel 1. Invoeren van éénrichtingsverkeer.
    is returned as:
    {"id": "N1", "shape": "S2", "start": 15, "end": 15, "first": "Artikel 1.",
     "last": "éénrichtingsverkeer.", "properties": [
      {"property": "S2.P2", "start": 15, "end": 15, "first": "1", "last": "1"},
      {"property": "S2.P1", "start": 15, "end": 15, "first": "Invoeren",
       "last": "éénrichtingsverkeer."}]}

    STRUCTURES
    """

    USER_PROMPT_TEMPLATE = """
    Find all STRUCTURES described in the system instructions in the numbered document
    below and return STRICT JSON as specified.

    Document:
    {text}
    """

    FEEDBACK_TEMPLATE = """
    PREVIOUS ATTEMPT
    Your previous answer was converted to RDF and validated against the SHACL shape.
    Correct the problems below where the document supports it. Keep every occurrence
    that was correct: do not drop occurrences to avoid problems, and never invent
    content that is not in the document. Correct the line numbers of the spans the
    problems point at, and copy their first and last words again, verbatim from the
    document.

    Previous answer:
    {previous_answer}

    Problems:
    {problems}
    """

    # Cardinalities are only a suggestion: properties that are not in the text (e.g. a language) stay unannotated
    IGNORED_CONSTRAINTS = {str(SH.MinCountConstraintComponent), str(SH.MaxCountConstraintComponent)}

    MU_UUID = URIRef("http://mu.semte.ch/vocabularies/core/uuid")
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
    # Placeholder in SYSTEM_PROMPT for the rules that depend on the target, and the width they are wrapped at
    TARGET_RULES = "<TARGET RULES>\n"
    RULE_WIDTH = 80
    # {class_name} is the lowercased local name of the shape's sh:targetClass, e.g. "artikel"
    URI_TEMPLATE = "http://data.lblod.info/id/{class_name}/{uuid}"
    EXAMPLE_MAX_LENGTH = 400
    # An occurrence's example shows how it is divided into its properties, so it gets more room
    OCCURRENCE_EXAMPLE_MAX_LENGTH = 2000
    # Language tag for rdf:langString values; detected from the text when None
    LANGUAGE: Optional[str] = None
    # Relative paths in the target are resolved against the project root
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    def __init__(self, target, api_key: str = None, base_url: str = None, model_name: str = "mistral-nemo", temperature: float = 0.0, provider: str = "ollama", max_retries: int = 3, retry_delay: float = 15.0):
        """target: what to segment and how, a TargetConfig of the app config (segmentation.targets)."""
        self.model_name = model_name

        # Shape, node shapes, examples, vocabularies and the vocabulary-specific settings come from the target
        self.usage_note_predicate = URIRef(target.usage_note_predicate) if target.usage_note_predicate else None
        self.text_predicate = URIRef(target.text_predicate)
        # Existing entities related to the target, which new entities can refer to (see fetch_related_entities)
        self.prefixes = target.prefixes
        self.related_entities = target.related_entities
        # Tuned to the target's documents: how they are divided (see _merge_sub_parts)
        self.prompt_rules = target.prompt_rules
        self.numbered_parts = target.numbered_parts
        # A relation to a class no structure in scope is of can also be given as the passage that mentions it, which
        # becomes a placeholder entity of that class (see _placeholder)
        self.placeholders = target.placeholders
        self.shape_graph = Graph().parse(self._resolve(target.shape_path), format="turtle")
        # rdfs:subClassOf hierarchy, e.g. besluit:Besluit -> eli:LegalExpression -> eli:Expression
        self.ontology = self._load_ontologies(target.ontologies)
        self.definitions = self._load_shape_definitions(target.node_shapes)
        if not self.definitions:
            raise ValueError("No sh:NodeShape found in the given SHACL shape.")
        self._load_examples(self._resolve(target.html_example_path).read_text(encoding="utf-8")
                            if target.html_example_path else None)

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

        self.system_prompt = self._render_system_prompt() + self._render_shape_definitions()

        self.analyzer = LLMAnalyzer(
            provider=provider,
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )

    # What the prompt calls a heading, as _is_heading_line decides it
    HEADING_RULE = ("A heading is a line of at most {max_words} words that ends with a colon, e.g. \"Decision:\"; a "
                    "longer line ending with a colon introduces what follows. A property's value never starts or ends "
                    "with a heading: a heading between two parts belongs to no property.")

    # For a target with numbered_parts, what _merge_sub_parts repairs when the LLM does not follow it
    NUMBERED_PARTS_RULE = ("Within one structure, a numbered sub-part (such as 1.1 or 2.1 under 1 or 2) is part of the "
                           "occurrence it belongs to, not a separate occurrence of that structure: the occurrence "
                           "includes its sub-parts and ends where the next part at the same level starts (2 after 1).")

    # For a target with placeholders: relations to what is not one of the structures (see _placeholder)
    PLACEHOLDER_RULE = ("Unlike other relations, a relation that says \"give ... the text that mentions it\" can "
                        "refer to something that is not one of the STRUCTURES: give the \"ref\" of an existing entity "
                        "when it says so and it is one of them, otherwise the span of the passage of the document that "
                        "mentions it (e.g. the name of a document), without \"ref\", which becomes an entity of its "
                        "own. Only when the document mentions it, and never the occurrence itself.")

    def _render_system_prompt(self) -> str:
        """
        The system prompt with, among the rules on spans, the heading rule, the numbered parts rule (for a target
        with numbered_parts) and the target's own rules.
        """
        rules = [self.HEADING_RULE.format(max_words=self.HEADING_MAX_WORDS),
                 *([self.NUMBERED_PARTS_RULE] if self.numbered_parts else []),
                 *([self.PLACEHOLDER_RULE] if self.placeholders else []),
                 *self.prompt_rules]
        rules = "".join(textwrap.fill(rule, width=self.RULE_WIDTH, initial_indent="    - ", subsequent_indent="      ")
                        + "\n" for rule in rules)
        return self.SYSTEM_PROMPT.replace(self.TARGET_RULES, rules)

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

    def _can_play(self, types: set[str], target_class: Optional[str]) -> bool:
        """An entity of the given types is an instance of target_class, through the class hierarchy."""
        return bool(target_class) and bool(types & self._superclasses(target_class))

    # Shape reading
    def _value(self, subject, predicate) -> Optional[str]:
        value = self.shape_graph.value(subject, predicate)
        return str(value) if value is not None else None

    def _usage_note(self, node) -> Optional[str]:
        """Id of the node's example element in the example documents; none without a usage_note_predicate."""
        return self._value(node, self.usage_note_predicate) if self.usage_note_predicate else None

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
                    "usage_note": self._usage_note(prop_node),
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
                "usage_note": self._usage_note(shape_node),
                "properties": properties,
            })
        return definitions

    # Examples: the shape's usage notes refer to element ids in example HTML documents
    def _load_examples(self, html: Optional[str]) -> None:
        """Attach an example to each node and property shape whose usage note matches an element id."""
        if not html:
            return
        soup = BeautifulSoup(html, "html.parser")

        def element_for(usage_note: Optional[str]):
            return soup.find(id=usage_note) if usage_note else None

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
                # A relation's example is the text of what it refers to; its parts in an occurrence's example are
                # marked with the structure they are an occurrence of
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

    def _takes_placeholder(self, prop: Dict[str, Any]) -> bool:
        """The relation can be given as the passage that mentions what it refers to: no structure in scope is of it."""
        return self.placeholders and bool(prop["class"]) and self._structure_of(prop["class"]) is None

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

    def _render_example(self, example: Optional[Dict[str, Any]], indent: str, hint: Optional[str] = None) -> List[str]:
        """The example of a structure or property; hint: for a relation, what it can refer to (see _relation_hint)."""
        if not example or not example["value"]:
            return []
        label = ("Example, with its properties marked [key] ... [/key]" if example.get("annotated") else
                 "Example (hidden metadata, not visible text)" if example["hidden"] else
                 f"Example of what it refers to ({hint})" if hint else
                 "Example")
        return [f"{indent}{label}: {example['value']}"]

    def _relation_hint(self, prop: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        What the relation can refer to, for the prompt: an occurrence of a structure in scope of its class, an existing
        entity of its class (in the context of this call), or, with placeholders, the passage that mentions it; none
        when nothing can be it.
        """
        structure = self._structure_of(prop["class"])
        options = []
        if structure:
            options.append(f"the \"ref\" of an {structure} occurrence")
        if any(self._can_play(entity["types"], prop["class"]) for entity in (context or {}).get("entities", {}).values()):
            options.append("the \"ref\" of an existing entity of that class")
        if self._takes_placeholder(prop):
            options.append("the span of the text that mentions it")
        return "give " + " or ".join(options) if options else None

    def _render_shape_definitions(self, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Describe the node shapes and their properties as plain text for the prompt. With the context of a call, a
        relation says which of its existing entities it can refer to.
        """
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
                hint = self._relation_hint(prop, context) if prop["class"] else None
                if prop["class"]:
                    details.append(f"relation to an instance of <{prop['class']}>: "
                                   + (hint or "nothing in scope can be it, leave it out"))
                if prop["min_count"] is not None:
                    details.append(f"min {prop['min_count']}")
                if prop["max_count"] is not None:
                    details.append(f"max {prop['max_count']}")
                lines.append(f"{line} ({', '.join(details)})")
                if not prop["class"] or hint:
                    lines += self._render_example(prop.get("example"), "        ", hint)
            lines.append("")
        return "\n".join(lines)

    def fetch_related_entities(self, target_uri: str) -> Dict[str, Optional[str]]:
        """
        The existing entities related to the target, which new entities can refer to, by URI, with their label:
        found by the target's related_entities query (a SPARQL pattern binding ?entity from ?target, and optionally a
        ?label describing it to the LLM, e.g. a mandataris's name and role); none without one.
        """
        from helpers import query
        from escape_helpers import sparql_escape_uri

        if not self.related_entities:
            return {}
        q = (f"{self.prefixes}\n"
             f"SELECT DISTINCT ?entity ?label WHERE {{\n"
             f"  VALUES ?target {{ {sparql_escape_uri(target_uri)} }}\n"
             f"  {self.related_entities}\n"
             f"}}")
        labels: Dict[str, Optional[str]] = {}
        for b in query(q, sudo=True).get("results", {}).get("bindings", []):
            if b.get("entity", {}).get("type") == "uri":
                labels[b["entity"]["value"]] = labels.get(b["entity"]["value"]) or b.get("label", {}).get("value")
        return labels

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

    def _build_context(self, target_uri: Optional[str]) -> Dict[str, Any]:
        """Existing entities related to the target (keys E1, E2, ...) with their label, types and known properties."""
        labels = self.fetch_related_entities(target_uri) if target_uri else {}
        # In the order of their labels (e.g. names), which the LLM looks them up by; without one, by URI
        labels = dict(sorted(labels.items(), key=lambda item: (" ".join((item[1] or "").split()).lower(), item[0])))
        graph = self.fetch_entities(list(labels))

        entities = {}
        for i, (uri, label) in enumerate(labels.items(), start=1):
            types = {str(t) for t in graph.objects(URIRef(uri), RDF.type)}
            entities[f"E{i}"] = {"uri": uri, "label": label, "types": types}
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

    # Above the list of existing entities: each one the document mentions is to be referred to
    RELATED_RULE = ("Check each of the existing entities below for whether the document mentions it (e.g. by the "
                    "name in its label), one by one, until the end of the list: every one it mentions is referred to "
                    "by a relation of an occurrence.")

    def _render_context(self, context: Dict[str, Any]) -> str:
        if not context["entities"]:
            return ""
        lines = ["", "    EXISTING ENTITIES",
                 *textwrap.wrap(self.RELATED_RULE, width=self.RULE_WIDTH, initial_indent="    ",
                                subsequent_indent="    "), ""]
        for key, entity in context["entities"].items():
            subject = URIRef(entity["uri"])
            lines.append(f"[{key}] <{entity['uri']}>")
            if entity.get("label"):
                lines.append(f"  Label: {' '.join(entity['label'].split())}")
            lines.append(f"  Types: {', '.join(f'<{t}>' for t in sorted(entity['types'])) or 'unknown'}")
            if entity.get("label"):
                continue  # the label describes it; its values would only bury the keys and labels in long lists
            known = []
            for p in sorted({p for p in context["graph"].predicates(subject) if p not in (RDF.type, self.MU_UUID)}, key=str):
                values = [f"<{o}>" if isinstance(o, URIRef) else f'"{" ".join(str(o).split())[:60]}"'
                          for o in list(context["graph"].objects(subject, p))[:3]]
                known.append(f"    <{p}>: {', '.join(values)}")
            if known:
                lines.append("  Known values:")
                lines += known
        return "\n".join(lines) + "\n"

    # Indexing the document: every line numbered, so the LLM can give a span as the lines it runs from and to with
    # the words at its start and end, which _resolve_span resolves back to exact character offsets
    def _index_document(self, text: str) -> tuple[str, List[int], str]:
        """
        The document as the LLM sees it: every line rendered with its 1-based number, truncated at the last line
        boundary <= TEXT_LIMIT so no partial line is ever indexed. Returns (rendered, line_offsets, source): the
        numbered document for the prompt, the start offset of every line in source (line_offsets[n - 1] is where
        line n starts), and the exact text the LLM saw, which the offsets and the resolved spans refer to.
        """
        source = text[:self.TEXT_LIMIT]
        if len(source) < len(text) and "\n" in source:
            source = source[:source.rfind("\n") + 1]
        lines = source.split("\n")
        if source.endswith("\n"):
            lines.pop()  # the final newline ends the last line: there is no empty line after it
        line_offsets = []
        offset = 0
        for line in lines:
            line_offsets.append(offset)
            offset += len(line) + 1
        rendered = "\n".join(f"{number:>4}| {line}" for number, line in enumerate(lines, start=1))
        return rendered, line_offsets, source

    def _resolve_span(self, index: tuple[str, List[int], str], span: Dict[str, Any], key: str) -> tuple[int, int] | str:
        """
        The (start, end) character offsets in the source text of a span the LLM gave as the 1-based line numbers it
        runs from and to ("start", "end"), with the verbatim words at its start and end ("first", "last"): the lines
        are sliced, and the anchors are matched in the first and last non-blank line of the range (leftmost and
        rightmost), which pins the span down exactly and checks the line numbers. A problem, phrased with the key
        the LLM knows the span by, when the line numbers and anchors do not pin it down.
        """
        _, line_offsets, source = index
        span = span if isinstance(span, dict) else {}
        count = len(line_offsets)
        # The last line's text ends before the newline that ends it
        document_end = len(source) - (1 if source.endswith("\n") else 0)

        def line_bounds(number: int) -> tuple[int, int]:
            """The bounds of a numbered line's text in source, without the newline that ends it."""
            return line_offsets[number - 1], (line_offsets[number] - 1 if number < count else document_end)

        def line_text(number: int) -> str:
            lo, hi = line_bounds(number)
            return source[lo:hi]

        start, end = self._line_number(span.get("start")), self._line_number(span.get("end"))
        if start is None or end is None:
            return (f"{key}: \"start\" and \"end\" must be the numbers of the lines the span runs from and to, "
                    f"got {span.get('start')!r} and {span.get('end')!r}")
        if not 1 <= start <= count or not 1 <= end <= count:
            return f"{key}: lines {start} to {end} are outside the document: it has {count} numbered lines"
        if start > end:
            return f"{key}: lines {start} to {end} are reversed: \"start\" must not come after \"end\""

        non_blank = [number for number in range(start, end + 1) if line_text(number).strip()]
        if not non_blank:
            return f"{key}: lines {start} to {end} are blank: the span has no text"
        first_line, last_line = non_blank[0], non_blank[-1]
        first, last = span.get("first"), span.get("last")
        if not isinstance(first, str) or not first.split() or not isinstance(last, str) or not last.split():
            return (f"{key}: \"first\" and \"last\" must be a few verbatim words from the start and the end of the "
                    f"span, got {first!r} and {last!r}")

        first_lo, first_hi = line_bounds(first_line)
        first_match = self._pattern(first).search(source, first_lo, first_hi)
        last_lo, last_hi = line_bounds(last_line)
        # The anchor ends the span: its rightmost match in the line
        last_matches = list(self._pattern(last).finditer(source, last_lo, last_hi))
        if first_match is None:
            return (f"{key}: the span does not start with {first!r}: line {first_line} of the range reads "
                    f"{' '.join(source[first_lo:first_hi].split())[:60]!r}")
        if not last_matches:
            return (f"{key}: the span does not end with {last!r}: line {last_line} of the range reads "
                    f"{' '.join(source[last_lo:last_hi].split())[:60]!r}")
        # search() and finditer() report their matches at their absolute offsets in the source text
        start_offset, end_offset = first_match.start(), last_matches[-1].end()
        # The span must actually run from the first anchor to the last: anchors that overlap the wrong way around
        # would resolve to a span that does not start with "first" or does not end with "last"
        if first_match.end() > end_offset or last_matches[-1].start() < start_offset:
            return f"{key}: \"first\" {first!r} does not come before \"last\" {last!r} in line {first_line}"
        return start_offset, end_offset

    def _line_number(self, value: Any) -> Optional[int]:
        """A line number as an int (a numeric string or a whole number is coerced), or None when it is not one."""
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str):
            try:
                return int(value.strip())
            except ValueError:
                return None
        return None

    # The words of an anchor, matched tolerantly: whitespace in them matches any run of whitespace (LLMs copy
    # line breaks inconsistently), and a non-ASCII symbol matches any symbol
    def _pattern(self, found_text: str) -> re.Pattern:
        return re.compile(r"\s+".join(self._token_pattern(token) for token in found_text.split()))

    def _token_pattern(self, token: str) -> str:
        """Exact characters, except that a non-ASCII symbol (e.g. a bullet "•") matches any single symbol."""
        return "".join(r"[^\w\s]" if ord(c) > 127 and not unicodedata.category(c).startswith(("L", "N")) else re.escape(c)
                       for c in token)

    def segment_and_validate(self, text: str, task: Any = None, target: Optional[str] = None) -> Dict[str, Any]:
        """
        Segment, serialize to Turtle and validate against the SHACL shape. While the result does not
        conform or has spans that cannot be resolved, ask the LLM again with its previous answer and
        the problems, up to max_attempts. Returns the attempt with the fewest problems per entity:
            {'turtle': str, 'segments': [...], 'conforms': bool, 'violations': [...],
             'unlocated': [...], 'attempts': int,
             'usage': {'calls': int, 'input_tokens': int, 'output_tokens': int}}  # summed over all attempts
        task: the task being processed (with task_uri), on which the LLM calls are recorded. target: the URI of the
        resource the text is of, whose related entities (see fetch_related_entities) new entities can refer to.
        """
        context = self._build_context(target)
        context["language"] = self._detect_language(text)
        system_prompt = self.system_prompt + self._render_context(context)
        # The document as the LLM sees it: numbered lines, truncated at a line boundary. _index_document is
        # deterministic, so one index serves every attempt: the prompt and the span resolution see the same text,
        # and the resolved offsets refer to exactly it.
        index = self._index_document(text)
        best = None
        feedback = None
        usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0}
        for attempt in range(1, self.MAX_ATTEMPTS + 1):
            answer = self._ask(index[0], task=task, feedback=feedback, system_prompt=system_prompt)
            tokens = getattr(self.analyzer.last_response, "usage_metadata", None) or {}
            usage["calls"] += 1
            usage["input_tokens"] += tokens.get("input_tokens", 0) or 0
            usage["output_tokens"] += tokens.get("output_tokens", 0) or 0
            unlocated: List[str] = []
            segments = self._segments_from_answer(index, answer, unlocated, context)
            turtle = self.to_turtle(segments)
            try:
                # The LLM knows entities by their keys (N1, E1, ...), not by their (minted) URIs
                keys = {e["uri"]: key for key, e in context["entities"].items()}
                keys.update({s["uri"]: s["key"] for s in segments})
                conforms, violations = self.validate(turtle, context["graph"], keys)
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
        Parse the Turtle (raises when invalid) and validate it, together with the existing entities' triples (e.g.
        the type a relation to one needs), against the SHACL shape. Cardinalities are ignored, and only violations on
        what the Turtle adds count: problems in the existing data cannot be fixed by the LLM.
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
                "value": (keys or {}).get(self._value_of(report_graph, result, SH.value))
                         or self._value_of(report_graph, result, SH.value),
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
        values = [f'{prop_keys.get(str(p), p)}='
                  + (keys[str(o)] if str(o) in keys else f'"{" ".join(str(o).split())[:60]}"')
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

    def to_turtle(self, segments: List[Dict[str, Any]]) -> str:
        """Serialize the new entities: their type, mu:uuid, property values and relations."""
        blocks = []
        for segment in segments:
            target_class = segment.get("class") or self.target_classes.get(segment["shape"])
            statements = [f"a {URIRef(target_class).n3()}"] if target_class else []
            statements.append(f"{self.MU_UUID.n3()} {Literal(segment['uuid']).n3()}")
            for prop in segment["properties"]:
                value = (URIRef(prop["ref"]) if "ref" in prop else
                         Literal(prop["value"], datatype=URIRef(prop["datatype"])) if "datatype" in prop else
                         Literal(prop["text"], lang=prop.get("lang")))
                statements.append(f"{URIRef(prop['path']).n3()} {value.n3()}")
            blocks.append(f"{URIRef(segment['uri']).n3()} " + " ;\n    ".join(statements) + " .")
        return "\n\n".join(blocks) + ("\n" if blocks else "")

    def _ask(self, document: str, task: Any = None, feedback: Optional[str] = None, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        logger.info(f"Running shape-driven LLM segmentation with {self.analyzer.model_name}...")

        # Attach the current task so the analyzer logs the AI call (see record_llm_call).
        self.analyzer._task = task

        try:
            result = self.analyzer.analyze_single_entry(
                text=document,
                system_prompt=system_prompt or self.system_prompt,
                # The template is str.format()-ed with the text, so braces in the feedback are escaped
                user_prompt_template=self.USER_PROMPT_TEMPLATE + (
                    feedback.replace("{", "{{").replace("}", "}}") if feedback else ""),
                expected_schema=self.RESULTS_SCHEMA,
                # The document is already exactly the text the resolved offsets refer to
                text_limit=None,
            )
        except Exception as e:
            logger.exception("Shape-driven LLM segmentation failed")
            raise RuntimeError(
                f"Shape-driven LLM segmentation failed ({self.analyzer.model_name}): {e}"
            ) from e

        return result

    # A heading line: at most this many words, ending with a colon, e.g. "BESLIST:" between a motivation and the
    # decision itself; a longer line ending with a colon introduces what follows, e.g. "... worden door:". The
    # prompt describes headings the same way (HEADING_RULE)
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
        relations, and its values of properties the parent has no value of, are the parent's. Only for a target with
        numbered_parts (see NUMBERED_PARTS_RULE).
        """
        if not self.numbered_parts:
            return segments
        merged: Dict[str, Dict[str, Any]] = {}  # sub-part URI -> parent
        new = sorted((s for s in segments if s["shape"]), key=lambda s: s["start"])
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
                    if "ref" not in prop and "value" not in prop and prop["end"] == old_end:
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
        """Values, and refs to existing entities by their mention, in document order; other relations last."""
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

    @staticmethod
    def _has_span(prop: Dict[str, Any]) -> bool:
        """The LLM gave (part of) a span: line numbers or anchors."""
        return any(prop.get(field) not in (None, "") for field in ("start", "end", "first", "last"))

    def _placeholder(self, source: str, start: int, end: int, class_uri: str,
                     placeholders: Dict[tuple, Dict[str, Any]]) -> Dict[str, Any]:
        """
        A placeholder entity of the class for the passage that mentions it: one per distinct passage, so that a passage
        referred to twice (e.g. the same document) is one entity. It has no properties, only its class and passage.
        """
        passage = " ".join(source[start:end].split())
        if (class_uri, passage) not in placeholders:
            uri, entity_uuid = self._mint_uri(class_uri)
            placeholders[(class_uri, passage)] = {
                "key": f"P{len(placeholders) + 1}", "shape": None, "class": class_uri, "uri": uri, "uuid": entity_uuid,
                "text": source[start:end], "start": start, "end": end, "properties": []}
            logger.info(f"Placeholder <{class_uri}> for {passage[:60]!r}")
        return placeholders[(class_uri, passage)]

    def _unlinked(self, segments: List[Dict[str, Any]]) -> List[str]:
        """New entities that another returned entity could refer to through a relation property, but none does."""
        referenced = {prop["ref"] for segment in segments for prop in segment["properties"] if "ref" in prop}
        definitions = {d["uri"]: d for d in self.definitions}
        problems = []
        for segment in segments:
            if segment["uri"] in referenced or not segment["shape"]:
                continue
            classes = self._superclasses(self.target_classes.get(segment["shape"]) or "")
            candidates = [f"{owner['key']} via {prop['key']} ({prop['name']})"
                          for owner in segments if owner is not segment and owner["shape"]
                          for prop in definitions[owner["shape"]]["properties"] if prop["class"] in classes]
            if candidates:
                problems.append(f"{segment['key']} is not referred to by any relation; it could be referred to by "
                                f"{', '.join(candidates)}")
        return problems

    def _segments_from_answer(self, index: tuple[str, List[int], str], result: Dict[str, Any],
                              problems: Optional[List[str]] = None,
                              context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Build the new entities from the LLM's answer: resolve the span of every occurrence and property value (the
        lines it runs from and to, with its first and last words) to exact offsets in the source text the LLM saw,
        and resolve relation refs to occurrences and existing entities. Anything that cannot be resolved is added to
        problems.
        """
        problems = problems if problems is not None else []
        context = context or {"graph": Graph(), "entities": {}}
        entities = context["entities"]
        _, _, source = index
        segments: List[Dict[str, Any]] = []
        by_key: Dict[str, Dict[str, Any]] = {}
        items = [dict(item) for item in result.get("segments", []) if isinstance(item, dict)]
        placeholders: Dict[tuple, Dict[str, Any]] = {}  # (class, passage) -> placeholder entity

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
                problems.append(f"{entity_key} is an existing entity: do not return it or give it properties; "
                                f"refer to it from an occurrence with \"ref\" instead")
                continue

            occurrence_key = str(item.get("id") or "").strip() or f"N?{len(segments) + 1}"
            found = self._resolve_span(index, item, occurrence_key)
            if isinstance(found, str):
                logger.warning(f"Could not resolve {occurrence_key} ({shape_key}): {found}")
                problems.append(found)
                continue
            start, end = found
            if occurrence_key in by_key or occurrence_key in entities:
                problems.append(f"id {occurrence_key!r} is used more than once")
                occurrence_key = f"{occurrence_key}?{len(segments) + 1}"
            uri, entity_uuid = self._mint_uri(target["shape"])
            segment = {"key": occurrence_key, "shape": target["shape"], "uri": uri, "uuid": entity_uuid,
                       "text": source[start:end], "start": start, "end": end, "properties": []}
            by_key[occurrence_key] = segment
            segments.append(segment)
            resolved_items.append((item, segment, shape_key))

        # Second pass: property values and relations. A value is usually inside its occurrence, but can be elsewhere in
        # the document when it is only there (e.g. a location in a page footer)
        for item, segment, shape_key in resolved_items:
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

                if self.properties_by_key[prop_key]["class"]:
                    ref = str(prop.get("ref") or "").strip()
                    relation_class = self.properties_by_key[prop_key]["class"]
                    # The span of a passage instead of a ref (also when the ref is not a key, e.g. "ref": "span")
                    if (ref not in by_key and ref not in entities and self._has_span(prop)
                            and self._takes_placeholder(self.properties_by_key[prop_key])):
                        found = self._resolve_span(index, prop, f"{segment['key']}, property {prop_key}")
                        if isinstance(found, str):
                            logger.warning(f"Could not resolve the passage of {prop_key} of {segment['key']}: {found}")
                            problems.append(found)
                            continue
                        placeholder = self._placeholder(source, *found, relation_class, placeholders)
                        if not any(s is placeholder for s in segments):
                            segments.append(placeholder)
                        segment["properties"].append({"path": prop_target["path"], "ref": placeholder["uri"]})
                        continue
                    if ref in by_key:
                        hint = self._relation_hint(self.properties_by_key[prop_key], context) or "leave it out"
                        referred_class = self.target_classes.get(by_key[ref]["shape"])
                        if by_key[ref] is segment:
                            problems.append(f"{segment['key']}, property {prop_key}: a relation never refers to the "
                                            f"occurrence itself; {hint}")
                            continue
                        if referred_class and relation_class not in self._superclasses(referred_class):
                            problems.append(f"{segment['key']}, property {prop_key}: {ref} is not an instance of "
                                            f"<{relation_class}>; {hint}")
                            continue
                    if ref in entities and not self._can_play(entities[ref]["types"], relation_class):
                        problems.append(f"{segment['key']}, property {prop_key}: existing entity {ref} is not an "
                                        f"instance of <{relation_class}>")
                        continue
                    ref_uri = by_key[ref]["uri"] if ref in by_key else entities[ref]["uri"] if ref in entities else None
                    if ref_uri is None:
                        problems.append(f"{segment['key']}, property {prop_key}: relation needs a \"ref\" to an "
                                        f"occurrence id or existing entity key, got {prop.get('ref')!r}")
                        continue
                    if ref in entities:
                        # Where the document mentions the existing entity (asked for so that the LLM goes through the
                        # document, mention by mention). Not every entity is mentioned by a name as such (e.g. a body
                        # as "de Gemeenteraad"), so the ref holds without
                        mention = self._resolve_span(index, prop, f"{segment['key']}, property {prop_key}")
                        if isinstance(mention, str):
                            logger.warning(f"No mention of {ref} for {prop_key} of {segment['key']}: {mention}")
                            segment["properties"].append({"path": prop_target["path"], "ref": ref_uri})
                            continue
                        segment["properties"].append({"path": prop_target["path"], "ref": ref_uri,
                                                      "text": source[mention[0]:mention[1]],
                                                      "start": mention[0], "end": mention[1]})
                        continue
                    segment["properties"].append({"path": prop_target["path"], "ref": ref_uri})
                    continue

                found = self._resolve_span(index, prop, f"{segment['key']}, property {prop_key}")
                if isinstance(found, str):
                    logger.warning(f"Could not resolve {prop_key} of {segment['key']}: {found}")
                    problems.append(found)
                    continue
                datatype = self.properties_by_key[prop_key]["datatype"]
                typed = datatype is not None and datatype not in self.STRING_DATATYPES
                if typed:
                    value = Literal(str(prop.get("value") or "").strip(), datatype=URIRef(datatype))
                    if not str(value) or value.ill_typed:
                        problems.append(f"{segment['key']}, property {prop_key}: needs a \"value\" in the form of "
                                        f"<{datatype}>, got {prop.get('value')!r}")
                        continue
                prop_start, prop_end = self._without_heading_lines(source, *found)
                segment["properties"].append({
                    "path": prop_target["path"],
                    "text": source[prop_start:prop_end],
                    "start": prop_start,
                    "end": prop_end,
                    "lang": context.get("language") if datatype == str(RDF.langString) else None,
                    **({"value": str(value), "datatype": datatype} if typed else {}),
                })

            segment["properties"].sort(key=self._property_order)

        segments = self._merge_sub_parts(segments, source)
        problems += self._unlinked(segments)

        # In document order
        segments.sort(key=lambda s: (s["start"], -s["end"]))
        logger.info(f"Resolved {len(segments)} shape segments")
        return segments
