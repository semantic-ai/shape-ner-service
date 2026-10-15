# Segmentation test cases

Each folder here is one test case for `tests/test_segmentation_cases.py`. A case is one text with the segmentation we
expect for it. A normal `pytest` run replays the LLM answers recorded for each case, so it needs neither the LLM nor the
triplestore. The file formats are described in the docstring of `tests/support.py`.

A folder is a case when it holds an `input.ttl`. You write these files by hand:

| File           | Required | Contents                                                                            |
|----------------|----------|-------------------------------------------------------------------------------------|
| `input.ttl`    | yes      | the resource to segment, with its text                                              |
| `context.ttl`  | no       | what else the triplestore holds about the input, e.g. its types from other graphs   |
| `patterns.ttl` | yes      | what the output must contain                                                        |
| `README.md`    | no       | title, description and source of the case (only for people, the tests don't read it) |

Recording the case writes `prompts/`, `answers.json` and `output.ttl`. Every run writes `out/`, which is not in git.

## Adding a case

### 1. Create the folder

Name it `SEGTC<number>-<short-name>`, with the next free number, e.g. `tests/cases/SEGTC0002-gent-notulen`.

### 2. Write `input.ttl`

`input.ttl` holds exactly one resource with the text to segment. The text goes under the `text_predicate` of the
case's target (see `test:target` in step 4; for the `uittreksel` target in `config.json`, `epvoc:expressionContent`).
Zero or several such resources make the case fail to load.

The resource also needs an `rdf:type` that can play one of the selected node shapes, e.g. `eli:Expression`. Without
one, the segmentor doesn't take it as the input.

```turtle
@prefix eli: <http://data.europa.eu/eli/ontology#> .
@prefix epvoc: <https://data.europarl.europa.eu/def/epvoc#> .

<http://data.lblod.info/id/expressions/...> a eli:Expression ;
    epvoc:expressionContent """
...the full text...
""" .
```

The quickest way to get real data is to query the triplestore of a running stack (e.g. in its SPARQL endpoint) and
paste the result as Turtle:

```sparql
CONSTRUCT { ?s ?p ?o } WHERE {
  VALUES ?s { <http://data.lblod.info/id/expressions/...> }
  GRAPH ?g { ?s ?p ?o }
}
```

### 3. Write `context.ttl` (optional)

Add what the triplestore holds about the input besides `input.ttl`, when it matters to the segmentation: e.g. its
types from other graphs (which decide the structures it can be enriched as), or values it already has (which the
segmentor does not add again).

### 4. Write `patterns.ttl`

`patterns.ttl` has one `test:Case` node with the case's target and counts. Every other triple in it is a
pattern the output must match:

```turtle
@prefix test: <http://example.org/segmentation-test#> .
@prefix besluit: <http://data.vlaanderen.be/ns/besluit#> .
@prefix eli: <http://data.europa.eu/eli/ontology#> .
@prefix prov: <http://www.w3.org/ns/prov#> .

[] a test:Case ;
    test:target "uittreksel" ;                                        # a target in config.json, default the first
    test:count [ test:class besluit:Besluit ; test:number 1 ] ,
               [ test:class besluit:Artikel ; test:number 6 ] .

<http://data.lblod.info/id/expressions/...> a besluit:Besluit ;
    eli:has_part [ a besluit:Artikel ; eli:number "1" ;
                   prov:value "Invoeren van éénrichtingsverkeer"^^test:startsWith ] .
```

- **URIs** must be in the output exactly as written.
- **Blank nodes** match any entity, since the URIs of new entities are minted per run.
- **Literals** are compared on their text, ignoring whitespace and language tags. `^^test:startsWith` and
  `^^test:contains` match part of the text.

Keep the patterns to what matters for the case: the general checks in `test_segmentation_cases.py` already cover what
holds for every segmentation. `SEGTC0001-zottegem-uittreksel/patterns.ttl` is a complete example.

### 5. Write `README.md`

Describe what the case is about and where the text comes from. Copy the layout of `SEGTC0001-zottegem-uittreksel/README.md`.

### 6. Record the case

Recording calls the LLM, so it needs the API key of the configured provider. Run it in the service container:

```sh
docker compose exec shape-ner-service sh -c 'cd /app && SEGMENTATION_RECORD=1 uv run --with pytest pytest tests/test_segmentation_cases.py -k SEGTC0002'
```

This writes `prompts/`, `answers.json` and `output.ttl`, and checks the patterns. If a pattern fails, look at
`out/segments.ttl`. If the pattern is wrong, fix it and replay (step 7); there's no need to record again. If the
segmentation is wrong, that's the bug the case is for.

Until a case is recorded, a normal run skips it ("has not been recorded yet").

### 7. Check the replay and commit

```sh
pytest tests/test_segmentation_cases.py -k SEGTC0002
```

Review `prompts/`, `answers.json` and `output.ttl`, then commit the folder (without `out/`).

## Keeping cases up to date

- **A change alters the prompts** (shape, examples, prompt text): the replay fails on the prompt snapshot. Record the
  affected cases again with `SEGMENTATION_RECORD=1` and review the diff.
- **A change alters only the output**, not the prompts: rewrite `output.ttl` from the recorded answers, without the
  LLM, with `SEGMENTATION_RECORD=output`.
