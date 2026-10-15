# Entity extraction service

Extracts the entities in decision texts, as described by a SHACL shape. For each expression in a `shape-ner` task, it
segments the text along the shape: an LLM finds the structures the shape describes (e.g. a decision and its articles)
and the values of their properties. The service then:

- locates every value verbatim in the text;
- validates the result against the shape, and asks the LLM again with the violations, up to three times;
- writes the result to the AI graph: the expression enriched with its segments, the new entities (e.g. articles), and
  an `oa:Annotation` linking a decision to an existing one with the same title.

The task is `EntityExtractionTask` in [src/task/entity_extraction.py](src/task/entity_extraction.py); its
segmentor is `LLMSegmentorWithShape` in [src/library/segmentors.py](src/library/segmentors.py).

## Requirements

- Docker and Docker Compose
- The app-decide stack (triplestore, delta-notifier, job-controller, annotation-job-splitter), whose Docker network
  `app-decide_default` the service joins
- An API key for the configured LLM provider

## Quick start

Put the API key in a `docker-compose.override.yml` (ignored by git, so the key is never committed):
```yaml
services:
  shape-ner-service:
    environment:
      SEGMENTATION__LLM__API_KEY: "..."
```
and start the service:
```bash
docker compose up
```

The service listens on host port 8086 (see `docker-compose.yaml`). The delta-notifier calls `POST /delta` when tasks
are scheduled.

## API endpoints

- `POST /delta` - Receive delta notifications for task processing
- `GET /task/operations` - List available task operations
- `POST /test/job` - **Development only** (`MODE=development`). Inserts a `shape-ner` job with a SHACL target shape and a scheduled `singleton-job` task, so the job-controller and annotation-job-splitter create the `shape-ner` tasks. Body: either `target_nodes` (expression URIs), or `target_class` together with `target_graph`:
  ```bash
  curl -X POST localhost:8086/test/job -H 'Content-Type: application/json' \
    -d '{"target_nodes": ["http://data.lblod.info/id/expressions/<uuid>"]}'
  ```

An expression that already has segmentation values in the AI graph is passed forward without segmenting it again; to
segment it again, delete those triples.

## Configuration

Settings are in `config.json`, validated at startup via Pydantic: invalid values cause a startup failure with a
descriptive error. Secrets go in environment variables, with `__` as the nested delimiter (e.g.
`SEGMENTATION__LLM__API_KEY`), set in `docker-compose.override.yml`; never commit them to `config.json`.

### `segmentation.llm`

| Key | Default | Effect |
|---|---|---|
| `provider` | `ollama` | LangChain provider name: `mistralai`, `ollama` (the providers whose packages are in `requirements.txt`) |
| `model_name` | `mistral-nemo` | Model identifier as understood by the provider |
| `api_key` | `null` | API key, required for remote providers; set it via `SEGMENTATION__LLM__API_KEY` |
| `base_url` | `null` | Custom endpoint, required for Ollama and self-hosted models |
| `temperature` | `0.0` | Lower = more deterministic segmentation |
| `max_retries` | `3` | Attempts per LLM call on failure |
| `retry_delay` | `15.0` | Seconds between retries |

### `segmentation.shape`

What to segment. Relative paths are resolved against the project root.

| Key | Effect |
|---|---|
| `shape_path` | SHACL shape (Turtle) describing what to segment |
| `node_shapes` | Node shapes of the shape to segment; empty for all |
| `example_paths` | Example HTML documents whose element ids match the shape's usage notes; they become the examples in the prompt |
| `ontologies` | Vocabularies whose `rdfs:subClassOf` hierarchy decides which shapes an existing entity can play |
| `usage_note_predicate` | Property of the shape's node and property shapes holding the id of their example element |
| `text_predicate` | Property of an input resource holding the text to segment |
| `same_as_predicate` | Property whose (almost) equal value identifies the same entity in the triplestore, e.g. a title |
| `same_as_class` | Segmented entities of this class are linked to the same entity in the triplestore (an original); `null` to link nothing |

## Tests

```bash
docker compose exec shape-ner-service sh -c 'cd /app && uv run --with pytest pytest'
```

The segmentation test cases replay recorded LLM answers, so they need neither the LLM nor the triplestore. To add a
case, see [tests/cases/README.md](tests/cases/README.md).

## Deployment

- **Network**: ensure the `app-decide_default` network exists, or update the network name in `docker-compose.yaml`.
- **Ports**: the service runs on port 80 internally. For production, expose it via the dispatcher instead of opening
  ports directly.
- **Volumes**: `config.json` is mounted read-only at `/config/config.json`.
