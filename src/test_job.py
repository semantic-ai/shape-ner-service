"""
Development-only endpoint to insert a shape-ner test job.

Mirrors what the job self-service frontend writes for an annotation job: a
busy job pointing to a SHACL target shape, and a scheduled singleton-job task
at index 0. The job-controller and annotation-job-splitter in app-decide then
take over and create one shape-ner task per target.
"""
import uuid
from datetime import datetime, timezone
from string import Template
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel, model_validator

from helpers import update, logger
from escape_helpers import sparql_escape_uri, sparql_escape_string, sparql_escape_datetime
from decide_ai_service_base.sparql_config import get_prefixes_for_query, GRAPHS, JOB_STATUSES

SHAPE_NER_JOB_OPERATION = "http://lblod.data.gift/id/jobs/concept/JobOperation/shape-ner"
SINGLETON_JOB_OPERATION = "http://lblod.data.gift/id/jobs/concept/TaskOperation/singleton-job"
TEST_JOB_CREATOR = "http://lblod.data.gift/services/shape-ner-service"

JOB_BASE = "http://redpencil.data.gift/id/annotation-job/"
TASK_BASE = "http://redpencil.data.gift/id/task/"
DATA_CONTAINER_BASE = "http://redpencil.data.gift/id/dataContainers/"
SHAPE_BASE = "http://redpencil.data.gift/id/shapes/"


class TestJobRequest(BaseModel):
    """Targets for the job, either explicit expression URIs or a class within a graph."""
    target_nodes: list[str] = []
    target_class: Optional[str] = None
    target_graph: Optional[str] = None

    @model_validator(mode="after")
    def check_targets(self):
        if not self.target_nodes and not (self.target_class and self.target_graph):
            raise ValueError(
                "Provide either target_nodes, or target_class together with target_graph")
        return self


class TestJobResponse(BaseModel):
    job: str
    task: str
    shape: str


router = APIRouter()


@router.post("/test/job", status_code=201)
def create_test_job(body: TestJobRequest) -> TestJobResponse:
    job_id, task_id, container_id, shape_id = (str(uuid.uuid4()) for _ in range(4))
    job_uri = JOB_BASE + job_id
    task_uri = TASK_BASE + task_id
    container_uri = DATA_CONTAINER_BASE + container_id
    shape_uri = SHAPE_BASE + shape_id
    now = sparql_escape_datetime(datetime.now(timezone.utc))

    shape_targets = "".join(
        f"\n                sh:targetNode {sparql_escape_uri(node)} ;" for node in body.target_nodes)
    if body.target_class:
        shape_targets += f"\n                sh:targetClass {sparql_escape_uri(body.target_class)} ;"
    job_graph_target = (
        f"\n                ext:graphForTargets {sparql_escape_uri(body.target_graph)} ;"
        if body.target_graph else "")

    q = Template(
        get_prefixes_for_query("task", "adms", "dct", "mu", "ext", "nfo") +
        """
        PREFIX cogs: <http://vocab.deri.ie/cogs#>
        PREFIX sh: <http://www.w3.org/ns/shacl#>
        INSERT DATA {
          GRAPH $graph {
            $shape a sh:NodeShape ;$shape_targets
                mu:uuid $shape_id .

            $job a cogs:Job, ext:AnnotationJob ;
                mu:uuid $job_id ;
                dct:created $now ;
                dct:modified $now ;
                dct:creator $creator ;
                adms:status $busy ;
                task:operation $job_operation ;$job_graph_target
                ext:shapeForTargets $shape .

            $container a nfo:DataContainer ;
                mu:uuid $container_id .

            $task a task:Task ;
                mu:uuid $task_id ;
                dct:created $now ;
                dct:modified $now ;
                dct:isPartOf $job ;
                adms:status $scheduled ;
                task:operation $task_operation ;
                task:index "0" ;
                task:inputContainer $container .
          }
        }
        """
    ).substitute(
        graph=sparql_escape_uri(GRAPHS["jobs"]),
        shape=sparql_escape_uri(shape_uri),
        shape_id=sparql_escape_string(shape_id),
        shape_targets=shape_targets,
        job=sparql_escape_uri(job_uri),
        job_id=sparql_escape_string(job_id),
        now=now,
        creator=sparql_escape_uri(TEST_JOB_CREATOR),
        busy=sparql_escape_uri(JOB_STATUSES["busy"]),
        job_operation=sparql_escape_uri(SHAPE_NER_JOB_OPERATION),
        job_graph_target=job_graph_target,
        container=sparql_escape_uri(container_uri),
        container_id=sparql_escape_string(container_id),
        task=sparql_escape_uri(task_uri),
        task_id=sparql_escape_string(task_id),
        scheduled=sparql_escape_uri(JOB_STATUSES["scheduled"]),
        task_operation=sparql_escape_uri(SINGLETON_JOB_OPERATION),
    )

    update(q, sudo=True)
    logger.info(f"Inserted test job {job_uri} with first task {task_uri}")
    return TestJobResponse(job=job_uri, task=task_uri, shape=shape_uri)
