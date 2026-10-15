import os
from threading import Lock
from helpers import logger

from src.airo import register_airo
from src.task import EntityExtractionTask, EXTRACTOR_COMPONENT
from decide_ai_service_base.util import fail_busy_and_scheduled_tasks, TaskProcessor, wait_for_triplestore, process_open_tasks, write_agent_info

from decide_ai_service_base.schema import NotificationResponse, TaskOperationsResponse
from decide_ai_service_base.task import Task

from fastapi import APIRouter, BackgroundTasks

# The template mounts every APIRouter found in this module, so the test job
# endpoint only exists when running in development mode
if os.environ.get("MODE") == "development":
    from src.test_job import router as test_job_router


_open_tasks_lock = Lock()


@app.on_event("startup")
async def startup_event():
    logger.info("Startup running in PID %s", os.getpid())
    wait_for_triplestore()
    write_agent_info("http://lblod.data.gift/id/components/shape-ner/v1.0.0", EXTRACTOR_COMPONENT)

    fail_busy_and_scheduled_tasks()
    register_airo()
    process_open_tasks(_open_tasks_lock)



router = APIRouter()


@router.post("/delta", status_code=202)
async def delta(background_tasks: BackgroundTasks) -> NotificationResponse:
    processor = TaskProcessor(_open_tasks_lock)
    background_tasks.add_task(processor)
    return NotificationResponse(status="accepted", message="Processing started")


@router.get("/task/operations")
def get_task_operations() -> TaskOperationsResponse:
    return TaskOperationsResponse(
        task_operations=[
            clz.__task_type__ for clz in Task.supported_operations()
        ]
    )
