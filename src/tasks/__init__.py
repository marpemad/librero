"""Gestor ligero de tareas — tablón Kanban en un Markdown del Vault."""
from .board import (
    STATE_EMOJI,
    STATE_LABEL,
    Task,
    TaskBoard,
    TaskState,
    extract_task_description,
)

__all__ = [
    "Task",
    "TaskBoard",
    "TaskState",
    "STATE_EMOJI",
    "STATE_LABEL",
    "extract_task_description",
]
