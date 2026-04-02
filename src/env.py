from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from src.dataset import load_synthetic_email_dataset
from src.graders import DeterministicTriageGrader
from src.models import Action, Observation, State
from src.rewards import compute_step_reward
from src.tasks import TaskConfig, get_task_config


class EmailTriageEnv:
    def __init__(self, task_id: str) -> None:
        self.task: TaskConfig = get_task_config(task_id)
        self.dataset = load_synthetic_email_dataset()
        self.grader = DeterministicTriageGrader(self.task)
        self.index = 0
        self.cumulative_reward = 0.0
        self.last_reward = 0.0
        self.done = False
        
        # Enhanced state tracking
        self.emails_processed = 0
        self.replies_sent = 0
        self.escalations = 0
        self.archived = 0
        self.urgent_handled = 0

    def reset(self) -> Observation:
        self.grader = DeterministicTriageGrader(self.task)
        self.index = 0
        self.cumulative_reward = 0.0
        self.last_reward = 0.0
        self.done = False
        
        # Reset enhanced tracking
        self.emails_processed = 0
        self.replies_sent = 0
        self.escalations = 0
        self.archived = 0
        self.urgent_handled = 0
        
        return self._observation_for_index(self.index)

    def state(self) -> State:
        return State(
            task_id=self.task.task_id,
            current_index=self.index,
            total_emails=len(self.dataset),
            cumulative_reward=self.cumulative_reward,
            last_reward=self.last_reward,
            done=self.done,
        )

    def step(self, action: Action | Dict[str, Any]) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        if self.done:
            raise RuntimeError("Environment is done. Call reset() before calling step() again.")

        if not isinstance(action, Action):
            action = Action.model_validate(action)

        current_email = self.dataset[self.index]
        reward_obj = compute_step_reward(action=action, truth=current_email, task=self.task)
        self.grader.update(action=action, truth=current_email)

        # Update tracking statistics
        self.emails_processed += 1
        if action.action and action.action.value == "reply":
            self.replies_sent += 1
        elif action.action and action.action.value == "escalate":
            self.escalations += 1
        elif action.action and action.action.value == "archive":
            self.archived += 1
        
        if action.priority and action.priority.value == "urgent":
            self.urgent_handled += 1

        self.last_reward = reward_obj.total
        self.cumulative_reward += reward_obj.total

        self.index += 1
        self.done = self.index >= len(self.dataset)

        next_observation = self._terminal_observation() if self.done else self._observation_for_index(self.index)

        info = {
            "task_id": self.task.task_id,
            "email_id": current_email.id,
            "truth": {
                "category": current_email.category.value,
                "priority": current_email.priority.value,
                "action": current_email.action.value,
                "reply_template": current_email.reply_template,
            },
            "reward_breakdown": reward_obj.model_dump(),
            "running_score": self.grader.score(),
            "state": self.state().model_dump(),
            "stats": {
                "emails_processed": self.emails_processed,
                "emails_remaining": len(self.dataset) - self.emails_processed,
                "replies_sent": self.replies_sent,
                "escalations": self.escalations,
                "archived": self.archived,
                "urgent_handled": self.urgent_handled,
            },
        }
        return next_observation, reward_obj.total, self.done, info

    def final_score(self) -> float:
        return self.grader.score()

    def _observation_for_index(self, index: int) -> Observation:
        email = self.dataset[index]
        return Observation(email_id=email.id, email_text=email.text, task_id=self.task.task_id)

    def _terminal_observation(self) -> Observation:
        return Observation(email_id="TERMINAL", email_text="", task_id=self.task.task_id)
