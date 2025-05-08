import json # For potential saving/loading later (optional)
import os
from typing import List
class TaskManager:
    """
    Manages a list of tasks for an agent, each with a status.
    """
    # --- Define Status Constants ---
    STATUS_PENDING = "pending"       # Task hasn't been attempted yet
    STATUS_ATTEMPTING = "attempting" # Task is currently being worked on
    STATUS_COMPLETED = "completed"   # Task was finished successfully
    STATUS_FAILED = "failed"         # Task attempt resulted in failure

    VALID_STATUSES = {
        STATUS_PENDING,
        STATUS_ATTEMPTING,
        STATUS_COMPLETED,
        STATUS_FAILED
    }
    # desktop path/todo.md
    file_path = os.path.join(os.path.expanduser("~"), "Desktop", "todo.txt")
    def __init__(self):
        """Initializes an empty task list."""
        self.tasks = []  # List to store task dictionaries
        self.task_completed = False 
        self.message = None
        self.start_execution = False
    # self.tasks is a property, make a getter and setter for it
    def set_tasks(self, tasks: List[str]):
        """
        Clears the current task list and sets new tasks from a list.
        Each task should be a string.

        Args:
            tasks: A list of strings, each representing a task.
        """
        try:
            self.tasks = []
            for task in tasks:
                if not isinstance(task, str) or not task.strip():
                    raise ValueError("Each task must be a non-empty string.")
                self.tasks.append({"description": task.strip(), "status": self.STATUS_PENDING})
            print(f"Tasks set: {len(self.tasks)} tasks added.")
            self.save_to_file()
        except Exception as e:
            print(f"Error setting tasks: {e}")

    def add_task(self, task_description: str):
        """
        Adds a new task to the list with a 'pending' status.

        Args:
            task_description: The string describing the task.

        Returns:
            int: The index of the newly added task.

        Raises:
            ValueError: If the task_description is empty or not a string.
        """
        if not isinstance(task_description, str) or not task_description.strip():
            raise ValueError("Task description must be a non-empty string.")

        task = {
            "description": task_description.strip(),
            "status": self.STATUS_PENDING
        }
        self.tasks.append(task)
        self.save_to_file()
        print(f"Task added: {task_description} (Status: {self.STATUS_PENDING})")
        
        return len(self.tasks) - 1 # Return the index of the new task

    def get_task(self, index: int):
        """
        Retrieves a specific task by its index.

        Args:
            index: The integer index of the task.

        Returns:
            dict: The task dictionary {'description': str, 'status': str}.

        Raises:
            IndexError: If the index is out of bounds.
        """
        if 0 <= index < len(self.tasks):
            return self.tasks[index]
        else:
            raise IndexError(f"Task index {index} out of bounds.")

    def get_all_tasks(self):
        """
        Returns a copy of the entire list of tasks.

        Returns:
            list[dict]: A list containing all task dictionaries.
                      Returns an empty list if no tasks exist.
        """
        return list(self.tasks) # Return a copy to prevent external modification

    def update_status(self, index: int, new_status: str):
        """
        Updates the status of a specific task.

        Args:
            index: The index of the task to update.
            new_status: The new status string (must be one of VALID_STATUSES).

        Raises:
            IndexError: If the index is out of bounds.
            ValueError: If the new_status is not a valid status.
        """
        if new_status not in self.VALID_STATUSES:
            raise ValueError(f"Invalid status '{new_status}'. Valid statuses are: {', '.join(self.VALID_STATUSES)}")

        # get_task will raise IndexError if index is invalid
        task = self.get_task(index)
        task["status"] = new_status
        self.save_to_file()
        # No need to re-assign task to self.tasks[index] as dictionaries are mutable

    def delete_task(self, index: int):
        """
        Deletes a task by its index.

        Args:
            index: The index of the task to delete.

        Raises:
            IndexError: If the index is out of bounds.
        """
        if 0 <= index < len(self.tasks):
            del self.tasks[index]
            self.save_to_file()
        else:
            raise IndexError(f"Task index {index} out of bounds.")

    def clear_tasks(self):
        """Removes all tasks from the list."""
        self.tasks = []
        print("All tasks cleared.")
        self.save_to_file()

    def get_tasks_by_status(self, status: str):
        """
        Filters and returns tasks matching a specific status.

        Args:
            status: The status string to filter by.

        Returns:
            list[dict]: A list of tasks matching the status.

        Raises:
            ValueError: If the status is not a valid status.
        """
        if status not in self.VALID_STATUSES:
             raise ValueError(f"Invalid status '{status}'. Valid statuses are: {', '.join(self.VALID_STATUSES)}")
        return [task for task in self.tasks if task["status"] == status]

    # --- Convenience methods for specific statuses ---
    def get_pending_tasks(self) -> list[dict]:
        return self.get_tasks_by_status(self.STATUS_PENDING)

    def get_attempting_task(self) -> dict | None:
        attempting_tasks = self.get_tasks_by_status(self.STATUS_ATTEMPTING)
        if attempting_tasks:
            return attempting_tasks[0]
        else:
            return None

    def get_completed_tasks(self) -> list[dict]:
        return self.get_tasks_by_status(self.STATUS_COMPLETED)

    def get_failed_tasks(self) -> dict | None:
        attempting_tasks = self.get_tasks_by_status(self.STATUS_FAILED)
        if attempting_tasks:
            return attempting_tasks[0]
        else:
            return None

    # --- Utility methods ---
    def __len__(self):
        """Returns the total number of tasks."""
        return len(self.tasks)

    def __str__(self):
        """Provides a user-friendly string representation of the task list."""
        if not self.tasks:
            return "Task List (empty)"

        output = "Task List:\n"
        output += "----------\n"
        for i, task in enumerate(self.tasks):
            output += f"{i}: [{task['status'].upper():<10}] {task['description']}\n"
        output += "----------"
        return output

    def __repr__(self):
        """Provides a developer-friendly representation."""
        return f"<TaskManager(task_count={len(self.tasks)}, completed={self.get_completed_tasks()}, attempting={self.get_attempting_task()}, pending={self.get_pending_tasks()})>"
    def save_to_file(self, filename=file_path):
        """Saves the current task list to a Markdown file."""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(str(self))
            #print(f"Tasks saved to {filename}.")
        except Exception as e:
            print(f"Error saving tasks to file: {e}")
    def completed(self, message: str):
        """
        Marks the goal as completed, use this whether the task completion was successful or on failure.
        This method should be called when the task is finished, regardless of the outcome.

        Args:
            message: The message to be logged.
        """
        self.task_completed = True
        self.message = message
        print(f"Goal completed: {message}")
    def start_agent(self):
        """Starts the sub-agent to perform the tasks if there are any tasks to perform.
Use this function after setting the tasks.
Args:
    None"""
        if len(self.tasks) == 0:
            print("No tasks to perform.")
            return
        self.start_execution = True



# --- Example Usage ---
if __name__ == "__main__":
    agent_tasks = TaskManager()

    # Add some tasks
    idx1 = agent_tasks.add_task("Research competitor pricing")
    idx2 = agent_tasks.add_task("Draft initial report outline")
    idx3 = agent_tasks.add_task("Schedule team meeting")
    idx4 = agent_tasks.add_task("Debug the login module")

    print(agent_tasks)
    print(f"Total tasks: {len(agent_tasks)}\n")

    # Access a task
    try:
        task_1 = agent_tasks.get_task(1)
        print(f"Task at index 1: {task_1}\n")
    except IndexError as e:
        print(e)

    # Update status
    try:
        print(f"Updating task {idx1} status...")
        agent_tasks.update_status(idx1, agent_tasks.STATUS_ATTEMPTING)
        print(f"Updating task {idx2} status...")
        agent_tasks.update_status(idx2, agent_tasks.STATUS_COMPLETED)
        print(f"Updating task {idx4} status...")
        agent_tasks.update_status(idx4, agent_tasks.STATUS_FAILED)
        # Try invalid status
        # agent_tasks.update_status(idx3, "on_hold") # This would raise ValueError
    except (IndexError, ValueError) as e:
        print(e)

    print("\nAfter updates:")
    print(agent_tasks)

    # Get specific status lists
    print("\nPending Tasks:")
    print(agent_tasks.get_pending_tasks())

    print("\nCompleted Tasks:")
    print(agent_tasks.get_completed_tasks())

    # Delete a task
    try:
        print(f"\nDeleting task at index {idx3} ('Schedule team meeting')...")
        agent_tasks.delete_task(idx3)
    except IndexError as e:
        print(e)

    print("\nAfter deletion:")
    print(agent_tasks)
    print(f"Total tasks: {len(agent_tasks)}\n")

    # Save and load (optional)
    # agent_tasks.save_to_file("my_agent_tasks.json")
    # agent_tasks.clear_tasks()
    # print(agent_tasks)
    # agent_tasks.load_from_file("my_agent_tasks.json")
    # print("\nAfter loading:")
    # print(agent_tasks)

    # Try accessing out of bounds
    try:
        agent_tasks.get_task(99)
    except IndexError as e:
        print(f"\nCaught expected error: {e}")