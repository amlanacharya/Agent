"""
Prompt Engineering Exercises
--------------------------
This file contains solutions for the prompt engineering exercises in Module 1, Lesson 2.
"""

from module1.code.prompt_template import PromptTemplate, PromptLibrary

class TaskParserExercise:
    """Solution for Exercise 1: Create a Task Parser Template"""

    def __init__(self):
        # Create a task parser template
        self.task_parser = PromptTemplate(
            "Extract detailed task information from the user input. Return a JSON object with all relevant fields.\n\n"
            "Example 1:\n"
            "Input: I need to call John tomorrow at 3pm about the project proposal\n"
            "Output: {'action': 'call', 'person': 'John', 'date': 'tomorrow', 'time': '3pm', 'topic': 'project proposal', 'priority': 'medium'}\n\n"
            "Example 2:\n"
            "Input: Remind me to buy groceries on Friday: milk, eggs, and bread\n"
            "Output: {'action': 'buy', 'items': ['milk', 'eggs', 'bread'], 'category': 'groceries', 'date': 'Friday', 'priority': 'medium'}\n\n"
            "Example 3:\n"
            "Input: Schedule an urgent meeting with the marketing team next Monday at 10am in the conference room\n"
            "Output: {'action': 'schedule', 'event_type': 'meeting', 'participants': 'marketing team', 'date': 'next Monday', 'time': '10am', 'location': 'conference room', 'priority': 'high'}\n\n"
            "Example 4:\n"
            "Input: I need to submit the quarterly report by the end of the month\n"
            "Output: {'action': 'submit', 'document': 'quarterly report', 'deadline': 'end of the month', 'priority': 'high'}\n\n"
            "Input: {input}\n"
            "Output:"
        )

    def parse_task(self, user_input):
        """
        Generate a prompt for task parsing

        Args:
            user_input (str): The user's task description

        Returns:
            str: The formatted prompt ready to be sent to an LLM
        """
        return self.task_parser.format(input=user_input)


class RoleBasedPromptsExercise:
    """Solution for Exercise 2: Implement Role-Based Prompts"""

    def __init__(self):
        self.roles = {
            "professional": {
                "description": "You are a professional assistant with a formal tone. You use precise language and maintain a respectful, business-like demeanor.",
                "examples": [
                    "I'll schedule that meeting for you right away.",
                    "Your task has been added to the system.",
                    "I've analyzed your calendar and found three potential time slots."
                ]
            },
            "friendly": {
                "description": "You are a friendly, casual assistant. You use conversational language, occasional emoji, and maintain a warm, approachable tone.",
                "examples": [
                    "Got it! I'll add that to your to-do list. 👍",
                    "Hey there! Looks like you've got a busy day ahead!",
                    "No problem at all, happy to help with that!"
                ]
            },
            "technical": {
                "description": "You are a technical assistant focused on precision and detail. You use technical terminology when appropriate and provide comprehensive information.",
                "examples": [
                    "I've parsed your request and identified the following parameters.",
                    "The system has successfully integrated your new task with priority level 2.",
                    "Based on dependency analysis, I recommend completing Task A before Task B."
                ]
            }
        }

        # Create role-based templates
        self.role_templates = {}
        for role_name, role_data in self.roles.items():
            self.role_templates[role_name] = PromptTemplate(
                f"{role_data['description']}\n\n"
                f"Examples of your response style:\n"
                f"- {role_data['examples'][0]}\n"
                f"- {role_data['examples'][1]}\n"
                f"- {role_data['examples'][2]}\n\n"
                f"User request: {{user_request}}\n\n"
                f"Respond to the user in your {role_name} style:"
            )

    def get_role_prompt(self, role_name, user_request):
        """
        Get a prompt for a specific role

        Args:
            role_name (str): The name of the role to use
            user_request (str): The user's request

        Returns:
            str: The formatted prompt

        Raises:
            ValueError: If the role doesn't exist
        """
        if role_name not in self.role_templates:
            raise ValueError(f"Role '{role_name}' not found. Available roles: {', '.join(self.role_templates.keys())}")

        return self.role_templates[role_name].format(user_request=user_request)


class ChainOfThoughtExercise:
    """Solution for Exercise 3: Chain-of-Thought Implementation"""

    def __init__(self):
        self.cot_template = PromptTemplate(
            "# Task Planning\n\n"
            "## Objective\n"
            "{objective}\n\n"
            "## Thinking Process\n"
            "Let's break down this task step by step:\n\n"
            "1. **Understand the goal**: What exactly needs to be accomplished?\n"
            "2. **Identify components**: What are the main parts or subtasks?\n"
            "3. **Resource assessment**: What resources (time, tools, information) are needed?\n"
            "4. **Dependencies**: What needs to happen first? Are there any blockers?\n"
            "5. **Timeline**: What's a realistic schedule for completion?\n"
            "6. **Potential challenges**: What might go wrong and how to mitigate?\n"
            "7. **Success criteria**: How will we know when the task is complete?\n\n"
            "## Task Details\n"
            "{task_description}\n\n"
            "## Step-by-Step Plan\n"
            "Based on the thinking process above, create a detailed plan with numbered steps, estimated time for each step, and any resources needed."
        )

        self.project_template = PromptTemplate(
            "# Project Planning: {project_name}\n\n"
            "## Project Overview\n"
            "{project_description}\n\n"
            "## Thinking Process\n"
            "Let's develop this project plan systematically:\n\n"
            "1. **Define scope**: What's included and what's not?\n"
            "2. **Identify stakeholders**: Who's involved or affected?\n"
            "3. **Break down deliverables**: What are the major components?\n"
            "4. **Resource planning**: What people, budget, and tools are needed?\n"
            "5. **Timeline development**: What are key milestones and deadlines?\n"
            "6. **Risk assessment**: What could go wrong and how to mitigate?\n"
            "7. **Communication plan**: How will progress be communicated?\n\n"
            "## Project Requirements\n"
            "- Deadline: {deadline}\n"
            "- Budget: {budget}\n"
            "- Team size: {team_size}\n\n"
            "## Detailed Project Plan\n"
            "Based on the thinking process above, create a comprehensive project plan with phases, tasks, timeline, resource allocation, and risk mitigation strategies."
        )

    def generate_task_plan(self, objective, task_description):
        """
        Generate a chain-of-thought prompt for task planning

        Args:
            objective (str): The main objective of the task
            task_description (str): Detailed description of the task

        Returns:
            str: The formatted prompt
        """
        return self.cot_template.format(
            objective=objective,
            task_description=task_description
        )

    def generate_project_plan(self, project_name, project_description, deadline, budget, team_size):
        """
        Generate a chain-of-thought prompt for project planning

        Args:
            project_name (str): The name of the project
            project_description (str): Description of the project
            deadline (str): Project deadline
            budget (str): Project budget
            team_size (str): Size of the team

        Returns:
            str: The formatted prompt
        """
        return self.project_template.format(
            project_name=project_name,
            project_description=project_description,
            deadline=deadline,
            budget=budget,
            team_size=team_size
        )


# Example usage
if __name__ == "__main__":
    print("=== Task Parser Exercise ===")
    task_parser = TaskParserExercise()
    prompt = task_parser.parse_task("I need to prepare for the client presentation next Thursday at 2pm and gather all the sales data")
    print(prompt)
    print("\n" + "-" * 50 + "\n")

    print("=== Role-Based Prompts Exercise ===")
    role_prompts = RoleBasedPromptsExercise()
    for role in ["professional", "friendly", "technical"]:
        prompt = role_prompts.get_role_prompt(role, "I need to reschedule my meeting with the team")
        print(f"[{role.upper()} ROLE]")
        print(prompt)
        print()
    print("-" * 50 + "\n")

    print("=== Chain-of-Thought Exercise ===")
    cot = ChainOfThoughtExercise()
    task_prompt = cot.generate_task_plan(
        objective="Organize a team-building event",
        task_description="Plan a one-day team-building event for a 15-person engineering team. The event should include activities that promote collaboration and problem-solving. Budget is $2000."
    )
    print(task_prompt)
    print("\n" + "-" * 50)
