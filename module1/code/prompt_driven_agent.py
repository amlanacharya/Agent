"""
Prompt-Driven Agent Implementation
--------------------------------
This file contains an implementation of an agent that uses prompt templates
to guide its behavior. Use this as a reference for Module 1, Lesson 2.
"""

import time
from prompt_template import PromptTemplate, PromptLibrary

class PromptDrivenAgent:
    def __init__(self):
        """Initialize the agent with prompt templates and state"""
        self.state = {
            'user_name': None,
            'conversation_history': [],
            'tasks': []
        }
        
        # Initialize prompt library
        self.prompt_library = PromptLibrary()
        
        # Add specialized templates
        self.prompt_library.add_template(
            "task_parser",
            PromptTemplate.from_examples(
                instructions="Extract task details from the user input.",
                examples=[
                    {
                        "input": "I need to call John tomorrow at 3pm",
                        "output": "{'action': 'call', 'person': 'John', 'date': 'tomorrow', 'time': '3pm'}"
                    },
                    {
                        "input": "Remind me to buy groceries on Friday",
                        "output": "{'action': 'buy', 'item': 'groceries', 'date': 'Friday'}"
                    }
                ]
            )
        )
        
        # Add chain-of-thought template
        self.prompt_library.add_template(
            "cot_planning",
            PromptTemplate(
                "To plan this task, let's think step by step:\n\n"
                "1. What is the main objective? {objective}\n"
                "2. What are the key components or subtasks?\n"
                "3. What resources are needed?\n"
                "4. What is the timeline?\n"
                "5. Are there any dependencies between subtasks?\n\n"
                "Now, create a plan for: {task_description}"
            )
        )
    
    def sense(self, user_input):
        """
        Process user input using prompt templates
        
        Args:
            user_input (str): The raw input from the user
            
        Returns:
            dict: Processed input with metadata
        """
        # Store in conversation history
        self.state['conversation_history'].append({
            'role': 'user',
            'content': user_input,
            'timestamp': time.time()
        })
        
        # Determine intent (in a real system, this would use an LLM with a prompt)
        intent = 'task_creation' if any(word in user_input.lower() for word in ['create', 'add', 'new', 'schedule']) else 'general'
        
        return {
            'text': user_input,
            'timestamp': time.time(),
            'intent': intent
        }
    
    def think(self, processed_input):
        """
        Decide how to respond using appropriate prompt templates
        
        Args:
            processed_input (dict): The processed input from the sense phase
            
        Returns:
            dict: Action plan with response type and content
        """
        if processed_input['intent'] == 'task_creation':
            # Use task parser template to extract details
            # In a real system, this would be sent to an LLM
            task_details = self._simulate_task_parsing(processed_input['text'])
            
            # Add task to state
            self.state['tasks'].append(task_details)
            
            return {
                'response_type': 'task_confirmation',
                'content': f"I've added a new task: {task_details['description']}",
                'task': task_details
            }
        else:
            # For general queries, use appropriate templates based on content
            if "help" in processed_input['text'].lower():
                return {
                    'response_type': 'help',
                    'content': "I can help you manage tasks. Try saying things like:\n- Add a new task: Call mom tomorrow\n- Show my tasks\n- What's due today?"
                }
            elif "show" in processed_input['text'].lower() or "list" in processed_input['text'].lower():
                tasks_list = "\n".join([f"- {task['description']}" for task in self.state['tasks']])
                return {
                    'response_type': 'task_list',
                    'content': f"Here are your tasks:\n{tasks_list if tasks_list else 'No tasks yet!'}"
                }
            else:
                return {
                    'response_type': 'default',
                    'content': "I'm not sure how to help with that. Try asking for 'help' to see what I can do."
                }
    
    def act(self, action_plan):
        """
        Execute the planned action and update state
        
        Args:
            action_plan (dict): The plan created in the think phase
            
        Returns:
            str: The response to be shown to the user
        """
        # Update internal state
        self.state['last_response'] = action_plan['content']
        self.state['last_response_type'] = action_plan['response_type']
        self.state['last_interaction_time'] = time.time()
        
        # Add to conversation history
        self.state['conversation_history'].append({
            'role': 'agent',
            'content': action_plan['content'],
            'timestamp': time.time(),
            'type': action_plan['response_type']
        })
        
        # Return the response to be shown to the user
        return action_plan['content']
    
    def agent_loop(self, user_input):
        """
        Run the full sense-think-act cycle
        
        Args:
            user_input (str): Raw input from the user
            
        Returns:
            str: The agent's response
        """
        sensed_data = self.sense(user_input)
        action_plan = self.think(sensed_data)
        response = self.act(action_plan)
        return response
    
    def _simulate_task_parsing(self, text):
        """
        Simulate task parsing (in a real system, this would use an LLM)
        
        Args:
            text (str): The text to parse
            
        Returns:
            dict: Extracted task details
        """
        # This is a simplified simulation - a real implementation would use an LLM
        description = text
        
        # Extract date if present
        date = "today"
        if "tomorrow" in text.lower():
            date = "tomorrow"
        elif "next week" in text.lower():
            date = "next week"
        
        # Extract priority if present
        priority = "medium"
        if "urgent" in text.lower() or "important" in text.lower():
            priority = "high"
        elif "low priority" in text.lower() or "when you can" in text.lower():
            priority = "low"
        
        return {
            'description': description,
            'date': date,
            'priority': priority,
            'created_at': time.time()
        }


# Example usage
if __name__ == "__main__":
    agent = PromptDrivenAgent()
    
    print("Prompt-Driven Agent Demo (type 'exit' to quit)")
    print("Try commands like 'add a new task: call mom tomorrow' or 'show my tasks'")
    print("-" * 50)
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Exiting agent demo...")
            break
            
        response = agent.agent_loop(user_input)
        print(f"Agent: {response}")
