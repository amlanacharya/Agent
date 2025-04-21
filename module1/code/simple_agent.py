"""
Simple Agent Implementation
--------------------------
This file contains a basic implementation of an AI agent following the sense-think-act loop.
Use this as a starting point for the exercises in Module 1, Lesson 1.
"""

import time

class SimpleAgent:
    def __init__(self):
        """Initialize the agent with an empty state dictionary"""
        self.state = {}  # Internal memory
    
    def sense(self, user_input):
        """
        Process user input (perception phase)
        
        Args:
            user_input (str): The raw input from the user
            
        Returns:
            dict: Processed input with metadata
        """
        return {
            'text': user_input,
            'timestamp': time.time()
        }
    
    def think(self, processed_input):
        """
        Decide how to respond to the input (cognition phase)
        
        Args:
            processed_input (dict): The processed input from the sense phase
            
        Returns:
            dict: Action plan with response type and content
        """
        if "hello" in processed_input['text'].lower():
            return {
                'response_type': 'greeting',
                'content': 'Hello! How can I help you today?'
            }
        elif "bye" in processed_input['text'].lower():
            return {
                'response_type': 'farewell',
                'content': 'Goodbye! Have a great day!'
            }
        else:
            return {
                'response_type': 'default',
                'content': 'I received your message: ' + processed_input['text']
            }
    
    def act(self, action_plan):
        """
        Execute the planned action (execution phase)
        
        Args:
            action_plan (dict): The plan created in the think phase
            
        Returns:
            str: The response to be shown to the user
        """
        # Update internal state
        self.state['last_response'] = action_plan['content']
        self.state['last_response_type'] = action_plan['response_type']
        self.state['last_interaction_time'] = time.time()
        
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


# Example usage
if __name__ == "__main__":
    agent = SimpleAgent()
    
    print("Simple Agent Demo (type 'exit' to quit)")
    print("-" * 50)
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Exiting agent demo...")
            break
            
        response = agent.agent_loop(user_input)
        print(f"Agent: {response}")
