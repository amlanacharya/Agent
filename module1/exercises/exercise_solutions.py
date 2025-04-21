"""
Exercise Solutions for Module 1, Lesson 1
----------------------------------------
This file contains example solutions for the practice exercises.
Note: These are just one way to solve the exercises - there are many valid approaches!
"""

import time

class EnhancedAgent:
    def __init__(self):
        """Initialize the agent with state including conversation history"""
        self.state = {
            'user_name': None,
            'conversation_history': [],
            'commands': {
                'help': 'Show available commands',
                'name': 'Tell the agent your name',
                'clear': 'Clear conversation history',
                'history': 'Show conversation history'
            }
        }
    
    def sense(self, user_input):
        """
        Enhanced sense function that extracts more information from input
        
        Args:
            user_input (str): The raw input from the user
            
        Returns:
            dict: Processed input with metadata and intent
        """
        # Basic intent recognition
        intent = 'statement'  # default
        if user_input.endswith('?'):
            intent = 'question'
        elif any(cmd in user_input.lower() for cmd in self.state['commands']):
            intent = 'command'
            
        # Store in conversation history
        self.state['conversation_history'].append({
            'role': 'user',
            'content': user_input,
            'timestamp': time.time()
        })
            
        return {
            'text': user_input,
            'timestamp': time.time(),
            'intent': intent,
            'word_count': len(user_input.split()),
            'contains_greeting': any(word in user_input.lower() for word in ['hello', 'hi', 'hey'])
        }
    
    def think(self, processed_input):
        """
        Enhanced think function with more sophisticated response system
        
        Args:
            processed_input (dict): The processed input from the sense phase
            
        Returns:
            dict: Action plan with response type and content
        """
        # Check for commands first
        if processed_input['intent'] == 'command':
            if 'help' in processed_input['text'].lower():
                return self._handle_help_command()
            elif 'name' in processed_input['text'].lower():
                return self._handle_name_command(processed_input['text'])
            elif 'history' in processed_input['text'].lower():
                return self._handle_history_command()
            elif 'clear' in processed_input['text'].lower():
                return self._handle_clear_command()
        
        # Handle greetings
        if processed_input['contains_greeting']:
            if self.state['user_name']:
                return {
                    'response_type': 'greeting',
                    'content': f"Hello {self.state['user_name']}! How can I help you today?"
                }
            else:
                return {
                    'response_type': 'greeting',
                    'content': "Hello! I don't think we've met. You can tell me your name with the 'name' command."
                }
        
        # Handle questions
        if processed_input['intent'] == 'question':
            if "your name" in processed_input['text'].lower():
                return {
                    'response_type': 'information',
                    'content': "I'm SimpleAgent, your personal assistant!"
                }
            elif "time" in processed_input['text'].lower():
                return {
                    'response_type': 'information',
                    'content': f"The current time is {time.strftime('%H:%M:%S')}"
                }
            else:
                return {
                    'response_type': 'question',
                    'content': "That's an interesting question. I'm still learning, so I don't have a specific answer yet."
                }
        
        # Handle farewells
        if any(word in processed_input['text'].lower() for word in ['bye', 'goodbye', 'farewell']):
            name_str = f" {self.state['user_name']}" if self.state['user_name'] else ""
            return {
                'response_type': 'farewell',
                'content': f"Goodbye{name_str}! Have a great day!"
            }
        
        # Default response
        return {
            'response_type': 'default',
            'content': f"I received your message: {processed_input['text']}"
        }
    
    def _handle_help_command(self):
        """Handle the help command"""
        help_text = "Available commands:\n"
        for cmd, desc in self.state['commands'].items():
            help_text += f"- {cmd}: {desc}\n"
        return {
            'response_type': 'command_response',
            'content': help_text
        }
    
    def _handle_name_command(self, text):
        """Handle the name command"""
        # Extract name from command (e.g., "my name is John" -> "John")
        name_indicators = ["my name is", "i am", "call me"]
        for indicator in name_indicators:
            if indicator in text.lower():
                name = text.lower().split(indicator)[1].strip().title()
                self.state['user_name'] = name
                return {
                    'response_type': 'command_response',
                    'content': f"Nice to meet you, {name}! I'll remember your name."
                }
        
        return {
            'response_type': 'command_response',
            'content': "I didn't catch your name. Try saying 'My name is [your name]'."
        }
    
    def _handle_history_command(self):
        """Handle the history command"""
        if not self.state['conversation_history']:
            return {
                'response_type': 'command_response',
                'content': "No conversation history yet."
            }
        
        history_text = "Conversation history:\n"
        # Skip the last message which is the history command itself
        for i, msg in enumerate(self.state['conversation_history'][:-1]):
            history_text += f"{i+1}. {msg['role']}: {msg['content']}\n"
        
        return {
            'response_type': 'command_response',
            'content': history_text
        }
    
    def _handle_clear_command(self):
        """Handle the clear command"""
        self.state['conversation_history'] = []
        return {
            'response_type': 'command_response',
            'content': "Conversation history cleared."
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


# Example usage
if __name__ == "__main__":
    agent = EnhancedAgent()
    
    print("Enhanced Agent Demo (type 'exit' to quit)")
    print("Try commands like 'help', 'my name is [your name]', or ask questions")
    print("-" * 50)
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Exiting agent demo...")
            break
            
        response = agent.agent_loop(user_input)
        print(f"Agent: {response}")
