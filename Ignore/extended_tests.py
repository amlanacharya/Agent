from agent_framework import AgentLearningFramework

def test_different_input_types():
    """Test how the agent handles different input types"""
    agent = AgentLearningFramework()
    
    # Test with a command input
    command_input = {
        'timestamp': '2023-01-01 10:00:00',
        'data': 'hello',
        'type': 'command'
    }
    
    # Test with a different input type
    observation_input = {
        'timestamp': '2023-01-01 10:05:00',
        'data': 'temperature is 72 degrees',
        'type': 'observation'
    }
    
    # Process both inputs
    command_result = agent.agent_loop(command_input)
    observation_result = agent.agent_loop(observation_input)
    
    print(f"Command response: {command_result}")
    print(f"Observation response: {observation_result}")
    
    # Check if the agent's state was updated
    print(f"Agent state after interactions: {agent.state}")

def test_agent_state_persistence():
    """Test if the agent maintains state between interactions"""
    agent = AgentLearningFramework()
    
    # First interaction
    input1 = {
        'timestamp': '2023-01-01 10:00:00',
        'data': 'set goal: learn python',
        'type': 'command'
    }
    
    # Second interaction
    input2 = {
        'timestamp': '2023-01-01 10:10:00',
        'data': 'what is my goal?',
        'type': 'command'
    }
    
    # Process both inputs sequentially
    result1 = agent.agent_loop(input1)
    print(f"First interaction: {result1}")
    print(f"Agent state after first interaction: {agent.state}")
    
    result2 = agent.agent_loop(input2)
    print(f"Second interaction: {result2}")
    print(f"Agent state after second interaction: {agent.state}")

def explore_agent_stages():
    """Explore the predefined stages in the agent framework"""
    agent = AgentLearningFramework()
    
    print("Agent Learning Framework Stages:")
    for stage_num, stage_name in agent.stages.items():
        print(f"Stage {stage_num}: {stage_name}")

if __name__ == "__main__":
    print("\n=== Testing Different Input Types ===")
    test_different_input_types()
    
    print("\n=== Testing State Persistence ===")
    test_agent_state_persistence()
    
    print("\n=== Exploring Agent Stages ===")
    explore_agent_stages()
