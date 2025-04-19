from agent_framework import AgentLearningFramework

def test_basic_agent():
    agent = AgentLearningFramework()

    # Test with a command input
    test_input = {
        'timestamp': '2023-01-01 10:00:00',
        'data': 'hello',
        'type': 'command'
    }

    result = agent.agent_loop(test_input)
    print(f"Agent response: {result}")

if __name__ == "__main__":
    test_basic_agent()