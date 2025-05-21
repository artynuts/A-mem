from memory_system import AgenticMemorySystem

def test_setup():
    # Initialize system with Llama2 backend
    memory_system = AgenticMemorySystem(
        llm_backend="ollama",
        llm_model="llama2"
    )

    # Add a test memory
    memory_id = memory_system.add_note(
        content="This is a test memory to verify the setup.",
        tags=["test", "setup"],
        category="Test"
    )

    # Search for the memory
    results = memory_system.search_agentic("test memory", k=1)
    
    if results:
        print("Setup test successful!")
        print(f"Found memory: {results[0]['content']}")
    else:
        print("Setup test failed - no memories found")

if __name__ == "__main__":
    test_setup()
