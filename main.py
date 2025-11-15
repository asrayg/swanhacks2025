"""
Example usage of the JARVIS library
Demonstrates how to use JARVIS in your own projects
"""

from JARVIS import JARVIS, create_jarvis


def example_basic_usage():
    """Basic example: Simple Q&A"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Usage")
    print("="*60 + "\n")
    
    # Create JARVIS instance
    jarvis = create_jarvis()
    
    # Ask a simple question
    response = jarvis.ask("What is artificial intelligence?")
    print(f"Q: What is artificial intelligence?")
    print(f"A: {response}\n")


def example_with_text_file():
    """Example: Using a large text file as context"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Large Text File Context")
    print("="*60 + "\n")
    
    # Create JARVIS instance
    jarvis = JARVIS()
    
    # Add a large text file as context
    # For this example, let's create a sample file
    import pathlib
    sample_file = pathlib.Path("sample_context.txt")
    sample_file.write_text("""
    JARVIS (Just A Rather Very Intelligent System) is an AI assistant
    designed to help with various tasks. It uses advanced natural language
    processing and retrieval-augmented generation (RAG) to provide accurate
    and contextual responses.
    
    Key Features:
    - Large text file context windows
    - Vector database for efficient retrieval
    - Easy integration into any Python project
    - Support for PDF and TXT files
    - Multi-turn conversations
    
    Technical Details:
    JARVIS uses OpenAI's GPT models for generation and embeddings for
    semantic search. It stores documents in ChromaDB, a vector database
    that enables fast similarity search.
    """)
    
    # Load the file into JARVIS
    chunks_added = jarvis.add_context_from_file("sample_context.txt")
    print(f"Loaded {chunks_added} chunks from file\n")
    
    # Now ask questions about the file
    response = jarvis.ask("What are the key features of JARVIS?")
    print(f"Q: What are the key features of JARVIS?")
    print(f"A: {response}\n")
    
    response = jarvis.ask("What technical components does JARVIS use?")
    print(f"Q: What technical components does JARVIS use?")
    print(f"A: {response}\n")
    
    # Cleanup
    sample_file.unlink()


def example_with_direct_text():
    """Example: Adding text directly"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Direct Text Context")
    print("="*60 + "\n")
    
    jarvis = JARVIS()
    
    # Add context directly as text
    company_info = """
    TechCorp Inc. was founded in 2020 and specializes in AI solutions.
    Our flagship product is an intelligent assistant called JARVIS.
    We have offices in San Francisco, New York, and London.
    Our CEO is Jane Smith, and we employ over 500 people worldwide.
    """
    
    jarvis.add_context_from_text(company_info)
    
    # Ask questions about the context
    response = jarvis.ask("When was TechCorp founded?")
    print(f"Q: When was TechCorp founded?")
    print(f"A: {response}\n")
    
    response = jarvis.ask("Who is the CEO?")
    print(f"Q: Who is the CEO?")
    print(f"A: {response}\n")


def example_multi_turn_conversation():
    """Example: Multi-turn conversation"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Multi-turn Conversation")
    print("="*60 + "\n")
    
    jarvis = JARVIS()
    
    # Add some context
    jarvis.add_context_from_text("""
    Python is a high-level programming language created by Guido van Rossum.
    It was first released in 1991. Python emphasizes code readability and
    allows programmers to express concepts in fewer lines of code.
    Popular frameworks include Django for web development and TensorFlow
    for machine learning.
    """)
    
    # Have a conversation
    messages = [
        {"role": "system", "content": "You are a helpful programming tutor."},
        {"role": "user", "content": "Tell me about Python"},
    ]
    
    response = jarvis.chat(messages)
    print(f"User: Tell me about Python")
    print(f"JARVIS: {response}\n")
    
    # Continue the conversation
    messages.append({"role": "assistant", "content": response})
    messages.append({"role": "user", "content": "What frameworks are popular?"})
    
    response = jarvis.chat(messages)
    print(f"User: What frameworks are popular?")
    print(f"JARVIS: {response}\n")


def example_with_directory():
    """Example: Loading multiple files from a directory"""
    print("\n" + "="*60)
    print("EXAMPLE 5: Directory Loading")
    print("="*60 + "\n")
    
    jarvis = JARVIS()
    
    # Create a sample directory with files
    import pathlib
    data_dir = pathlib.Path("sample_data")
    data_dir.mkdir(exist_ok=True)
    
    (data_dir / "file1.txt").write_text("AI is transforming healthcare.")
    (data_dir / "file2.txt").write_text("Machine learning improves predictions.")
    (data_dir / "file3.txt").write_text("Neural networks mimic the human brain.")
    
    # Load all files from directory
    total_chunks = jarvis.add_context_from_directory("sample_data")
    print(f"Loaded {total_chunks} total chunks from directory\n")
    
    # Ask a question spanning multiple files
    response = jarvis.ask("How is AI being used?")
    print(f"Q: How is AI being used?")
    print(f"A: {response}\n")
    
    # Show stats
    stats = jarvis.get_stats()
    print(f"JARVIS Stats: {stats}\n")
    
    # Cleanup
    import shutil
    shutil.rmtree(data_dir)


def example_custom_configuration():
    """Example: Custom configuration"""
    print("\n" + "="*60)
    print("EXAMPLE 6: Custom Configuration")
    print("="*60 + "\n")
    
    # Create JARVIS with custom settings
    jarvis = JARVIS(
        model="gpt-4o-mini",
        temperature=0.7,  # More creative responses
        db_path="./custom_jarvis_db"
    )
    
    jarvis.add_context_from_text("The sky is blue during the day.")
    
    response = jarvis.ask(
        "Why is the sky blue?",
        system_prompt="You are a friendly science teacher. Explain things simply."
    )
    
    print(f"Q: Why is the sky blue?")
    print(f"A: {response}\n")
    
    # Clear context
    jarvis.clear_context()
    print("Context cleared!\n")


def interactive_mode():
    """Interactive JARVIS session"""
    print("\n" + "="*60)
    print("INTERACTIVE MODE")
    print("="*60)
    print("Type 'quit' to exit")
    print("Type 'load <filepath>' to load a document")
    print("Type 'stats' to see JARVIS statistics")
    print("Type 'clear' to clear context\n")
    
    jarvis = JARVIS()
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == 'quit':
                print("Goodbye! 👋")
                break
            
            elif user_input.lower().startswith('load '):
                filepath = user_input[5:].strip()
                try:
                    chunks = jarvis.add_context_from_file(filepath)
                    print(f"✅ Loaded {chunks} chunks from {filepath}\n")
                except Exception as e:
                    print(f"❌ Error loading file: {e}\n")
            
            elif user_input.lower() == 'stats':
                stats = jarvis.get_stats()
                print(f"\n📊 JARVIS Stats:")
                for key, value in stats.items():
                    print(f"  {key}: {value}")
                print()
            
            elif user_input.lower() == 'clear':
                jarvis.clear_context()
                print("🧹 Context cleared!\n")
            
            else:
                response = jarvis.ask(user_input)
                print(f"JARVIS: {response}\n")
        
        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"❌ Error: {e}\n")


def main():
    """Run all examples"""
    print("\n" + "="*60)
    print("🤖 JARVIS LIBRARY - Example Usage")
    print("="*60)
    
    # Run examples
    example_basic_usage()
    example_with_text_file()
    example_with_direct_text()
    example_multi_turn_conversation()
    example_with_directory()
    example_custom_configuration()
    
    # Ask if user wants interactive mode
    print("\n" + "="*60)
    try:
        choice = input("Would you like to try interactive mode? (y/n): ").strip().lower()
        if choice == 'y':
            interactive_mode()
    except:
        pass


if __name__ == "__main__":
    main()

