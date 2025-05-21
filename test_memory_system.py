import unittest
from memory_system import AgenticMemorySystem, MemoryNote
from datetime import datetime
import json

class TestAgenticMemorySystem(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test."""
        self.memory_system = AgenticMemorySystem(
            # model_name='sentence-transformers/paraphrase-mpnet-base-v2',
            model_name='all-MiniLM-L6-v2',
            llm_backend="ollama",  # Changed from 'openai' to 'ollama'
            llm_model="llama2"     # Changed from 'gpt-4o-mini' to 'llama2'
        )
        
    def test_create_memory(self):
        """Test creating a new memory with complete metadata."""
        content = "Test memory content"
        tags = ["test", "memory"]
        keywords = ["test", "content"]
        links = ["link1", "link2"]
        context = "Test context"
        category = "Test category"
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        
        memory_id = self.memory_system.add_note(
            content=content,
            tags=tags,
            keywords=keywords,
            links=links,
            context=context,
            category=category,
            timestamp=timestamp
        )
        
        # Verify memory was created
        self.assertIsNotNone(memory_id)
        memory = self.memory_system.read(memory_id)
        self.assertIsNotNone(memory)
        self.assertEqual(memory.content, content)
        self.assertEqual(memory.tags, tags)
        self.assertEqual(memory.keywords, keywords)
        self.assertEqual(memory.links, links)
        self.assertEqual(memory.context, context)
        self.assertEqual(memory.category, category)
        self.assertEqual(memory.timestamp, timestamp)
        
    def test_memory_metadata_persistence(self):
        """Test that memory metadata persists through ChromaDB storage and retrieval."""
        # Create a memory with complex metadata
        content = "Complex test memory"
        tags = ["test", "complex", "metadata"]
        keywords = ["test", "complex", "keywords"]
        links = ["link1", "link2", "link3"]
        context = "Complex test context"
        category = "Complex test category"
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        evolution_history = ["evolution1", "evolution2"]
        
        memory_id = self.memory_system.add_note(
            content=content,
            tags=tags,
            keywords=keywords,
            links=links,
            context=context,
            category=category,
            timestamp=timestamp,
            evolution_history=evolution_history
        )
        
        # Search for the memory using ChromaDB
        results = self.memory_system.search_agentic(content, k=1)
        self.assertGreater(len(results), 0)
        
        # Verify metadata in search results
        result = results[0]
        self.assertEqual(result['content'], content)
        self.assertEqual(result['tags'], tags)
        self.assertEqual(result['keywords'], keywords)
        self.assertEqual(result['context'], context)
        self.assertEqual(result['category'], category)
        
    def test_memory_update(self):
        """Test updating memory metadata through ChromaDB."""
        # Create initial memory
        content = "Initial content"
        memory_id = self.memory_system.add_note(content=content)
        
        # Update memory with new metadata
        new_content = "Updated content"
        new_tags = ["updated", "tags"]
        new_keywords = ["updated", "keywords"]
        new_context = "Updated context"
        
        success = self.memory_system.update(
            memory_id,
            content=new_content,
            tags=new_tags,
            keywords=new_keywords,
            context=new_context
        )
        
        self.assertTrue(success)
        
        # Verify updates in ChromaDB
        results = self.memory_system.search_agentic(new_content, k=1)
        self.assertGreater(len(results), 0)
        result = results[0]
        self.assertEqual(result['content'], new_content)
        self.assertEqual(result['tags'], new_tags)
        self.assertEqual(result['keywords'], new_keywords)
        self.assertEqual(result['context'], new_context)
        
    def test_memory_relationships(self):
        """Test memory relationships and linked memories."""
        # Create related memories
        content1 = "First memory"
        content2 = "Second memory"
        content3 = "Third memory"
        
        id1 = self.memory_system.add_note(content1)
        id2 = self.memory_system.add_note(content2)
        id3 = self.memory_system.add_note(content3)
        
        # Add relationships
        memory1 = self.memory_system.read(id1)
        memory2 = self.memory_system.read(id2)
        memory3 = self.memory_system.read(id3)
        
        memory1.links.append(id2)
        memory2.links.append(id1)
        memory2.links.append(id3)
        memory3.links.append(id2)
        
        # Update memories with relationships
        self.memory_system.update(id1, links=memory1.links)
        self.memory_system.update(id2, links=memory2.links)
        self.memory_system.update(id3, links=memory3.links)
        
        # Test relationship retrieval
        results = self.memory_system.search_agentic(content1, k=3)
        self.assertGreater(len(results), 0)
        
        # Verify relationships are maintained
        memory1_updated = self.memory_system.read(id1)
        self.assertIn(id2, memory1_updated.links)
        
    def test_memory_evolution(self):
        """Test memory evolution system with ChromaDB."""
        # Create related memories
        contents = [
            "Deep learning neural networks",
            "Neural network architectures",
            "Training deep neural networks"
        ]
        
        memory_ids = []
        for content in contents:
            memory_id = self.memory_system.add_note(content)
            memory_ids.append(memory_id)
            
        # Verify that memories have been properly evolved
        for memory_id in memory_ids:
            memory = self.memory_system.read(memory_id)
            self.assertIsNotNone(memory.tags)
            self.assertIsNotNone(memory.context)
            self.assertIsNotNone(memory.keywords)
            
        # Test evolution through search
        results = self.memory_system.search_agentic("neural networks", k=3)
        self.assertGreater(len(results), 0)
        
        # Verify evolution metadata
        for result in results:
            self.assertIsNotNone(result['tags'])
            self.assertIsNotNone(result['context'])
            self.assertIsNotNone(result['keywords'])
            
    def test_memory_deletion(self):
        """Test memory deletion from ChromaDB."""
        # Create and delete a memory
        content = "Memory to delete"
        memory_id = self.memory_system.add_note(content)
        
        # Verify memory exists
        memory = self.memory_system.read(memory_id)
        self.assertIsNotNone(memory)
        
        # Delete memory
        success = self.memory_system.delete(memory_id)
        self.assertTrue(success)
        
        # Verify deletion
        memory = self.memory_system.read(memory_id)
        self.assertIsNone(memory)
        
        # Verify memory is removed from ChromaDB
        results = self.memory_system.search_agentic(content, k=1)
        self.assertEqual(len(results), 0)
        
    def test_memory_consolidation(self):
        """Test memory consolidation with ChromaDB."""
        # Create multiple memories
        contents = [
            "Memory 1",
            "Memory 2",
            "Memory 3"
        ]
        
        for content in contents:
            self.memory_system.add_note(content)
            
        # Force consolidation
        self.memory_system.consolidate_memories()
        
        # Verify memories are still accessible
        for content in contents:
            results = self.memory_system.search_agentic(content, k=1)
            self.assertGreater(len(results), 0)
            self.assertEqual(results[0]['content'], content)
            
    def test_find_related_memories(self):
        """Test finding related memories."""
        # Create test memories
        contents = [
            "Python programming language",
            "Python data science",
            "Machine learning with Python",
            "Web development with JavaScript"
        ]
        
        for content in contents:
            self.memory_system.add_note(content)
            
        # Test finding related memories
        results = self.memory_system.find_related_memories("Python", k=2)
        self.assertGreater(len(results), 0)
        
    def test_find_related_memories_raw(self):
        """Test finding related memories with raw format."""
        # Create test memories
        contents = [
            "Python programming language",
            "Python data science",
            "Machine learning with Python"
        ]
        
        for content in contents:
            self.memory_system.add_note(content)
            
        # Test finding related memories in raw format
        results = self.memory_system.find_related_memories_raw("Python", k=2)
        self.assertIsNotNone(results)
        
    def test_process_memory(self):
        """Test memory processing and evolution."""
        # Create a test memory
        content = "Test memory for processing"
        memory_id = self.memory_system.add_note(content)
        
        # Get the memory
        memory = self.memory_system.read(memory_id)
        
        # Process the memory
        should_evolve, processed_memory = self.memory_system.process_memory(memory)
        
        # Verify processing results
        self.assertIsInstance(should_evolve, bool)
        self.assertIsInstance(processed_memory, MemoryNote)
        self.assertIsNotNone(processed_memory.tags)
        self.assertIsNotNone(processed_memory.context)
        self.assertIsNotNone(processed_memory.keywords)
        
    def test_research_assistant_scenario(self):
        """Test a real-world scenario where the system acts as a research assistant
        tracking papers and their relationships in AI/ML domain."""
        
        # Add several research paper summaries
        transformer_paper = self.memory_system.add_note(
            content="'Attention Is All You Need' introduces the Transformer architecture, "
                   "eliminating the need for recurrence and convolutions in sequence tasks. "
                   "Key innovations include multi-head self-attention mechanism.",
            tags=["deep learning", "transformers", "attention mechanism"],
            keywords=["transformer", "self-attention", "neural networks"],
            category="Research Paper",
            context="Fundamental paper that revolutionized NLP and became foundation for models like GPT and BERT"
        )
        
        bert_paper = self.memory_system.add_note(
            content="'BERT: Pre-training of Deep Bidirectional Transformers' demonstrates "
                   "how pre-training and fine-tuning can create powerful language models. "
                   "Introduces masked language modeling objective.",
            tags=["deep learning", "BERT", "pre-training"],
            keywords=["BERT", "transformers", "language modeling"],
            category="Research Paper",
            context="Built upon transformer architecture to create pre-trained models"
        )
        
        gpt_paper = self.memory_system.add_note(
            content="'Language Models are Few-Shot Learners' shows that scaling up language "
                   "models leads to better few-shot learning abilities without task-specific "
                   "fine-tuning.",
            tags=["deep learning", "GPT", "few-shot learning"],
            keywords=["GPT-3", "language models", "scaling"],
            category="Research Paper",
            context="Demonstrated emergent abilities in large language models"
        )
        
        # Create relationships between papers
        bert = self.memory_system.read(bert_paper)
        bert.links.append(transformer_paper)  # BERT builds on Transformer
        self.memory_system.update(bert_paper, links=bert.links)
        
        gpt = self.memory_system.read(gpt_paper)
        gpt.links.append(transformer_paper)  # GPT also builds on Transformer
        self.memory_system.update(gpt_paper, links=gpt.links)
        
        # Test finding related papers about transformers
        results = self.memory_system.search_agentic("transformer architecture applications", k=3)
        self.assertEqual(len(results), 3)  # Should find all three papers
        
        # Test finding specific implementation details
        attention_results = self.memory_system.search_agentic("attention mechanism implementation", k=1)
        self.assertIn("multi-head self-attention", attention_results[0]['content'])
        
        # Test finding papers that built upon transformers
        raw_results = self.memory_system.find_related_memories_raw("transformer", k=2)
        self.assertGreaterEqual(len(raw_results), 2)  # Should find at least BERT and GPT papers
        
        # Test evolution of knowledge
        # Add a new finding about transformers
        update_note = "Recent studies show transformers excel in computer vision tasks too."
        transformer_memory = self.memory_system.read(transformer_paper)
        transformer_memory.content += " " + update_note
        
        # Update the paper with new information
        self.memory_system.update(
            transformer_paper,
            content=transformer_memory.content,
            tags=transformer_memory.tags + ["computer vision"],
            keywords=transformer_memory.keywords + ["vision transformers"]
        )
        
        # Verify the knowledge evolution
        updated_results = self.memory_system.search_agentic("transformers in vision", k=1)
        self.assertIn("vision", updated_results[0]['content'])

    def test_automatic_metadata_generation(self):
        """Test that the system automatically generates metadata when only content is provided."""
        
        # Add a note with only content
        content = "The transformer architecture revolutionized natural language processing by introducing self-attention mechanisms that can process sequences in parallel."
        memory_id = self.memory_system.add_note(content=content)
        
        # Retrieve the memory and check that metadata was automatically generated
        memory = self.memory_system.read(memory_id)
        
        # Verify that the system generated these automatically
        self.assertIsNotNone(memory.tags)
        self.assertIsNotNone(memory.keywords)
        self.assertIsNotNone(memory.context)
        self.assertTrue(len(memory.tags) > 0, "Tags should be automatically generated")
        self.assertTrue(len(memory.keywords) > 0, "Keywords should be automatically generated")
        self.assertTrue(len(memory.context) > 0, "Context should be automatically generated")
        
        print("\nAutomatically generated metadata:")
        print(f"Tags: {memory.tags}")
        print(f"Keywords: {memory.keywords}")
        print(f"Context: {memory.context}")        # Add another related note to test automatic relationship detection
        second_content = "BERT models use bidirectional training of the transformer architecture to understand context better."
        second_id = self.memory_system.add_note(content=second_content)        # The system should automatically detect the relationship through content similarity
        results = self.memory_system.find_related_memories("BERT", k=1)
        self.assertGreater(len(results), 0)
        
        print("\nResults from find_related_memories:")
        print(json.dumps(results, indent=2))
        
        # Parse and check the result for BERT reference
        # Results are returned as a list of strings, each containing tab-separated metadata
        result_str = results[0]
        self.assertIn("BERT", result_str)

if __name__ == '__main__':
    unittest.main()
