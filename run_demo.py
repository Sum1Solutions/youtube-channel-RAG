#!/usr/bin/env python3
"""
Demo script to test YouTube Channel RAG functionality
"""

from youtube_extractor import YouTubeExtractor
from rag_system import RAGSystem

def main():
    print("YouTube Channel RAG Demo")
    print("=" * 50)
    
    # Initialize systems
    extractor = YouTubeExtractor()
    rag = RAGSystem()
    
    # Test channel URL - using a channel that likely has transcripts
    channel_url = "https://www.youtube.com/@3Blue1Brown"
    
    print(f"Testing with channel: {channel_url}")
    print("\n1. Extracting channel transcripts...")
    
    # Extract transcripts (limit to 5 videos for demo)
    result = extractor.extract_channel_transcripts(channel_url, max_videos=5)
    
    if result['success']:
        print(f"✓ Successfully extracted {len(result['transcripts'])} transcripts")
        print(f"  Channel: {result['channel_name']}")
        
        # Add to RAG system
        print("\n2. Adding transcripts to RAG system...")
        rag.add_transcripts(result['transcripts'])
        
        # Test search
        print("\n3. Testing search functionality...")
        search_query = "meditation"
        search_results = rag.search(search_query, top_k=3)
        
        print(f"Search query: '{search_query}'")
        print(f"Found {len(search_results)} results:")
        
        for i, result in enumerate(search_results, 1):
            print(f"\n  Result {i}:")
            print(f"    Video: {result['metadata']['title']}")
            print(f"    Score: {result['similarity_score']:.3f}")
            print(f"    Text: {result['text'][:200]}...")
        
        # Test Q&A
        print("\n4. Testing Q&A functionality...")
        question = "What is meditation about?"
        answer = rag.generate_answer(question)
        
        print(f"Question: {question}")
        print(f"Answer: {answer}")
        
        # Show stats
        stats = rag.get_stats()
        print(f"\n5. Database stats:")
        print(f"   Total chunks: {stats.get('total_chunks', 'Unknown')}")
        
    else:
        print(f"✗ Failed to extract transcripts: {result['error']}")

if __name__ == "__main__":
    main()