#!/usr/bin/env python3
"""
Test script to verify transcript extraction works with a single video
"""

from youtube_extractor import YouTubeExtractor
from rag_system import RAGSystem

def main():
    print("Testing single video transcript extraction")
    print("=" * 50)
    
    # Initialize systems
    extractor = YouTubeExtractor()
    rag = RAGSystem()
    
    # Test with videos that are known to have transcripts
    # Testing multiple video IDs
    test_videos = [
        "YUjzYBgXWzI",  # Khan Academy video (usually has transcripts)
        "3NjQ9b3pgIg",  # Crash Course video
        "airccYLambE",  # Stanford lecture
        "dQw4w9WgXcQ",  # Rick Roll (famous video, likely has transcripts)
    ]
    
    print("Testing multiple videos to find one with transcripts...")
    
    successful_video = None
    successful_transcript = None
    
    for video_id in test_videos:
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        print(f"\nTrying video: {video_url}")
        
        transcript = extractor.get_transcript(video_id)
        
        if transcript:
            print(f"✓ Successfully extracted transcript ({len(transcript)} characters)")
            print(f"Preview: {transcript[:200]}...")
            successful_video = video_id
            successful_transcript = transcript
            break
        else:
            print("✗ No transcript available")
    
    if successful_transcript:
        # Create a fake video info for testing
        fake_video_info = {
            'video_id': successful_video,
            'title': 'Test Video with Transcript',
            'url': f"https://www.youtube.com/watch?v={successful_video}",
            'transcript': successful_transcript
        }
        
        # Add to RAG system
        print("\n2. Adding to RAG system...")
        rag.add_transcripts([fake_video_info])
        
        # Test search
        print("\n3. Testing search...")
        results = rag.search("video", top_k=3)
        print(f"Found {len(results)} search results")
        
        for i, result in enumerate(results, 1):
            print(f"Result {i}: {result['text'][:100]}...")
        
        # Test Q&A
        print("\n4. Testing Q&A...")
        answer = rag.generate_answer("What is this video about?")
        print(f"Answer: {answer}")
        
        # Show stats
        stats = rag.get_stats()
        print(f"\n5. Database stats: {stats}")
        
    else:
        print("\n✗ Could not find any videos with transcripts")
        print("This might be due to:")
        print("- Videos having transcripts disabled")
        print("- Network connectivity issues")
        print("- YouTube API changes")
        print("\nThe application should still work with videos that do have transcripts available.")

if __name__ == "__main__":
    main()