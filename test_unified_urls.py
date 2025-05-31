#!/usr/bin/env python3
"""
Test script to verify unified URL handling for videos, playlists, and channels
"""

from youtube_extractor import YouTubeExtractor

def main():
    print("Testing Unified YouTube URL Handler")
    print("=" * 50)
    
    extractor = YouTubeExtractor()
    
    # Test URLs
    test_urls = [
        {
            'type': 'Single Video',
            'url': 'https://www.youtube.com/watch?v=3NjQ9b3pgIg',
            'expected': 'video'
        },
        {
            'type': 'Channel',
            'url': 'https://www.youtube.com/@3Blue1Brown',
            'expected': 'channel'
        },
        {
            'type': 'Playlist',
            'url': 'https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab',
            'expected': 'playlist'
        }
    ]
    
    for test in test_urls:
        print(f"\n{test['type']} Test:")
        print(f"URL: {test['url']}")
        
        # Test URL detection
        detected_type = extractor.detect_url_type(test['url'])
        print(f"Detected type: {detected_type}")
        print(f"Expected type: {test['expected']}")
        print(f"✓ Detection correct: {detected_type == test['expected']}")
        
        # Test content extraction
        print("Getting content preview...")
        result = extractor.get_content_videos(test['url'])
        
        if result['success']:
            print(f"✓ Successfully extracted content")
            print(f"  Content type: {result['content_type']}")
            print(f"  Title: {result['title']}")
            print(f"  Videos found: {len(result['videos'])}")
            
            if result['videos']:
                print(f"  First video: {result['videos'][0]['title']}")
                
                # Test transcript extraction for first video
                print("Testing transcript extraction...")
                transcript = extractor.get_transcript(result['videos'][0]['id'])
                if transcript:
                    print(f"✓ Successfully extracted transcript ({len(transcript)} characters)")
                    print(f"  Preview: {transcript[:100]}...")
                else:
                    print("✗ No transcript available for this video")
        else:
            print(f"✗ Failed to extract content: {result['error']}")
        
        print("-" * 40)

if __name__ == "__main__":
    main()