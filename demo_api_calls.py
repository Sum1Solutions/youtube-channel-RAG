#!/usr/bin/env python3
"""
Demonstrate the API calls that the web interface makes
"""

import requests
import json
import time

def test_preview_and_extract(url, content_type_name):
    """Test the preview and extract workflow"""
    print(f"\n🎯 Testing {content_type_name}: {url}")
    print("=" * 60)
    
    # Step 1: Preview Content
    print("📋 Step 1: Previewing content...")
    preview_response = requests.post(
        'http://localhost:5001/preview_content',
        json={'url': url},
        headers={'Content-Type': 'application/json'}
    )
    
    if preview_response.status_code == 200:
        preview_data = preview_response.json()
        if preview_data['success']:
            print(f"✅ Preview successful!")
            print(f"   📝 Type: {preview_data['content_type']}")
            print(f"   📺 Title: {preview_data['title']}")
            print(f"   🎬 Total Videos: {preview_data['total_videos']}")
            
            if preview_data['videos']:
                print(f"   📋 Sample Videos:")
                for i, video in enumerate(preview_data['videos'][:3], 1):
                    print(f"      {i}. {video['title']}")
            
            # Step 2: Extract Transcripts (limit to 2 videos for demo)
            print(f"\n⚡ Step 2: Extracting transcripts (limiting to 2 videos for demo)...")
            # Note: This would normally extract all videos, but we're limiting for demo
            print(f"   ⏳ This would process {min(preview_data['total_videos'], 2)} videos...")
            print(f"   🔄 Each video would be checked for transcript availability...")
            print(f"   ✅ Successfully processed videos would be added to the RAG system")
            print(f"   🔍 You could then search and ask questions about the content!")
            
        else:
            print(f"❌ Preview failed: {preview_data.get('error', 'Unknown error')}")
    else:
        print(f"❌ HTTP Error: {preview_response.status_code}")

def main():
    print("🎯 YouTube Content RAG - Complex Discussion Distillation Demo")
    print("=" * 70)
    print("🚀 MISSION: Rapidly distill complex video discussions into high-quality RAG")
    print("   Transform hours of content into instantly searchable knowledge")
    print("")
    print("This demonstrates the API calls made by the web interface")
    print("Visit http://localhost:5001 to use the actual web interface!")
    
    # Test different URL types
    test_cases = [
        {
            'url': 'https://www.youtube.com/watch?v=3NjQ9b3pgIg',
            'type': 'Single Video'
        },
        {
            'url': 'https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab',
            'type': 'Playlist'
        },
        {
            'url': 'https://www.youtube.com/@3Blue1Brown',
            'type': 'Channel'
        }
    ]
    
    for test_case in test_cases:
        test_preview_and_extract(test_case['url'], test_case['type'])
        time.sleep(1)  # Small delay between tests
    
    print(f"\n🌟 Complex Discussion Distillation Summary:")
    print(f"   ✅ ANY YouTube URL type supported (video/playlist/channel)")
    print(f"   ✅ Preview shows content scope before processing (e.g., '50 videos, 25 hours')")
    print(f"   ✅ Intelligent processing preserves conversational context")
    print(f"   ✅ Semantic search finds exact moments discussing specific concepts")
    print(f"   ✅ AI synthesis generates insights from hours of complex discussions")
    print(f"")
    print(f"💡 Perfect for: Podcasts • Educational Series • Conference Talks • Interview Archives")
    print(f"")
    print(f"🌐 Visit http://localhost:5001 to distill your complex video content!")

if __name__ == "__main__":
    main()