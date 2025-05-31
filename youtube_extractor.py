import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi
import re
import time
import json
from typing import List, Dict, Optional

class YouTubeExtractor:
    def __init__(self):
        self.ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True,
            'ignoreerrors': True,
        }
    
    def detect_url_type(self, url: str) -> str:
        """Detect if URL is a video, playlist, or channel"""
        if 'watch?v=' in url:
            if 'list=' in url:
                return 'playlist'
            else:
                return 'video'
        elif 'playlist?list=' in url:
            return 'playlist'
        elif any(x in url for x in ['/channel/', '/c/', '/@', '/user/']):
            return 'channel'
        elif '/watch?v=' in url:
            return 'video'
        else:
            # Try to determine from yt-dlp
            try:
                with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                    info = ydl.extract_info(url, download=False, process=False)
                    if info.get('_type') == 'playlist':
                        return 'playlist'
                    elif info.get('_type') == 'video' or 'entries' not in info:
                        return 'video'
                    else:
                        return 'channel'
            except:
                return 'unknown'
    
    def extract_channel_id(self, channel_url: str) -> Optional[str]:
        """Extract channel ID from various YouTube channel URL formats"""
        patterns = [
            r'youtube\.com/channel/([^/?]+)',
            r'youtube\.com/c/([^/?]+)',
            r'youtube\.com/@([^/?]+)',
            r'youtube\.com/user/([^/?]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, channel_url)
            if match:
                return match.group(1)
        
        return None
    
    def get_single_video(self, video_url: str) -> Dict:
        """Extract info from a single video"""
        try:
            # Extract video ID from URL
            video_id = None
            if 'watch?v=' in video_url:
                video_id = video_url.split('watch?v=')[1].split('&')[0]
            elif 'youtu.be/' in video_url:
                video_id = video_url.split('youtu.be/')[1].split('?')[0]
            
            if not video_id:
                return {'success': False, 'error': 'Could not extract video ID from URL'}
            
            with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                video_info = ydl.extract_info(video_url, download=False)
                
                if not video_info:
                    return {'success': False, 'error': 'Could not extract video information'}
                
                video = {
                    'id': video_id,
                    'title': video_info.get('title', 'Unknown Title'),
                    'url': video_url,
                    'duration': video_info.get('duration'),
                    'upload_date': video_info.get('upload_date'),
                    'uploader': video_info.get('uploader', 'Unknown Channel')
                }
                
                return {
                    'success': True,
                    'content_type': 'video',
                    'title': video_info.get('title', 'Unknown Video'),
                    'videos': [video]
                }
                
        except Exception as e:
            return {'success': False, 'error': f'Error extracting video: {str(e)}'}
    
    def get_playlist_videos(self, playlist_url: str) -> Dict:
        """Extract all videos from a playlist"""
        try:
            with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                playlist_info = ydl.extract_info(playlist_url, download=False)
                
                if not playlist_info:
                    return {'success': False, 'error': 'Could not extract playlist information'}
                
                videos = []
                if 'entries' in playlist_info:
                    for entry in playlist_info['entries']:
                        if entry and 'id' in entry:
                            video_id = entry['id']
                            if len(video_id) == 11:  # YouTube video IDs are 11 characters
                                videos.append({
                                    'id': video_id,
                                    'title': entry.get('title', 'Unknown Title'),
                                    'url': f"https://www.youtube.com/watch?v={video_id}",
                                    'duration': entry.get('duration'),
                                    'upload_date': entry.get('upload_date'),
                                    'uploader': entry.get('uploader', 'Unknown Channel')
                                })
                
                return {
                    'success': True,
                    'content_type': 'playlist',
                    'title': playlist_info.get('title', 'Unknown Playlist'),
                    'uploader': playlist_info.get('uploader', 'Unknown Channel'),
                    'videos': videos
                }
                
        except Exception as e:
            return {'success': False, 'error': f'Error extracting playlist: {str(e)}'}
    
    def get_channel_videos(self, channel_url: str) -> Dict:
        """Get all videos from a YouTube channel"""
        try:
            # First get the channel's video playlist URL
            channel_videos_url = channel_url
            if not channel_videos_url.endswith('/videos'):
                channel_videos_url = channel_url.rstrip('/') + '/videos'
            
            with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                # Extract channel info and all videos
                channel_info = ydl.extract_info(channel_videos_url, download=False)
                
                if not channel_info:
                    return {'success': False, 'error': 'Could not extract channel information'}
                
                # Get all video entries
                videos = []
                if 'entries' in channel_info:
                    for entry in channel_info['entries']:
                        if entry and 'id' in entry and entry['id'] != channel_info.get('id'):
                            # Skip if this is the channel ID itself, not a video
                            video_id = entry['id']
                            if len(video_id) == 11:  # YouTube video IDs are 11 characters
                                videos.append({
                                    'id': video_id,
                                    'title': entry.get('title', 'Unknown Title'),
                                    'url': f"https://www.youtube.com/watch?v={video_id}",
                                    'duration': entry.get('duration'),
                                    'upload_date': entry.get('upload_date')
                                })
                
                return {
                    'success': True,
                    'content_type': 'channel',
                    'title': channel_info.get('title', 'Unknown Channel'),
                    'channel_name': channel_info.get('title', 'Unknown Channel'),
                    'channel_id': channel_info.get('id'),
                    'videos': videos
                }
                
        except Exception as e:
            return {'success': False, 'error': f'Error extracting channel videos: {str(e)}'}
    
    def get_content_videos(self, url: str) -> Dict:
        """Unified method to get videos from any YouTube URL (video, playlist, or channel)"""
        try:
            url_type = self.detect_url_type(url)
            
            if url_type == 'video':
                return self.get_single_video(url)
            elif url_type == 'playlist':
                return self.get_playlist_videos(url)
            elif url_type == 'channel':
                return self.get_channel_videos(url)
            else:
                return {'success': False, 'error': f'Unsupported or unrecognized URL type: {url_type}'}
                
        except Exception as e:
            return {'success': False, 'error': f'Error processing URL: {str(e)}'}
    
    def get_transcript(self, video_id: str) -> Optional[str]:
        """Get transcript for a single video"""
        try:
            # Try to get transcript
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            
            # Combine all transcript segments
            full_transcript = ""
            for segment in transcript_list:
                text = segment['text'].replace('\n', ' ').strip()
                if text:
                    full_transcript += text + " "
            
            return full_transcript.strip()
            
        except Exception as e:
            print(f"Could not get transcript for video {video_id}: {str(e)}")
            return None
    
    def extract_content_transcripts(self, url: str, max_videos: int = 100) -> Dict:
        """Extract transcripts from any YouTube URL (video, playlist, or channel)"""
        try:
            # Get content videos first
            content_result = self.get_content_videos(url)
            
            if not content_result['success']:
                return content_result
            
            videos = content_result['videos'][:max_videos]  # Limit number of videos
            transcripts = []
            
            content_type = content_result['content_type']
            content_title = content_result['title']
            
            print(f"Found {len(videos)} videos in {content_type}: {content_title}")
            print("Extracting transcripts...")
            
            for i, video in enumerate(videos):
                print(f"Processing video {i+1}/{len(videos)}: {video['title']}")
                
                transcript = self.get_transcript(video['id'])
                
                if transcript:
                    transcripts.append({
                        'video_id': video['id'],
                        'title': video['title'],
                        'url': video['url'],
                        'transcript': transcript,
                        'duration': video.get('duration'),
                        'upload_date': video.get('upload_date'),
                        'uploader': video.get('uploader', content_title)
                    })
                
                # Add small delay to be respectful to YouTube's servers
                time.sleep(0.5)
            
            print(f"Successfully extracted {len(transcripts)} transcripts out of {len(videos)} videos")
            
            return {
                'success': True,
                'content_type': content_type,
                'title': content_title,
                'transcripts': transcripts,
                'total_videos': len(videos),
                'successful_transcripts': len(transcripts)
            }
            
        except Exception as e:
            return {'success': False, 'error': f'Error extracting transcripts: {str(e)}'}

    def extract_channel_transcripts(self, channel_url: str, max_videos: int = 100) -> Dict:
        """Extract transcripts from all videos in a channel"""
        try:
            # Get channel videos
            channel_result = self.get_channel_videos(channel_url)
            
            if not channel_result['success']:
                return channel_result
            
            videos = channel_result['videos'][:max_videos]  # Limit number of videos
            transcripts = []
            
            print(f"Found {len(videos)} videos. Extracting transcripts...")
            
            for i, video in enumerate(videos):
                print(f"Processing video {i+1}/{len(videos)}: {video['title']}")
                
                transcript = self.get_transcript(video['id'])
                
                if transcript:
                    transcripts.append({
                        'video_id': video['id'],
                        'title': video['title'],
                        'url': video['url'],
                        'transcript': transcript,
                        'duration': video.get('duration'),
                        'upload_date': video.get('upload_date')
                    })
                
                # Add small delay to be respectful to YouTube's servers
                time.sleep(0.5)
            
            print(f"Successfully extracted {len(transcripts)} transcripts out of {len(videos)} videos")
            
            return {
                'success': True,
                'channel_name': channel_result['channel_name'],
                'channel_id': channel_result['channel_id'],
                'transcripts': transcripts,
                'total_videos': len(videos),
                'successful_transcripts': len(transcripts)
            }
            
        except Exception as e:
            return {'success': False, 'error': f'Error extracting channel transcripts: {str(e)}'}
    
    def save_transcripts_to_json(self, transcripts: List[Dict], filename: str):
        """Save transcripts to a JSON file"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(transcripts, f, ensure_ascii=False, indent=2)
            print(f"Transcripts saved to {filename}")
        except Exception as e:
            print(f"Error saving transcripts: {str(e)}")