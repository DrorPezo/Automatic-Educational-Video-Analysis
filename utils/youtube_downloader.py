import pafy
from youtube_transcript_api import YouTubeTranscriptApi
import youtube_dl


def download_mp4_from_youtube(url):
    vPafy = pafy.new(url)
    best = vPafy.getbest(preftype='mp4')
    best.download()
    return vPafy.title


def download_video_transcript(url):
    vPafy = pafy.new(url)
    video_id = vPafy.videoid
    transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
    print(transcript)


