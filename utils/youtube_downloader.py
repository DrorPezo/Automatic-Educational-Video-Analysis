import pafy
from youtube_transcript_api import YouTubeTranscriptApi
import youtube_dl


def download_mp4_from_youtube(url):
    vPafy = pafy.new(url)
    best = vPafy.getbest(preftype='mp4')
    best.download()


def download_video_transcript(url):
    vPafy = pafy.new(url)
    video_id = vPafy.videoid
    transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
    print(transcript)


download_video_transcript('https://www.youtube.com/watch?v=4kiHsIaK9_w')
