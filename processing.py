import requests
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from transformers import pipeline
import pandas as pd
import googleapiclient.discovery
from urllib.parse import urlparse, parse_qs
from datetime import datetime
from isodate import parse_duration

def get_video_id(url):
    parsed_url = urlparse(url)
    if parsed_url.hostname == 'youtu.be':
        return parsed_url.path[1:]
    elif parsed_url.hostname in ('www.youtube.com', 'youtube.com'):
        if parsed_url.path == '/watch':
            p = parse_qs(parsed_url.query)
            return p['v'][0] if 'v' in p else None
        elif parsed_url.path[:7] == '/embed/':
            return parsed_url.path.split('/')[2]
        elif parsed_url.path[:3] == '/v/':
            return parsed_url.path.split('/')[2]
    return None

def get_video_info(video_url, api_key, video_id):
    api_url = f'https://www.googleapis.com/youtube/v3/videos?id={video_id}&key={api_key}&part=snippet,contentDetails,statistics'
    response = requests.get(api_url)
    video_data = response.json()

    video_info = {
        'title': video_data['items'][0]['snippet']['title'],
        'channel_title': video_data['items'][0]['snippet']['channelTitle'],
        'upload_date': video_data['items'][0]['snippet']['publishedAt'],
        'duration': video_data['items'][0]['contentDetails']['duration'],
        'view_count': video_data['items'][0]['statistics']['viewCount'],
        'like_count': video_data['items'][0]['statistics'].get('likeCount', 0),
        'dislike_count': video_data['items'][0]['statistics'].get('dislikeCount', 0),
        'comment_count': video_data['items'][0]['statistics'].get('commentCount', 0)
    }
    return video_info

def get_youtube_comments(video_id, api_key,num_comments):
    api_service_name = "youtube"
    api_version = "v3"
    DEVELOPER_KEY = api_key
    youtube = googleapiclient.discovery.build( api_service_name, api_version, developerKey=DEVELOPER_KEY)
    Request = youtube.commentThreads().list(part="snippet", videoId=video_id, maxResults=100)
    comments = []

    response = Request.execute()
    for item in response['items']:
        comment = item['snippet']['topLevelComment']['snippet']
        public = item['snippet']['isPublic']
        comments.append([
            comment['authorDisplayName'],
            comment['publishedAt'],
            comment['likeCount'],
            comment['textOriginal'],
            public
        ])

    num = 1
    while (num < (num_comments % 100)):
        try:
            nextPageToken = response['nextPageToken']
            nextRequest = youtube.commentThreads().list(part="snippet", videoId=video_id, maxResults=100, pageToken=nextPageToken)
            response = nextRequest.execute()
            for item in response['items']:
                comment = item['snippet']['topLevelComment']['snippet']
                public = item['snippet']['isPublic']
                comments.append([
                    comment['authorDisplayName'],
                    comment['publishedAt'],
                    comment['likeCount'],
                    comment['textOriginal'],
                    public
                ])
            num += 1
        except KeyError:
            break

    df = pd.DataFrame(comments, columns=['author', 'updated_at', 'like_count', 'text', 'public'])
    df = df[1:]
    return df

def analyze_sentiment(df):
    model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    model = TFAutoModelForSequenceClassification.from_pretrained(model_name, from_pt=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

    column_text = df['text'].tolist()
    output = classifier(column_text)
    df_new = pd.DataFrame(output)
    result = pd.concat([df, df_new], axis=1)

    positive_comments = result[result['label'] == 'LABEL_2']['text'].tolist()
    neutral_comments = result[result['label'] == 'LABEL_1']['text'].tolist()
    negative_comments = result[result['label'] == 'LABEL_0']['text'].tolist()

    return positive_comments, neutral_comments, negative_comments

def process_video(video_url, num_comments, api_key):
    video_id = get_video_id(video_url)
    df = get_youtube_comments(video_id, api_key,num_comments)
    positive_comments, neutral_comments, negative_comments = analyze_sentiment(df)
    return positive_comments, neutral_comments, negative_comments

def process_video_info(video_url, api_key):
    video_id = get_video_id(video_url)
    video_info = get_video_info(video_url, api_key, video_id)

    upload_date_new = datetime.strptime(video_info['upload_date'], '%Y-%m-%dT%H:%M:%SZ')
    video_info['upload_date'] = upload_date_new.strftime('%Y-%m-%d %H:%M:%S')

    duration = parse_duration(video_info['duration'])
    total_seconds = duration.total_seconds()
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    formatted_duration = '{} hours, {} minutes, and {} seconds'.format(hours, minutes, seconds)
    video_info['duration'] = formatted_duration

    return video_info
