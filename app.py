from flask import Flask, render_template, request
import pandas as pd
from datetime import datetime
from isodate import parse_duration
import processing

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('test.html')

@app.route('/result', methods=['POST'])
def result():
    action = request.form['action']
    
    if action == 'action1':
        video_url = request.form['video_url']
        num_comments = int(request.form['num_comments'])
        api_key = "Api - Key"

        positive_comments, neutral_comments, negative_comments = processing.process_video(video_url, num_comments, api_key)

        return render_template('result.html', positive_comments=positive_comments, neutral_comments=neutral_comments, negative_comments=negative_comments)
    
    elif action == 'action2':
        video_url = request.form['video_url']
        api_key = "Api - Key"

        video_info = processing.process_video_info(video_url, api_key)

        return render_template('info.html', info=video_info)

if __name__ == '__main__':
    app.run(debug=True)
