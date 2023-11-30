from flask import Flask,render_template,url_for
import requests
import pyjokes
import json

def get_meme():
    #Uncomment these two lines and comment out the other url line if you want to use a specific meme subreddt
    # sr = "/wholesomememes"
    # url = "<https://meme-api.herokuapp.com/gimme>" + sr
    url = "https://meme-api.com/gimme"
    response = json.loads(requests.request("GET", url).text)
    meme_large = response['preview'][0]
    subreddit = response['subreddit']
    return meme_large, subreddit
   
def getquote():
    url = "https://api.quotable.io/random"
    response = json.loads(requests.request("GET", url).text)
    quote = response['content']
    author = response['author']
    return quote, author

def get_fact():
    url = "https://useless-facts.sameerkumar.website/api"
    response = json.loads(requests.request("GET", url).text)
    fact = response['data']
    return fact



app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('layout.html')

@app.route('/jokes')
def api():
    joke = pyjokes.get_joke()
    return render_template('api.html',joke=joke)


@app.route('/memes')
def memes():
   meme_pic,subreddit = get_meme()
   return render_template('memes.html',meme_pic=meme_pic,subreddit=subreddit)

@app.route('/newjoke')
def newjoke():
    ran_joke = pyjokes.get_joke()
    return ran_joke

@app.route('/quote')
def quote():
    quote,author = getquote()
    return render_template('quote.html',quote=quote,author=author)

@app.route('/newquote')
def newquote():
    quote,author = getquote()
    return quote,author


@app.route('/fact')
def fact():
    fact = get_fact()
    return render_template('fact.html',fact=fact)

@app.route('/newfact')
def newfact():
    fact = get_fact()
    return fact



# @app.route('/newmeme')
# def newmeme():
#    meme_pic,subreddit = get_meme()
#    return meme_pic,subreddit

if __name__ == '__main__':
    app.run(debug=True,port=8000)
