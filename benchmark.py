import pandas as pd
import random
from transformers import pipeline
import plotly.express as px

print("Generating comprehensive business call evaluation dataset...")

business_call_scenarios = {
    "admiration": ["Your customer success team has been absolutely incredible throughout this rollout."],
    "amusement": ["Haha, that was a funny misunderstanding, I thought the server was down but it was unplugged!"],
    "anger": ["This is completely unacceptable! Your API outage has been costing us thousands every hour!"],
    "annoyance": ["The interface keeps refreshing every time I try to save a layout, it's getting irritating."],
    "approval": ["Yes, that resolution plan sounds perfectly reasonable to me."],
    "caring": ["Please take your time looking into the logs, I know your team is under a lot of pressure today."],
    "confusion": ["Wait, if the database connection failed, why is the frontend reporting a status 200?"],
    "curiosity": ["What kind of throughput optimization can we expect if we upgrade to the premium tier?"],
    "desire": ["Our team desperately needs a tool that can orchestrate multiple agents at once."],
    "disappointment": ["I was really hoping this update would fix the background processing bottleneck, but it didn't."],
    "disapproval": ["I don't think deprecating that API version so quickly is a smart business decision."],
    "disgust": ["Ugh, the messy code structure in this third-party SDK is absolutely repulsive."],
    "embarrassment": ["Oh no, I realize now that I was running the test script against our production database, my bad."],
    "excitement": ["Wow, our throughput just doubled after implementing your microservices strategy!"],
    "fear": ["I am terrified that a security breach might happen if we leave this port open."],
    "gratitude": ["Thank you so much for hopping on a quick call to debug this with me, you saved my day."],
    "grief": ["Our business took a massive, devastating blow when that unbacked database corrupted."],
    "joy": ["Everything is running so smoothly now, our whole team is incredibly happy!"],
    "love": ["I absolutely love working with your development platform, it makes my life so easy."],
    "nervousness": ["My hand is shaking a bit as I hit deploy, I really hope this doesn't drop connections."],
    "optimism": ["I am confident that this new microservice paradigm will solve our long-term scaling issues."],
    "pride": ["Our engineering squad built this entire automated pipeline in less than forty-eight hours."],
    "realization": ["Ah, I see what's happening now! The environment variable wasn't getting injected properly."],
    "relief": ["Whew, thank goodness the database backup restored perfectly without any data loss."],
    "remorse": ["I am deeply sorry for causing that service disruption by pushing the unverified code branch."],
    "sadness": ["It's really disheartening to see our deployment fail after spending weeks preparing it."],
    "surprise": ["Oh wow! I didn't expect the server response time to drop to five milliseconds!"],
    "neutral": ["The current server deployment is running on version two dot four dot zero."]
}

expanded_rows = []
for _ in range(300):
    emotion = random.choice(list(business_call_scenarios.keys()))
    text = random.choice(business_call_scenarios[emotion])
    expanded_rows.append({"Emotion": emotion, "Text": text})

test_df = pd.DataFrame(expanded_rows)
print(f"Generated dataset with {len(test_df)} rows.")

print("Loading RoBERTa (SamLowe/roberta-base-go_emotions)...")
roberta_model = pipeline('text-classification', model='SamLowe/roberta-base-go_emotions')

print("Loading BERT (bhadresh-savani/bert-base-uncased-emotion)...")
bert_model = pipeline('text-classification', model='bhadresh-savani/bert-base-uncased-emotion')

def get_roberta_emotion(text):
    try: return roberta_model(text)[0]['label']
    except: return "error"

def get_bert_emotion(text):
    try: return bert_model(text)[0]['label']
    except: return "error"

print("Running RoBERTa inference...")
test_df['RoBERTa_Prediction'] = test_df['Text'].apply(get_roberta_emotion)

print("Running BERT inference...")
test_df['BERT_Prediction'] = test_df['Text'].apply(get_bert_emotion)

agreement_rate = (test_df['RoBERTa_Prediction'] == test_df['BERT_Prediction']).mean() * 100
print(f"Model Agreement: BERT and RoBERTa predicted the exact same label {agreement_rate:.2f}% of the time.")

melted_df = test_df.melt(
    id_vars=['Text', 'Emotion'], 
    value_vars=['RoBERTa_Prediction', 'BERT_Prediction'],
    var_name='Model', 
    value_name='Predicted_Emotion'
)

fig = px.histogram(
    melted_df, x='Predicted_Emotion', color='Model', barmode='group',
    title='Comparison of Predicted Emotion Distribution: BERT vs RoBERTa'
)
fig.write_html("comparison.html")
print("Interactive visualization saved successfully as 'comparison.html'!")

test_df.to_csv("business_call_evaluation_results.csv", index=False)
print("Raw text classifications saved to 'business_call_evaluation_results.csv'!")