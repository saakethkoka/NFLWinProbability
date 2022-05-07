import pickle
from sklearn.ensemble import RandomForestClassifier

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

def lambda_handler(event, context):
    print(event)
    score_diff = event["queryStringParameters"]["score_diff"]
    time_left = event["queryStringParameters"]["time_left"]
    print(model.predict_proba([[score_diff, time_left]]))
    return {
        'statusCode': 200,
        'headers': {
            'Access-Control-Allow-Origin': '*'
        },
        'body': model.predict_proba([[score_diff, time_left]])[0][0]
    }