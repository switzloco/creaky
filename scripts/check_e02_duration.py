import json

data = json.load(open('checkpoints/e02_trained/training-book.log', encoding='utf-8'))
print("First event:", data[0].get("time"))
print("Last event:", data[-1].get("time"))
