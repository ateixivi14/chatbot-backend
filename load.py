from ConversationalAgent import CboConversationalAgent

assistant = CboConversationalAgent('intents.json', model_name="model_1")
assistant.load_model()
done = False

# https://towardsdatascience.com/the-right-way-to-build-an-api-with-python-cd08ab285f8f

while not done:
    message = input()
    if message == "STOP":
        done = True
    else:
        assistant.request(message)