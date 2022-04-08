from ConversationalAgent import CboConversationalAgent

assistant = CboConversationalAgent('intents.json', model_name="model_LSTM")
assistant.train_model()
assistant.save_model()

done = False

while not done:
    message = input()
    if message == "STOP":
        done = True
    else:
        assistant.request(message)