from ConversationalAgent import CboConversationalAgent


def get_chatbot_response(message):
    assistant = CboConversationalAgent('intents.json', model_name="test_model")
    assistant.load_model()

    response = assistant.request(message)

    return response
