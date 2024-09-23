from model import predict_response

def chatbot():
    print("Chatbot: Hi! How can I help you today?")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("Chatbot: Goodbye!")
            break
        response = predict_response(user_input)
        print(f"Chatbot: {response}")

if __name__ == "__main__":
    chatbot()
