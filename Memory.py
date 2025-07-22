chat_history = []

def format_chat_history(history):
    return "\n".join([f"{sender}: {message}" for sender, message in history])
