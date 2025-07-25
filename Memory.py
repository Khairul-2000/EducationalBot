chat_history = []

def format_chat_history(history):
    return "\n".join([f"{sender}: {message}" for sender, message in history])

def get_short_term_memory(history, limit=5):
    """Retrieve the most recent inputs in the chat sequence."""
    return history[-limit:]
