# Define ANSI escape codes for text formatting
BOLD_START = "\033[1m"
BOLD_END = "\033[0m"


def print_messages(messages):
    """
    Print the current message history in a formatted way.

    Args:
        messages (list): List of message dictionaries with 'role' and 'content'
    """
    if not messages:
        print(f"\n{BOLD_START}No Messages{BOLD_END}")
        return

    print(f"\n{BOLD_START}Current Messages{BOLD_END} " + "-" * 65)
    for i, message in enumerate(messages):
        role = message["role"]
        content = message["content"]

        # Truncate long content to first 200 chars with ellipsis
        if len(content) > 200:
            content = content[:197] + "..."

        # Add extra formatting for better readability
        print(f"\n{BOLD_START}{i+1}. {role}{BOLD_END}:")

        # Indent the content
        for line in content.split("\n"):
            print(f"   {line}")

        # Add a separator between messages except after the last one
        if i < len(messages) - 1:
            print("   " + "-" * 50)


def print_user_prompt():
    """Print the user input prompt."""
    print(f"\n{BOLD_START}User{BOLD_END} " + "-" * 70)


def print_assistant_header(responding_to_tool=False):
    """Print the assistant header."""
    if responding_to_tool:
        print(f"\n{BOLD_START}Assistant (responding to tool){BOLD_END} " + "-" * 70)
    else:
        print(f"\n{BOLD_START}Assistant{BOLD_END} " + "-" * 70)


def print_tool_header():
    """Print the tool call header."""
    print(f"\n{BOLD_START}Tool Call{BOLD_END} " + "-" * 70)


def print_tool_details(tool_name, arguments, result=None):
    """Print tool call details."""
    import json

    print(f"Tool: {tool_name}")
    print(f"Arguments: {json.dumps(arguments, indent=2)}")
    if result:
        print(f"\nResult: {result}")
