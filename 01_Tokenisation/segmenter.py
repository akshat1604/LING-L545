import re
import sys
def segment_text(text):
    # Define a regular expression pattern to match full stop followed by a space
    pattern = r'\. '

    # Replace the matched pattern with a full stop and a newline character
    segmented_text = re.sub(pattern, '.\n', text)

    return segmented_text

if __name__ == "__main__":
    input_text = sys.stdin.read()


    segmented_text = segment_text(input_text)

    print(segmented_text)
