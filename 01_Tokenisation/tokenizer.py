import sys
def tokenize_text(text):
    # Replace spaces with newline characters
    text = text.replace(' ', '\n')

    # Define a list of punctuation characters to add spaces before and after
    punctuation_to_space = [',', ':', '(', ')', '"','\'']

    # Iterate through each punctuation character and add spaces
    for char in punctuation_to_space:
        text = text.replace(char, f'{char} ')
    text = text.replace(' ', '\n')
    print(text)

if __name__ == "__main__":
    input_text = sys.stdin.read()

    
    tokenized_text = tokenize_text(input_text)

    print(tokenized_text)
