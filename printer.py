PINK = '\033[95m'
BLUE = '\033[94m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BLACK = '\033[0m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'


def write(text, color=None):
    if color == 'pink':
        print(PINK + text + BLACK)
    elif color == 'blue':
        print(BLUE + text + BLACK)
    elif color == 'green':
        print(GREEN + text + BLACK)
    elif color == 'yellow':
        print(YELLOW + text + BLACK)
    elif color == 'red':
        print(RED + text + BLACK)
    elif color == 'black':
        print(BLACK + text + BLACK)
    elif color == 'bold':
        print(BOLD + text + BLACK)
    elif color == 'underline':
        print(UNDERLINE + text + BLACK)
    elif color is None:
        print(text)