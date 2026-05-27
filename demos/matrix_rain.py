import curses
import random
import time
import sys

# Constants
MAX_COLUMNS = 100
MIN_SPEED = 1
MAX_SPEED = 5
MIN_TRAIL_LENGTH = 3
MAX_TRAIL_LENGTH = 15
HEAD_CHAR = '█'
TAIL_CHAR = '·'
MAX_DELAY = 10

# Katakana range
KATAKANA_START = 0xFF66
KATAKANA_END = 0xFF9D

# Character pool for rain
CHAR_POOL = [
    chr(random.randint(KATAKANA_START, KATAKANA_END)),
    str(random.randint(0, 9)),
    '•', '•', '•', '•', '•', '•', '•', '•', '•', '•', '•', '•', '•', '•',
    '→', '←', '↑', '↓', '•', '•', '•', '•', '•', '•', '•', '•', '•', '•'
]

def get_random_char():
    # Sample a fresh character from the pool at runtime
    return random.choice([
        chr(random.randint(KATAKANA_START, KATAKANA_END)),
        str(random.randint(0, 9)),
        '•', '•', '•', '•', '•', '•', '•', '•', '•', '•', '•', '•', '•', '•',
        '→', '←', '↑', '↓', '•', '•', '•', '•', '•', '•', '•', '•', '•', '•'
    ])

def main(stdscr):
    # Hide cursor
    stdscr.nodelay(True)
    curses.curs_set(0)
    
    # Set up colors
    try:
        curses.start_color()
        curses.init_pair(1, curses.COLOR_WHITE, 0)  # Head: white
        curses.init_pair(2, curses.COLOR_GREEN, 0)  # Bright green
        curses.init_pair(3, curses.COLOR_GREEN, curses.COLOR_BLACK)  # Dim green
        curses.init_pair(4, curses.COLOR_BLACK, 0)  # Black
    except:
        # Fallback if color support is missing
        pass

    # Terminal dimensions
    height, width = stdscr.getmaxyx()
    if height < 10 or width < 10:
        stdscr.addstr(0, 0, "Terminal too small. Please use a larger terminal.", curses.A_BOLD)
        stdscr.refresh()
        time.sleep(1)
        return

    # Initialize columns
    columns = []
    for _ in range(random.randint(1, MAX_COLUMNS)):
        column = {
            'x': random.randint(0, width - 1),
            'y': -10,  # Start above screen
            'speed': random.randint(MIN_SPEED, MAX_SPEED),
            'trail': [],
            'start_delay': random.randint(0, MAX_DELAY),
            'last_update': time.time()
        }
        columns.append(column)

    # Main loop
    last_update = time.time()
    frame_time = 1.0 / 30.0  # Target ~30 FPS

    while True:
        try:
            # Handle resize
            current_height, current_width = stdscr.getmaxyx()
            if current_width != width or current_height != height:
                height, width = current_height, current_width
                # Recreate columns if needed
                columns = []
                for _ in range(random.randint(1, MAX_COLUMNS)):
                    column = {
                        'x': random.randint(0, width - 1),
                        'y': -10,
                        'speed': random.randint(MIN_SPEED, MAX_SPEED),
                        'trail': [],
                        'start_delay': random.randint(0, MAX_DELAY),
                        'last_update': time.time()
                    }
                    columns.append(column)

            # Update time
            now = time.time()
            dt = now - last_update
            if dt >= frame_time:
                last_update = now

                # Update columns
                for column in columns:
                    # Skip if not time to update
                    if now - column['last_update'] < 0.1:
                        continue

                    # Apply delay
                    if column['start_delay'] > 0:
                        column['start_delay'] -= dt
                        if column['start_delay'] <= 0:
                            # Start falling
                            column['y'] = -10
                            column['trail'] = []
                            column['last_update'] = now

                    # Move down
                    if column['y'] < height:
                        column['y'] += column['speed']

                    # If below screen, respawn at top
                    if column['y'] >= height:
                        column['y'] = -10
                        column['trail'] = []

                    # Generate new trail
                    if len(column['trail']) < MAX_TRAIL_LENGTH:
                        # Add new character at head
                        if len(column['trail']) == 0:
                            # Head is white
                            char = HEAD_CHAR
                        else:
                            # Tail is fading green
                            if len(column['trail']) < 3:
                                char = get_random_char()
                            else:
                                char = get_random_char()
                        column['trail'].append(char)

                    # Remove old trail entries
                    if len(column['trail']) > MAX_TRAIL_LENGTH:
                        column['trail'].pop(0)

                    # Update last update time
                    column['last_update'] = now

                # Redraw screen
                stdscr.clear()

                # Draw each column
                for column in columns:
                    if column['y'] < 0:
                        continue

                    # Draw trail with gradient
                    for i, char in enumerate(column['trail']):
                        y = int(column['y'] - i)
                        if y < 0:
                            continue
                        if y >= height:
                            continue

                        x = column['x']
                        if x < 0 or x >= width:
                            continue

                        # Determine color based on position in trail
                        if i == 0:
                            # Head: white
                            color_pair = 1
                        elif i < len(column['trail']) // 2:
                            # Bright green
                            color_pair = 2
                        else:
                            # Dim green
                            color_pair = 3

                        # Draw character
                        try:
                            stdscr.addstr(y, x, char, curses.color_pair(color_pair))
                        except:
                            pass

                # Refresh screen
                stdscr.refresh()

            # Check for quit
            if stdscr.getch() == ord('q'):
                break

        except KeyboardInterrupt:
            break

    # Clean exit
    stdscr.clear()
    stdscr.addstr(0, 0, "Exiting Matrix rain simulation...", curses.A_BOLD)
    stdscr.refresh()
    time.sleep(0.5)
    stdscr.clear()
    stdscr.refresh()
    curses.curs_set(1)

if __name__ == "__main__":
    try:
        curses.wrapper(main)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)