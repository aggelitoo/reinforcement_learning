def cycle_results_histogram(episodes_list):
    """
    Takes in an ordered (from first to last training cycle) episodes list where the length of the
    list denotes the number of training cycles that were needed to create it (each element in the
    list should be a list of episodes).
    
    In return, plots a histogram with sections colored according to the number of white wins, black
    wins, and draws (grey).

    As white should win at 6x6 othello, we expect to see the bars further to the right to be
    increasingly dominated by the white sections.
    """
    results = []
    for episodes in episodes_list:
        results.append(results_distribution(episodes))

    # Generate x-axis indices for the number of tuples in the list
    x = range(len(results))
    labels = list(range(1, len(results) + 1))

    # Unpack each tuple into separate segments
    segment1 = [t[0] for t in results]  # Black section
    segment2 = [t[1] for t in results]  # Grey section
    segment3 = [t[2] for t in results]  # White section

    plt.figure(figsize=(7.2, 5))

    # Plot the first segment (black)
    plt.bar(x, segment1, color='black', edgecolor='black')

    # Plot the second segment (grey), stacked on top of the first
    plt.bar(x, segment2, bottom=segment1, color='grey', edgecolor='black')

    # Compute the bottom for the third segment (segment1 + segment2)
    bottom3 = [a + b for a, b in zip(segment1, segment2)]

    # Plot the third segment (white)
    plt.bar(x, segment3, bottom=bottom3, color='white', edgecolor='black')

    # Add a text annotation above each bar showing the white percentage.
    for i, t in enumerate(results):
        total = sum(t)
        # Calculate the white percentage (t[2] is the white segment)
        white_percentage = (t[2] / total) * 100 if total else 0
        # Use .3g to format the number with a maximum of 3 significant digits
        plt.text(i, total + 2, f'{white_percentage:.3g}%', ha='center', va='bottom', fontsize=10)

    plt.xlabel('Training cycle')
    plt.ylabel('Episodes')
    plt.ylim(0, max([sum(t) for t in results]) * 1.1)  # Slightly higher than max for visual clarity
    plt.xticks(x, labels)
    plt.show()

def terminal_visits_histogram_colored(episodes):
    """
    Takes in a list of episodes and returns a histogram with as many bars as
    unique terminal states that were visited in those episodes, with each bar
    colored according to who wins (or draw) in that position. White bars for
    white wins, black bars for black wins, and grey for draws.
    """

    nr_games = len(episodes)
    win, draw, loss = results_distribution(episodes)
    w_rate, d_rate, l_rate = np.round(np.array([win, draw, loss])*(100/nr_games), decimals=0)

    terminal_states = {}
    for episode in episodes:
        key = (episode[0][0].tobytes(), episode[1])
        if key not in terminal_states:
            terminal_states[key] = 1
        else:
            terminal_states[key] += 1

    color_mapping = {
        -1: 'white',  # white for -1
        0: 'grey',   # grey for 0
        1: 'black'   # black for 1
    }

    # Extract rewards, visits, and colors
    identifiers = []
    heights = []
    colors = []
    for (identifier, y_value), height in terminal_states.items():
        if isinstance(identifier, bytes):
            identifier = identifier.decode('latin-1')
        identifiers.append(identifier)
        heights.append(height)
        colors.append(color_mapping[y_value])

    x = range(len(terminal_states))

    plt.figure(figsize=(7.2, 5))
    plt.bar(identifiers, heights, color=colors, edgecolor='black')

    # Add a text box with game results
    textstr = f"Total games = {nr_games}\nBlack wins: {w_rate}%\nDraws: {d_rate}%\nWhite wins: {l_rate}%"

    # Position the text box
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    plt.text(0.025, 0.975, textstr, transform=plt.gca().transAxes, fontsize=9,
            verticalalignment='top', bbox=props)


    plt.xticks(x, [str(i + 1) for i in x])

    plt.gca().yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    plt.xlabel('Unique terminal states')
    plt.ylabel('Visits')

    plt.show()