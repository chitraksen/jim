import sys

from jim.play import play


def show_usage():
    """Display usage information."""
    print("Usage: uv run main.py [command] [game]")
    print("Commands:")
    print("  play            - play the game")


def entry(argv: list):
    """Main entry point that dispatches to the appropriate module based on command line arguments."""
    if len(argv) != 3:
        show_usage()
        sys.exit(1)

    command = argv[1].lower()
    game = argv[2].lower()

    if command == "play":
        play(game)
    else:
        print(f"Error: Unknown command '{command}'")
        show_usage()
        sys.exit(1)
