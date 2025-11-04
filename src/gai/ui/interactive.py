"""
Interactive UI components
"""

import sys
import subprocess
import tempfile
from typing import Optional
import click
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.prompt import Confirm, Prompt


console = Console()


def confirm(message: str, default: bool = False) -> bool:
    """
    Ask user for confirmation.

    Args:
        message: Question to ask
        default: Default value

    Returns:
        True if confirmed
    """
    return Confirm.ask(message, default=default)


def show_commit_message(message: str, title: str = "Generated Commit Message"):
    """
    Display commit message in a nice panel.

    Args:
        message: Commit message to display
        title: Panel title
    """
    console.print()
    console.print(Panel(
        message,
        title=title,
        border_style="green",
        padding=(1, 2)
    ))
    console.print()


def show_diff_summary(summary: str, title: str = "Diff Analysis"):
    """
    Display diff summary.

    Args:
        summary: Summary text
        title: Panel title
    """
    console.print()
    console.print(Panel(
        summary,
        title=title,
        border_style="blue",
        padding=(1, 2)
    ))
    console.print()


def show_merge_analysis(analysis: str, title: str = "Merge Analysis"):
    """
    Display merge analysis.

    Args:
        analysis: Analysis text
        title: Panel title
    """
    console.print()
    console.print(Panel(
        analysis,
        title=title,
        border_style="yellow",
        padding=(1, 2)
    ))
    console.print()


def show_error(message: str):
    """
    Display error message.

    Args:
        message: Error message
    """
    console.print()
    console.print(f"[bold red]Error:[/bold red] {message}")
    console.print()


def show_warning(message: str):
    """
    Display warning message.

    Args:
        message: Warning message
    """
    console.print()
    console.print(f"[bold yellow]Warning:[/bold yellow] {message}")
    console.print()


def show_info(message: str):
    """
    Display info message.

    Args:
        message: Info message
    """
    console.print(f"[cyan]{message}[/cyan]")


def show_success(message: str):
    """
    Display success message.

    Args:
        message: Success message
    """
    console.print(f"[bold green]✓[/bold green] {message}")


def edit_text(text: str, extension: str = "txt") -> Optional[str]:
    """
    Open text in editor for user to edit.

    Args:
        text: Initial text
        extension: File extension for syntax highlighting

    Returns:
        Edited text or None if cancelled
    """
    # Get editor from env
    editor = subprocess.run(
        ["git", "config", "--get", "core.editor"],
        capture_output=True,
        text=True
    ).stdout.strip() or "vim"

    # Create temp file
    with tempfile.NamedTemporaryFile(
        mode='w+',
        suffix=f'.{extension}',
        delete=False
    ) as f:
        f.write(text)
        temp_path = f.name

    try:
        # Open in editor
        subprocess.run([editor, temp_path], check=True)

        # Read edited content
        with open(temp_path, 'r') as f:
            edited = f.read().strip()

        return edited if edited else None

    except subprocess.CalledProcessError:
        show_error("Editor failed")
        return None

    finally:
        # Cleanup
        import os
        try:
            os.unlink(temp_path)
        except:
            pass


def select_option(options: list[tuple[str, str]], prompt: str = "Select:") -> Optional[str]:
    """
    Let user select from options with single keypress.
    Accepts both letter keys and numbers (1-indexed).

    Args:
        options: List of (key, description) tuples
        prompt: Prompt text

    Returns:
        Selected key or None
    """
    console.print()

    # Create number-to-key mapping
    number_map = {}
    for idx, (key, desc) in enumerate(options, start=1):
        number_map[str(idx)] = key
        console.print(f"  {idx}) [cyan]{key}[/cyan]: {desc}")
    console.print()

    valid_keys = [k for k, _ in options]
    console.print(f"[cyan]{prompt}[/cyan] ", end="")
    sys.stdout.flush()

    while True:
        try:
            char = click.getchar()

            # Handle Ctrl+C
            if char == '\x03':
                console.print()
                raise KeyboardInterrupt

            # Handle Enter/ESC (cancel)
            if char in ['\r', '\n', '\x1b']:
                console.print()
                return None

            # Check if number was pressed
            if char in number_map:
                selected_key = number_map[char]
                console.print(f"[bold cyan]{char} ({selected_key})[/bold cyan]")
                return selected_key

            # Check if valid letter key (case insensitive)
            if char.lower() in valid_keys:
                console.print(f"[bold cyan]{char.lower()}[/bold cyan]")
                return char.lower()

            # Invalid key - beep or ignore
            # Just continue waiting for valid input

        except (KeyboardInterrupt, EOFError):
            console.print()
            return None


def commit_confirm(message: str, allow_edit: bool = True) -> tuple[bool, Optional[str], bool]:
    """
    Show commit message and ask for confirmation with edit option.

    Args:
        message: Commit message
        allow_edit: Whether to allow editing

    Returns:
        (should_commit, final_message, should_push)
    """
    show_commit_message(message)

    if allow_edit:
        options = [
            ("y", "Yes, commit with this message"),
            ("p", "Push after commit"),
            ("n", "No, cancel"),
            ("e", "Edit message")
        ]
        choice = select_option(options, "Commit?")

        if choice == "y":
            return True, message, False
        elif choice == "p":
            return True, message, True
        elif choice == "e":
            edited = edit_text(message, "txt")
            if edited:
                return commit_confirm(edited, allow_edit=True)
            else:
                return False, None, False
        else:
            return False, None, False
    else:
        confirmed = confirm("Commit with this message?", default=True)
        return confirmed, message if confirmed else None, False
