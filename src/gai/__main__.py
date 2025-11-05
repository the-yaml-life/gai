#!/usr/bin/env python3
"""
gai - Git AI Assistant

AI-powered git commands for commit messages, merge previews, and more.
"""

import click
import os
import sys
from pathlib import Path

from gai.core.config import get_config
from gai.core.git import Git, GitError
from gai.core.stats import get_stats


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.option('--config', '-c', type=click.Path(), help='Config file path')
@click.pass_context
def cli(ctx, verbose, config):
    """gai - Git AI Assistant"""
    ctx.ensure_object(dict)

    # Init command doesn't need config validation
    if ctx.invoked_subcommand == 'init':
        return

    try:
        # Load config
        cfg = get_config()
        if verbose:
            cfg.set('general.verbose', True)

        ctx.obj['config'] = cfg
        ctx.obj['git'] = Git()

        # Initialize stats with config (will use DB backend if configured)
        get_stats(cfg)

        # Check if in git repo (except for commands that don't need it)
        if ctx.invoked_subcommand not in ['models', 'config-show']:
            if not ctx.obj['git'].is_repo():
                click.echo("Error: Not a git repository", err=True)
                sys.exit(1)

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--auto', is_flag=True, help='Auto-commit without confirmation')
@click.option('--amend', is_flag=True, help='Amend last commit')
@click.option('--context', '-m', help='Additional context for generation')
@click.option('--breaking', is_flag=True, help='Mark as breaking change')
@click.option('--dry-run', is_flag=True, help='Show message without committing')
@click.pass_context
def commit(ctx, auto, amend, context, breaking, dry_run):
    """Auto-generate commit message"""
    from gai.commands.commit import CommitCommand

    verbose = ctx.obj['config'].get('general.verbose', False)
    cmd = CommitCommand(
        config=ctx.obj['config'],
        git=ctx.obj['git'],
        verbose=verbose
    )

    cmd.run(
        auto=auto,
        amend=amend,
        context=context,
        breaking=breaking,
        dry_run=dry_run
    )


@cli.command()
@click.argument('base_branch')
@click.option('--verbose', '-v', is_flag=True, help='Show detailed analysis')
@click.pass_context
def review_merge(ctx, base_branch, verbose):
    """Preview merge with AI analysis"""
    from gai.commands.review_merge import ReviewMergeCommand

    verbose_mode = verbose or ctx.obj['config'].get('general.verbose', False)
    cmd = ReviewMergeCommand(
        config=ctx.obj['config'],
        git=ctx.obj['git'],
        verbose=verbose_mode
    )

    cmd.run(base_branch=base_branch, show_verbose=verbose)


@cli.command()
@click.argument('branch', required=False)
@click.option('--stat', is_flag=True, help='Show file statistics')
@click.option('--files', is_flag=True, help='List changed files')
@click.pass_context
def diff(ctx, branch, stat, files):
    """AI summary of branch differences"""
    from gai.commands.diff import DiffCommand

    verbose = ctx.obj['config'].get('general.verbose', False)
    cmd = DiffCommand(
        config=ctx.obj['config'],
        git=ctx.obj['git'],
        verbose=verbose
    )

    cmd.run(branch=branch, show_stat=stat, show_files=files)


@cli.command()
@click.option('--reset', is_flag=True, help='Reset all statistics')
@click.pass_context
def stats(ctx, reset):
    """Show usage statistics"""
    from gai.core.stats import get_stats
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel

    console = Console()
    stats_data = get_stats()

    if reset:
        if click.confirm('Reset all statistics?'):
            stats_data.reset()
            console.print("[green]✓[/green] Statistics reset")
        return

    summary = stats_data.get_summary()

    # Header
    console.print()
    console.print(Panel(
        "[bold cyan]gai Usage Statistics[/bold cyan]",
        border_style="cyan"
    ))
    console.print()

    # Commands table
    cmd_table = Table(title="Commands", show_header=True)
    cmd_table.add_column("Command", style="cyan")
    cmd_table.add_column("Times Used", justify="right", style="green")
    cmd_table.add_column("Last Used", style="yellow")

    for cmd, data in summary["commands"]["breakdown"].items():
        last_used = data.get("last_used", "Never")
        if last_used != "Never" and last_used:
            # Format timestamp
            from datetime import datetime
            try:
                dt = datetime.fromisoformat(last_used)
                last_used = dt.strftime("%Y-%m-%d %H:%M")
            except:
                pass

        cmd_table.add_row(
            cmd,
            str(data["count"]),
            last_used
        )

    console.print(cmd_table)
    console.print()

    # API usage table
    api_table = Table(title="API Usage", show_header=True)
    api_table.add_column("Metric", style="cyan")
    api_table.add_column("Value", justify="right", style="green")

    api_table.add_row("Total API calls", f"{summary['api']['total_calls']:,}")
    api_table.add_row("Total tokens", f"{summary['api']['total_tokens']:,}")
    api_table.add_row("Today calls", f"{summary['api']['today_calls']:,}")
    api_table.add_row("Today tokens", f"{summary['api']['today_tokens']:,}")
    api_table.add_row("This week tokens", f"{summary['api']['week_tokens']:,}")

    console.print(api_table)
    console.print()

    # Models table
    if summary['api']['by_model']:
        model_table = Table(title="By Model", show_header=True)
        model_table.add_column("Model", style="cyan")
        model_table.add_column("Calls", justify="right", style="green")
        model_table.add_column("Tokens", justify="right", style="yellow")

        for model, data in summary['api']['by_model'].items():
            # Shorten model name
            short_model = model.split('/')[-1] if '/' in model else model
            model_table.add_row(
                short_model,
                f"{data['calls']:,}",
                f"{data['tokens']:,}"
            )

        console.print(model_table)
        console.print()

    # Commits table
    commit_table = Table(title="Commits Generated", show_header=True)
    commit_table.add_column("Type", style="cyan")
    commit_table.add_column("Count", justify="right", style="green")

    commit_table.add_row("Total", f"{summary['commits']['total']:,}")
    commit_table.add_row("Auto-committed", f"{summary['commits']['auto']:,}")
    commit_table.add_row("Manually edited", f"{summary['commits']['manual_edit']:,}")
    commit_table.add_row("Dry-run only", f"{summary['commits']['dry_run']:,}")

    console.print(commit_table)
    console.print()

    # Footer
    if summary.get('created_at'):
        from datetime import datetime
        try:
            dt = datetime.fromisoformat(summary['created_at'])
            created = dt.strftime("%Y-%m-%d %H:%M")
            console.print(f"[dim]Tracking since: {created}[/dim]")
        except:
            pass

    console.print()


@cli.command()
@click.pass_context
def config_show(ctx):
    """Show current configuration"""
    cfg = ctx.obj['config']
    click.echo("Current gai configuration:")
    click.echo(f"  Config file: {cfg.config_path}")
    click.echo(f"  Model: {cfg.get('ai.model')}")
    click.echo(f"  Max tokens: {cfg.get('ai.max_tokens')}")
    click.echo(f"  Commit format: {cfg.get('commit.format')}")


@cli.command()
@click.option('--edit', '-e', is_flag=True, help='Interactively edit parallel models selection')
@click.option('--select', '-s', is_flag=True, hidden=True, help='DEPRECATED: use --edit instead')
@click.option('--tier', type=str, help='Filter by tier if API provides it (e.g., flagship, fast, light, reasoning)')
@click.option('--debug', is_flag=True, help='Show raw API response for debugging')
@click.option('--backend', type=str, help='Show models from specific backend only (groq, anannas, ollama, openrouter)')
@click.option('--free', is_flag=True, help='Show only free models (those with :free suffix)')
@click.pass_context
def models(ctx, edit, select, tier, debug, backend, free):
    """List configured models (or all available with filters)"""
    from gai.commands.models import ModelsCommand

    verbose = ctx.obj['config'].get('general.verbose', False)
    cmd = ModelsCommand(
        config=ctx.obj['config'],
        verbose=verbose
    )

    cmd.run(edit=edit, select=select, filter_tier=tier, debug=debug, filter_backend=backend, filter_free=free)


@cli.command()
@click.pass_context
def debug(ctx):
    """Explain git problems without fixing them"""
    from gai.commands.debug import DebugCommand

    verbose = ctx.obj['config'].get('general.verbose', False)
    cmd = DebugCommand(
        config=ctx.obj['config'],
        git=ctx.obj['git'],
        verbose=verbose
    )

    cmd.run()


@cli.command()
@click.option('--auto', is_flag=True, help='Automatically apply AI suggestions')
@click.pass_context
def resolve(ctx, auto):
    """Resolve merge conflicts with AI assistance"""
    from gai.commands.resolve import ResolveCommand

    verbose = ctx.obj['config'].get('general.verbose', False)
    cmd = ResolveCommand(
        config=ctx.obj['config'],
        git=ctx.obj['git'],
        verbose=verbose
    )

    cmd.run(auto=auto)


@cli.command()
@click.option('--rebase', is_flag=True, help='Rebase local commits on top of remote')
@click.option('--merge', is_flag=True, help='Merge remote changes with local')
@click.option('--reset', is_flag=True, help='Reset to match remote (DANGEROUS)')
@click.option('--pull', is_flag=True, help='Simple pull (when just behind)')
@click.option('--auto', is_flag=True, help='Automatically choose best solution')
@click.option('--force', '-f', is_flag=True, help='Skip confirmation prompts')
@click.pass_context
def fix(ctx, rebase, merge, reset, pull, auto, force):
    """Fix git problems automatically"""
    from gai.commands.fix import FixCommand

    verbose = ctx.obj['config'].get('general.verbose', False)
    cmd = FixCommand(
        config=ctx.obj['config'],
        git=ctx.obj['git'],
        verbose=verbose
    )

    cmd.run(
        rebase=rebase,
        merge=merge,
        reset=reset,
        pull=pull,
        auto=auto,
        force=force
    )


@cli.command()
@click.option('--commits', '-c', is_flag=True, help='Show commits since last version')
@click.pass_context
def version(ctx, commits):
    """Show current version and suggest next"""
    from gai.commands.version import VersionCommand

    verbose = ctx.obj['config'].get('general.verbose', False)
    cmd = VersionCommand(
        config=ctx.obj['config'],
        git=ctx.obj['git'],
        verbose=verbose
    )

    cmd.run(show_commits=commits)


@cli.command()
@click.option('--major', is_flag=True, help='Force major version bump')
@click.option('--minor', is_flag=True, help='Force minor version bump')
@click.option('--patch', is_flag=True, help='Force patch version bump')
@click.option('--auto', is_flag=True, help='Automatically execute without confirmation')
@click.option('--dry-run', is_flag=True, help='Show what would happen without executing')
@click.option('--skip-changelog', is_flag=True, help='Skip CHANGELOG.md generation')
@click.pass_context
def release(ctx, major, minor, patch, auto, dry_run, skip_changelog):
    """Create a semantic version release"""
    from gai.commands.release import ReleaseCommand

    # Determine bump type from flags
    bump_type = None
    if major:
        bump_type = 'major'
    elif minor:
        bump_type = 'minor'
    elif patch:
        bump_type = 'patch'

    verbose = ctx.obj['config'].get('general.verbose', False)
    cmd = ReleaseCommand(
        config=ctx.obj['config'],
        git=ctx.obj['git'],
        verbose=verbose
    )

    cmd.run(
        bump_type=bump_type,
        auto=auto,
        dry_run=dry_run,
        skip_changelog=skip_changelog
    )


@cli.command()
@click.option('--force', '-f', is_flag=True, help='Overwrite existing config')
def init(force):
    """Initialize gai configuration in ~/.config/gai/"""
    import shutil
    from gai.core.config import Config

    # Get user config directory
    xdg_config = os.getenv('XDG_CONFIG_HOME')
    if xdg_config:
        config_dir = Path(xdg_config) / 'gai'
    else:
        config_dir = Path.home() / '.config' / 'gai'

    config_file = config_dir / 'config.yaml'
    env_file = config_dir / '.env'

    # Check if already exists
    if config_file.exists() and not force:
        click.echo(f"Config already exists at {config_file}")
        click.echo("Use --force to overwrite")
        return

    # Create directory
    config_dir.mkdir(parents=True, exist_ok=True)

    # Copy templates from package
    package_dir = Path(__file__).parent.parent.parent
    template_yaml = package_dir / '.gai.yaml'
    template_env = package_dir / '.env.example'

    # Copy config template
    if template_yaml.exists():
        shutil.copy(template_yaml, config_file)
        click.echo(f"✓ Created config: {config_file}")
    else:
        # Create default config from code
        cfg = Config._default_config(None)
        import yaml
        with open(config_file, 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        click.echo(f"✓ Created default config: {config_file}")

    # Copy .env template
    if template_env.exists():
        shutil.copy(template_env, env_file)
        click.echo(f"✓ Created env template: {env_file}")
    else:
        # Create basic .env template
        with open(env_file, 'w') as f:
            f.write("GROQ_API_KEY=your-api-key-here\n")
        click.echo(f"✓ Created env file: {env_file}")

    click.echo()
    click.echo("Next steps:")
    click.echo(f"1. Edit {env_file} and add your GROQ_API_KEY")
    click.echo(f"2. Optionally edit {config_file} to customize settings")
    click.echo(f"3. Run 'gai commit' from any git repo")
    click.echo()
    click.echo(f"DB will be stored at: {config_dir / 'gai.db'}")


if __name__ == '__main__':
    cli()
