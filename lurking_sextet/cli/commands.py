import click

@click.group()
def main_cli():
    """ lurking_sextet """
    pass

from .cmd_debug import cli as debug
main_cli.add_command(debug)
