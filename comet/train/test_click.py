import click

cli = click.Group()

@cli.command()
@click.option('--opt1', default=1)
@click.option('--opt2', default=2)
def test(opt1, opt2):
    print(opt1)
    print(opt2)

@cli.command()
@click.pass_context
def dist(ctx):
    args = {"opt1":3, "opt2": 4}
    ctx.invoke(test, **args)


if __name__ == "__main__":
    dist()
