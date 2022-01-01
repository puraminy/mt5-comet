import click
import pandas as pd
@click.command()
@click.argument("fname")
@click.option(
    "--sample",
    "-n",
    default=2000,
    type=int,
    help=""
)
@click.option(
    "--random_state",
    "-rs",
    default=1,
    type=int,
    help=""
)
def main(fname, sample,random_state):
    col_name = "prefix"
    df = pd.read_table(fname)
    probs = df[col_name].map(df[col_name].value_counts())
    sample_df = df.sample(n=sample, weights=probs, random_state=random_state)
    ff = fname.replace(".tsv","") + "_sample_" + str(sample) + ".tsv"
    print(ff)
    sample_df.to_csv(ff, index=False, sep="\t") 

main()
