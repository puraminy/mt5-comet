import glob
import click
from comet.train.mylogs import *


@click.command()
@click.argument("fname", type=str)
@click.option(
    "--exp",
    "-exp",
    default="",
    type=str,
    help=""
)
@click.option(
    "--model_id",
    "-pm",
    default="t5-base",
    type=str,
    help=""
)
def main(fname, exp, model_id):
    inps = glob.glob(f"*{fname}*")
    if len(inps) == 0:
        print(f"A file with this pattern '*{fname}*' wasn't found")
        return
    preds_file = inps[0]
    extra = "_" + now
    model_id = "t5_base"
    m_name = model_id + "-" + method
    lang = "en2en"
    w_str = "unwrapped"
    f_str = "unfrozen"
    epochs_num = 1
    trial = 1
    experiment = Path(fname).stem + "_" + exp
    exp_info = {"exp":experiment, "model":model_id, "lang": lang, 
                    "method":method, 
                    "wrap": w_str, 
                    "frozen":f_str, 
                    "steps":train_samples,
                    "epochs":epochs_num,
                    "trial":trial,
                    "date":extra}
    evaluate(ds, split_dir, exp_info, 
            test_samples, preds_file = preds_file, scorers=scorers)

if __name__ == "__main__":
    main()
