import subprocess
import tempfile
from pathlib import Path
from transformers import T5Config, T5ForConditionalGeneration, load_tf_weights_in_t5
import click
import os, glob
import shutil

def remove(pat):
	fileList = glob.glob(pat)
	# Iterate over the list of filepaths & remove each file.
	for filePath in fileList:
		try:
			os.remove(filePath)
		except:
			print("Error while deleting file : ", filePath)

def convert_model(base, path, new_path):
    base_model = f"/home/pouramini/pret/{base}/"
    model = T5ForConditionalGeneration(T5Config.from_pretrained(base_model))
    print("loading weights...")
    load_tf_weights_in_t5(model, None, path)
    model.eval()
    print("saving HF weights...")
    model.save_pretrained(new_path)

def run_system_process(cmd):
    print(f"Running command: {cmd}")
    res = subprocess.run(cmd, shell=True, capture_output=True)
    print(f"Command result: {res.stdout}")
    return res

def move(pat, dest_dir, copy =False):
    for file in glob.glob(pat):
        print(file)
        if copy:
            shutil.copy(file, dest_dir)
        else:
            try:
                shutil.move(file, dest_dir)
            except:
                print(file, " already exists!")

@click.command()
@click.argument("fid", type=str)
@click.option(
    "--path",
    envvar="PWD",
    #    multiple=True,
    type=click.Path(),
    help="The current path (it is set by system)"
)
@click.option(
    "--base",
    "-b",
    default="t5-base",
    type=str,
    help=""
)
@click.option(
    "--do_remove",
    is_flag=True,
    help=""
)
@click.option(
    "--hfname",
    "-hn",
    default="hf",
    type=str,
    help=""
)
def main(fid, path, base, do_remove, hfname):
    flag = True
    ii = 0
    hf_path = path + f"/{hfname}" 
    base = base.replace("_","-")
    if Path(hf_path).exists():
        print("The folder already exists.")
    Path(hf_path).mkdir(parents=True, exist_ok=True)
    ckp_path = ""
    all_ckps = sorted(glob.glob("model.ckpt-*.index"))
    rep = 0
    for fname in all_ckps[1:]:
        ckp = int(fname.split("-")[1].split(".")[0])
        print("checkpoint:", ckp)
        pat1 = f"model.ckpt-{ckp}*"
        if do_remove:
            remove(path + "/" + pat1)
            continue
        pat2 = "operative_config.gin"
        pat3 = "data/*"
        src = path + "/" + fname
        print("src:", src)
        flag = Path(src).exists()
        ckp_path = hf_path + f"/tf_{base}_{ckp}"
        out_dir = f"{ckp_path}/hf-{ckp}-" + str(Path(path).stem)
        print("dest:", ckp_path)

        if Path(ckp_path + "/" + fname).exists():
            remove(path + "/" + pat1)
        elif flag:
            Path(ckp_path).mkdir(parents=True, exist_ok=True)
            print("moving...")
            copy = rep >= len(all_ckps)
            move(pat1, ckp_path, copy)
            move(pat2, ckp_path, True)
            Path(ckp_path + "/data").mkdir(parents=True, exist_ok=True)
            move(pat3, ckp_path + "/data", True)
            with open(ckp_path + "/checkpoint", "w") as f:
                cont = f'model_checkpoint_path: "model.ckpt-{ckp}"'
                f.write(cont)
        if not Path(out_dir).exists() and Path(ckp_path).exists():
            convert_model(base, ckp_path, out_dir)
            print("saved in ", out_dir)
        rep += 1
    print("saved in ", out_dir)


if __name__ == "__main__":
    #convert_model()
    main()
