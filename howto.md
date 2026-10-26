# How to train a model using the repository

## Prerequisites
This guide assumes that you have [Docker](https://docs.docker.com/engine/install/) installed on your system. If you have [uv](https://docs.astral.sh/uv/) installed you can run the `fetch` and `combine` scripts outside the training environment, otherwise you can simply run them inside the training environment because `uv` is installed by default. To ensure that the paths are correct `export` should always be run inside the training environment.

## Building and running the training environment
The Makefile is the front door — it computes your UID/GID automatically (so in-container file ownership matches your host user) and creates the `.env` on first use:

```sh
make setup   # one-off: writes .env with your HF_TOKEN (prompts if unset)
make build   # build the training image (>20 min first time, cached afterwards)
make run     # start the training environment
```

`build` and `run` create the `.env` for you (via `setup`) if it doesn't exist yet.

### Configuration via .env
`compose.yaml` interpolates its variables from the `.env` file in this directory (native Docker Compose behaviour). The recognised variables:

| Variable | Purpose | Default |
|----------|---------|---------|
| `HF_TOKEN` | HuggingFace token (private datasets/models) | — (written by `make setup`) |
| `HF_USERNAME` | HuggingFace username | — |
| `DEVICE` | GPU id to expose to the training env (single id) | `2` |

UID/GID need no configuration — the `make` targets pass them in automatically (they take precedence over any stale values in `.env`). If you bypass make and invoke `docker compose` directly, pass them yourself: `UID=$(id -u) GID=$(id -g) docker compose run --remove-orphans training_env`.

Similar to [Development Containers](https://containers.dev/), the current repository is mounted as a volume at `/workspaces/vosk`. Any changes made to this path affect the same files in the host system.

## Initialising the submodules
In case it wasn't done when you cloned the repository run the following command to initialise any submodules:
```sh
git submodule update --init --recursive
```
This is required to make the phonemiser work since it depends on [Geiriadur Ynganu Bangor](https://github.com/techiaith/geiriadur-ynganu-bangor) and [Lecsicon Cymraeg Bangor](https://github.com/techiaith/lecsicon-cymraeg-bangor), both of which are located in `data/external` as submodules.


## Starting again from scratch

If you are re-running rather than starting fresh, wipe the previous run's data first so nothing is silently inherited. **On the host**, from this directory:

```sh
git clean -xfdn data/   # dry run — see below
git clean -xfd  data/
rm -f models/model.tar.gz
```

The dry run should list exactly two paths and nothing else:

```
Would remove data/output/
Would remove data/processed/
```

Use `git clean` rather than `rm -rf data/`. `data/.gitignore` is `*` with `!external`, and `data/external/{geiriadur-ynganu-bangor,lecsicon-cymraeg-bangor}` are submodules that `git clean` leaves alone — the phonemiser reads `data/external/geiriadur-ynganu-bangor/bangordict.dict`, so wiping those breaks `uv run export` until you re-initialise them.

The recipe's own working directories (`data/`, `exp/`, `mfcc/` under `/opt/kaldi/egs/cy`) live inside the container and are wiped by `bash run.sh -1`, so they need no separate handling.

## Getting the required data
All four sources (Banc Trawsgrifiadau Bangor, Common Voice, Lleisiau Arfor, Enwau Cymraeg) are downloaded from HuggingFace by the `fetch` script, which writes them to `data/interim/<source>/`:
```sh
uv run fetch
```
`HF_TOKEN` must be set (via `.env` or the environment) — Banc Trawsgrifiadau Bangor is a private repo.

> Common Voice no longer needs downloading by hand. Earlier versions of this guide told you to fetch it from [commonvoice.mozilla.org/cy/datasets](https://commonvoice.mozilla.org/cy/datasets) and unpack the `.tsv` splits into `data/raw/cv/cy/`; since the loader switched to HuggingFace (`techiaith/commonvoice_23_0_cy`) nothing reads `data/raw` at all.

## Combining the data into one dataset
To combine the data from different datasets into one, run the `combine` script by running the following command:
```sh
uv run combine
```
That should create a dataset in processed that can them be converted into a format that Kaldi understands.

## Exporting the data to Kaldi
**NB: This command should always be run inside the training environment to ensure that the outputted paths are correct**

To export the combined dataset into a format that Kaldi accepts, run the `export` script by running the following command:
```sh
uv run export
```
The export script takes several arguments, but in most cases the defaults are good enough. The two worth knowing about:

| Flag | Effect |
|------|--------|
| `--lang` | Which languages to output. Default is Welsh **and** English; `--lang cy` restricts it to Welsh. |
| `--additional_text` | Off by default. Adds five external text corpora (cofnod y cynulliad, legislation.gov.uk, llyw.cymru, TTS prompts, placenames) to the LM/lexicon text. See below. |

The resulting dataset should then be exported to `data/output/`

### About `--additional_text`

This flag is worth understanding because it silently sets a ceiling on WER. The sentences it controls feed **two** outputs: `data/local/corpus.txt` (the 3-gram ARPA, hence `G.fst`) and `data/local/dict_nosp/lexicon.txt` (hence `words.txt`). A WFST decoder can only ever emit words present in `words.txt` — unlike Whisper, which can spell anything from subword pieces.

With the flag off, both come from your training transcripts alone. For one 2026 training set (~1.36M words of transcripts) that meant a 60,681-word decoding vocabulary, against which the held-out references were 2.0% (BTB), 3.3% (Lleisiau Arfor) and 3.4% (Common Voice) out-of-vocabulary — words the model is structurally incapable of producing, mostly names and English code-switching.

Two caveats before switching it on:

- [`export_kaldi.py`](src/vosk_cymraeg/scripts/export_kaldi.py) concatenates the external corpora and de-duplicates, so every sentence carries equal weight. Several million sentences of legislative and government Welsh will swamp ~100k sentences of spontaneous speech. Expect it to help read speech (Common Voice) and possibly *hurt* the spontaneous sets. The fix is to keep the sources separate and let pocolm tune their interpolation weights on dev — see [Language modelling](#language-modelling-pocolm) below — rather than concatenating them here.
- It also activates `--lang`. With a Welsh-only import the default `--lang cy en` is currently a no-op, but the external loaders return both sides of each bilingual corpus, so `uv run export --additional_text` with defaults pulls English text and English pronunciations into the lexicon too. Given how much of the OOV is English that may be what you want — but make it a deliberate choice.

No model trained from this recipe has used this flag so far.

## Language modelling (pocolm)

Stage 2 of the recipe builds the 3-gram ARPA with [pocolm](https://github.com/danpovey/pocolm)
(Apache-2.0). It previously used SRILM's `ngram-count -wbdiscount`.

SRILM was replaced for two reasons. Its Research Community License restricts use to
non-commercial research, which is the reason published Kaldi/Vosk models carried
CC-BY-NC; and the Dockerfile was fetching it from an unauthorised third-party GitHub
mirror rather than SRI's download form, so the licence had never actually been
accepted. pocolm is permissive, so models built with it can ship under Apache-2.0 or
CC-BY. It also tunes its discounting constants against held-out dev text instead of
using fixed Witten-Bell discounting.

### Adding more text sources

pocolm reads a *directory* of text files — one per source — plus a `dev.txt` it tunes
against. The recipe builds this at `data/local/pocolm/text/`:

```
data/local/pocolm/text/
    transcripts.txt   # from data/local/corpus.txt
    dev.txt           # data/dev/text, utterance ids stripped
```

To bring in extra corpora (the `--additional_text` datasets, or OSCAR), write each one
as its **own** `.txt` file in that directory rather than concatenating it into
`corpus.txt`. pocolm then interpolates them at the count level with weights fitted on
`dev.txt`, so a source is weighted by how much it actually helps rather than by how
many sentences it happens to contain. That is the swamping problem described under
`--additional_text` above.

The LM currently has a single source, so pocolm is buying tuned discounting and
nothing more. The interpolation only starts paying off once a second file appears.

### Checking the result

Stage 2 logs dev perplexity to `data/local/pocolm/dev_perplexity.log`, which lets you
compare LM changes without waiting for a full decode. Treat it as indicative rather
than exact: pocolm tunes its metaparameters on the same `dev.txt`, so the figure is
mildly optimistic. WER on a held-out test set remains the number that counts.

`pocolm_max_memory` at the top of `run.sh` (default `4G`) caps pocolm's memory use;
raise it for a substantially larger corpus.

## Initialising the recipe
The recipe located at `recipes/cy` should automatically be copied to `/opt/kaldi/egs/cy` when the image is build, however, if you make changes to the recipe or want to re-initialise the recipe, simply delete content of the folder and copy it back over by doing the following:
```sh
rm -rf /opt/kaldi/egs/cy/*; cp -r /workspaces/vosk/recipes/cy/* /opt/kaldi/egs/cy
```

## Initalising the data in the recipe
Before running the main training script you need to copy over the data folder and clean the recipe. This can be done automatically by changing your working directory to `/opt/kaldi/egs/cy` and running `bash run.sh -1` like so:
```sh
cd /opt/kaldi/egs/cy; bash run.sh -1
```

This cleans any intermediary files and copies the Kaldi dataset from `data/output` to the local data folder. Since Kaldi writes a lot of things to this folder it is better to copy it as opposed to linking it. The audio files themselves stay where they are (`data/interim/<source>/`) and `wav.scp` points at them by absolute path, so you are only really copying the dataset metadata.

Note that `run.sh -1` *only* prepares: the later stages are guarded on `$stage -ge N`, which `-1` never satisfies. Be aware too that the argument is an **upper bound, not a starting point** — the opposite of the usual Kaldi `--stage` convention. `bash run.sh 10` runs every stage from 0 through 10 (i.e. the whole pipeline, which is why that is the command below); `bash run.sh 0` would run the data-validation stage and nothing else.

## Running the training script
You can now run the training script by running the following command:
```sh
bash run.sh 10
```

Note that there are some things that you might want to do first and some issues you might run into, and I'll explain these now.

### Changing the hyperparameters
The `local/chain` folder contains a lot of script that are simply the same script with different hyperparameters. If you want to change the parameters either change the script that is called in stage 10 of `run.sh` with one in that folder or make one yourself. I recommend trying `local/chain/run_tdnn_260412.sh` which is the one currently being used by the Breton project.


### CPU training optimization

Beware of out of memory errors when changing the following training parameters in Kaldi.

Increase the number of parallel jobs used for feature extraction and training, by setting the parameter `nj` in script `run.sh`:

```sh
nj=8
```

### Nvidia issues
Firstly make sure that the Nvidia drivers are working as expected. You can do this by running the following command: `nvidia-smi`. If this command works as expected you are good to go, otherwise you might want to restart the training environment.

Note that since stage 1-13 takes quite a while and are run exclusively on the CPU and since stage 14 is really the only one that uses the GPU the training might fail once it reaches stage 14 due to Nvidia timing out. If that happens simply restart the training environment and re-run the chain script that stage 10 of `run.sh` invokes — currently `local/chain/run_tdnn_260412.sh` — as follows:
```sh
bash local/chain/run_tdnn_260412.sh --stage 14
```
This resumes training from stage 14. (Unlike `run.sh`, the chain scripts use the standard Kaldi convention, where `--stage` *is* a starting point.) If it complains about some files already existing, delete these and re-run the command.

### In-recipe evaluation when there is no test split

If your data has **no test split** (for example because the final evaluation runs outside the recipe, on a separate benchmark), the recipe falls back to the **dev** split so you still get a quick signal that the chain model trained into something sane: when `data/test` is absent but `data/dev` exists, `local/nnet3/run_ivector_common.sh` builds dev iVectors and the chain script's decode stages (16/17) run against `data/dev`. Read the dev WER with:

```sh
for d in exp/chain/tdnn1j_sp/decode_*_dev; do echo "== $d =="; cat "$d"/scoring_kaldi/best_wer; done
```

Notes:

- This dev figure is a **pre-flight sanity check**, not a result to report: pocolm tunes the language model on the same dev transcripts. Use it only to confirm the model isn't broken before you bundle and upload it.
- The decode job count is capped at `decode_nj` so a many-speaker dev split (e.g. from Common Voice) doesn't spawn a runaway number of jobs.
- If neither `data/test` nor `data/dev` is present, in-recipe decoding is skipped entirely; evaluate the bundled model externally.

## Bundling the model
To bundle the files at the end, run the following command:
```sh
bash copy_final_result.sh
```
This will compile the required files from the training to a folder called `model/`. If you are using a newer version of the script it will also export the model to a `.tar.gz` file called `model.tar.gz`. If it doesn't, you can do this manually by running the command:
```sh
tar -czvf model.tar.gz model/
```

Since the recipe folder is not persistent, in order to export this model from the training environment, copy it to the `/workspaces/vosk/` folder, e.g. by running the following command:
```sh
mkdir -p /workspaces/vosk/models/; cp model.tar.gz /workspaces/vosk/models/
```

Then upload the bundle to a HuggingFace model repo with `publish.sh` (which reads `HF_TOKEN` from `.env`), from the repository root on the host:
```sh
./publish.sh <user>/<repo> models/model.tar.gz
```

## What next?

You know got a model that you are able to test and run. I'd recommend checking out [vosk-cli](https://github.com/Cymru-Breizh-Agile-Cymru-Project/vosk-cli) which is a small Python script that allows you to run a vosk model by simply providing it a folder like so:
```sh
uv run vosk-cli -m models/nn-no
```
where the content of `models/nn-no` is the content inside the `model.tar.gz` file.
