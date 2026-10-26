# Vosk Cymraeg / Welsh Vosk

Little package currently under development

## Fetching datasets

To fetch the datasets from HuggingFace, run the following script:
```bash
uv run fetch
```
Note that you need to have the enviroment variable HF_TOKEN set in order to read from HuggingFace. This can be set using a .env file in the root directory of the project.

## Publishing a trained model

`publish.sh` uploads the bundled model (`models/model.tar.gz`, as written by the
recipe's `copy_final_result.sh`) to a HuggingFace model repo, reading `HF_TOKEN`
from `.env`:

```sh
./publish.sh <user>/<repo> models/model.tar.gz
```

## Running the training environment
This repository supports both [Dev Containers](https://containers.dev/) and [Docker compose](https://docs.docker.com/compose/) and both use the same Dockerfile.

The Makefile is the front door:

```sh
make setup   # one-off: writes .env with your HF_TOKEN (prompts if unset)
make build   # build the training image
make run     # start and attach to the training environment
```

Your UID/GID are detected and passed in automatically by the `make` targets, so files created inside the environment have the correct host ownership — no manual configuration needed. GPU selection and other variables live in `.env` (see [howto.md](howto.md) for the full table).

## Datasets Used

### Transcribed Speech
|Name|Author|Source|
|---|---|---|
| Banc Trawsgrifiadau Bangor | Language Technologies Unit, Bangor University | [link](https://huggingface.co/datasets/techiaith/banc-trawsgrifiadau-bangor) |
| Lleisiau ARFOR | Cymen in collaboration with the Language Technologies Unit, Bangor University | [link](https://huggingface.co/datasets/cymen-arfor/lleisiau-arfor)
| Common Voice | Mozilla Foundation | [link](https://commonvoice.mozilla.org) |

### Pronunciation Dictionaries
|Name|Author|Source|
|---|---|---|
| Geiriadur Ynganu Bangor | Language Technologies Unit, Bangor University | [link](https://github.com/techiaith/geiriadur-ynganu-bangor) |

In addition the resources listed above, the final pronunciation dictionary contains entires from [Lecsicon Cymraeg Bangor](https://github.com/techiaith/lecsicon-cymraeg-bangor) that has been converted using an [updated version](https://github.com/Cymru-Breizh-Agile-Cymru-Project/vosk-cymraeg/blob/main/src/vosk_cymraeg/llef_py3.py) of [welsh-lts](https://github.com/techiaith/welsh-lts/blob/master/llef.py).

## Maintainers

This repository is being created and maintained by:
- [Gweltaz Duval-Guennoc](https://github.com/gweltou)
- [Dewi Bryn Jones](https://github.com/DewiBrynJones)
- [Preben Vangberg](https://github.com/prvInSpace)
- [Sasha Wanasky](https://github.com/wanasash)

This work is being funded by the Welsh Government through their [Agile Cymru](https://www.gov.wales/agile-cymru-guidance) programme.