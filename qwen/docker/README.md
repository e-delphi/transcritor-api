# Transcritor API — Qwen (Docker com GPU)

A mesma API de transcrição e **legendas** da [versão Windows](../windows/README.md), num
container: Qwen3-ASR (texto) e Qwen3-ForcedAligner (instante de cada palavra) servidos
pelo [audio.cpp](https://github.com/0xShug0/audio.cpp), na placa de vídeo. Motor e
modelos vão dentro da imagem: o container roda **offline, sem token e sem login**.

É um projeto independente dos outros do repositório: o WhisperX em Docker, que separa
falantes ([CPU](../../whisper/docker/cpu/README.md) ou [GPU NVIDIA](../../whisper/docker/gpu-cuda/README.md)),
e a [instalação nativa do Windows](../windows/README.md).

| | Docker WhisperX | **Qwen (este)** | Windows nativo |
|---|---|---|---|
| Onde roda | container, CPU | container, GPU | Windows, GPU |
| Separação por falante | sim | não | não |
| Erro mediano no início da palavra | ~65 ms | ~22 ms | ~22 ms |
| Reunião de 88 min | ~36 min | ~7 min na GPU* | ~7 min |

\* Medido com a instalação nativa numa RX 9070 XT; o container fala com o mesmo
driver e tem o mesmo desempenho quando a placa chega até ele (veja abaixo).

## Duas variantes

| Variante | Placas | Onde funciona |
|---|---|---|
| `vulkan` | AMD, Intel | Linux, com a placa repassada (`/dev/dri`) |
| `cuda` | NVIDIA | Linux, ou Windows com Docker Desktop (`--gpus all`) |

**Windows com placa AMD ou Intel não serve:** o Docker Desktop roda os containers numa
máquina virtual (WSL2) que só repassa placas NVIDIA. Aí o container funciona, mas na
CPU — cerca de 10× mais devagar. Para essas máquinas, use a
[instalação nativa](../windows/README.md).

Sem placa nenhuma, as duas variantes caem sozinhas para a CPU em vez de parar.

## Build

```bash
docker build -t transcritor-api:qwen-vulkan .
docker build --build-arg VARIANT=cuda -t transcritor-api:qwen-cuda .
```

O build baixa ~3,7 GB (motor e modelos) e confere o SHA-256 de cada arquivo. Se o
GitHub ou o HuggingFace falharem, cada arquivo é buscado num espelho no Google Drive.
Nenhum download exige conta. A imagem ocupa ~4,2 GB em disco (o `docker images`
mostra quase o dobro: o armazenamento do containerd conta as camadas comprimidas e
descomprimidas).

## Executar

**AMD ou Intel no Linux** — repasse a placa e o grupo que dá acesso a ela (o número
do grupo é o da máquina, não o do container):

```bash
docker run -d -p 8000:8000 -v qwen-dados:/data \
  --device /dev/dri --group-add "$(stat -c %g /dev/dri/renderD128)" \
  transcritor-api:qwen-vulkan
```

**NVIDIA** (precisa do [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
no Linux; no Windows o Docker Desktop já traz):

```bash
docker run -d -p 8000:8000 -v qwen-dados:/data --gpus all transcritor-api:qwen-cuda
```

Ou com compose, escolhendo o perfil:

```bash
RENDER_GID=$(stat -c %g /dev/dri/renderD128) docker compose --profile vulkan up -d
docker compose --profile cuda up -d
```

**Confira que a placa está em uso:** `GET /health` responde o backend.

```json
{"status": "ok", "models_loaded": true, "engine": "qwen", "backend": "vulkan"}
```

`"backend": "cpu"` quer dizer que a placa não chegou ao container — o log dele
mostra o aviso. O volume em `/data` guarda os jobs; sem ele, tudo se perde ao
recriar o container.

## A API

As rotas e o resultado são os da [versão Windows](../windows/README.md#rotas):
`POST /transcribe`, a fila em `POST /jobs` / `GET /jobs/{id}` /
`GET /jobs/{id}/download` (`json`, `txt`, `srt`, `vtt`), e `GET /health`.

- Legendas em blocos de até **2 linhas de 42 caracteres**, ajustáveis no download
  (`max_caracteres`, `max_linhas`, `max_segundos`).
- Em português, números por extenso acima de dez viram algarismos ("dois mil e vinte
  e seis" → `2026`).
- Só os trechos com fala vão para o modelo, e repetições em laço do modelo (em música
  ou ruído contínuo) são cortadas.
- `diarization=true` e `task=translate` respondem HTTP 400: este motor não separa
  falantes nem traduz.

## Variáveis de ambiente

| Variável | Padrão | Descrição |
|---|---|---|
| `QWEN_BACKEND` | `auto` | `auto`, `vulkan`, `cuda` ou `cpu`. `auto` usa a placa se ela estiver no container |
| `QWEN_MAX_MODELS` | `2` | `1` carrega um modelo de cada vez: ~2,5 GB de memória de vídeo em vez de ~3,6 GB, quase sem perda de velocidade |
| `QWEN_THREADS` | núcleos, até 8 | Threads da CPU usadas pelo motor |
| `JOB_TTL_SECONDS` | `21600` | Validade de um job (6 h) antes de áudio e resultado serem apagados |
| `MAX_UPLOAD_MB` | `1024` | Limite de upload |
| `NUMEROS_POR_EXTENSO_ATE` | `10` | Números até este valor ficam por extenso |
| `LEGENDA_MAX_CARACTERES` / `LEGENDA_MAX_LINHAS` / `LEGENDA_MAX_SEGUNDOS` | `42` / `2` / `7` | Padrões das legendas |

## Licença

GPL-3.0 ou posterior — veja o [LICENSE](../../LICENSE) na raiz do repositório. As
dependências têm licenças próprias, compatíveis com a GPLv3:

| Componente | Licença |
|---|---|
| audio.cpp | Apache-2.0 |
| Qwen3-ASR, Qwen3-ForcedAligner | Apache-2.0 |
| FastAPI, Uvicorn | MIT / BSD-3-Clause |
| PyAV, NumPy | BSD-3-Clause |
| Mesa (drivers Vulkan) | MIT |
| Imagem base CUDA (variante `cuda`) | [licença NVIDIA](https://docs.nvidia.com/cuda/eula/) |
