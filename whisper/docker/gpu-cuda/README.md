# Transcritor API — WhisperX (Docker, GPU NVIDIA)

Container autocontido de transcrição com **separação por falante**, construído sobre
[WhisperX](https://github.com/m-bain/whisperX). Os modelos são gravados dentro da
imagem durante o build, então o container roda **offline, sem token e sem
autenticação** — verificado com `--network none`, inclusive o alinhamento.

Esta é a variante **GPU NVIDIA (CUDA)**: transcrição, alinhamento e separação de
falantes rodam na placa de vídeo. Precisa de placa NVIDIA com driver recente e do
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
no Linux — no Windows, o Docker Desktop com WSL2 já repassa a placa. Para máquinas sem
NVIDIA, use a variante [cpu](../cpu/README.md); a API é a mesma.

A imagem ocupa **13 GB** em disco — uns 5 GB a mais que a de CPU, pelas bibliotecas CUDA (`docker images` reporta um número bem maior porque
o image store do containerd conta blobs comprimidos e descomprimidos).

## Como funciona

| Etapa | Componente |
|---|---|
| Decodificação de áudio/vídeo | FFmpeg — qualquer formato vira PCM mono 16 kHz |
| Detecção de fala (VAD) | pyannote, com os pesos embarcados no próprio WhisperX |
| Transcrição | `faster-whisper` (CTranslate2) com `large-v3` |
| Alinhamento de palavras | wav2vec2 — timestamps precisos por palavra |
| Separação de falantes | `pyannote/speaker-diarization-community-1` |
| API e fila | FastAPI + Uvicorn |

O VAD é o padrão do WhisperX, cujos pesos vêm dentro do pacote — nada é baixado e não
há repositório restrito envolvido. O modo Silero seria pior aqui: ele busca o modelo no
GitHub via `torch.hub` na primeira execução, exigindo rede e permissão de escrita.

## Token do HuggingFace: só no build

O modelo de diarização exige aceite de termos no HuggingFace. Isso afeta **apenas o
build** — o container em si nunca usa token nem acessa a rede.

Antes do primeiro build, uma vez:

1. Aceite os termos em
   [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1).
2. Crie um token de leitura em [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
   e grave-o num arquivo fora do repositório, por exemplo `~/hf_token.txt`.

O token entra no build por `--secret`, que o monta apenas durante o passo de download.
Ele **não fica gravado em nenhuma camada da imagem** e não aparece no histórico dela.

## Build

```bash
docker build --secret id=hf_token,src=$HOME/hf_token.txt -t transcritor-api:whisper-cuda .
```

Ou com compose (aponte `HF_TOKEN_FILE` para o seu arquivo):

```bash
HF_TOKEN_FILE=$HOME/hf_token.txt docker compose build
```

Para trocar o modelo Whisper (`tiny`, `base`, `small`, `medium`, `large-v2`, `large-v3`)
ou os idiomas com alinhamento embutido:

```bash
docker build --secret id=hf_token,src=$HOME/hf_token.txt \
  --build-arg WHISPER_MODEL=small --build-arg ALIGN_LANGUAGES=pt \
  -t transcritor-api:small .
```

Cada idioma de alinhamento pesa na imagem: o português usa um wav2vec2 de ~1,3 GB, o
inglês um modelo do torchaudio bem menor. Idiomas sem alinhador embutido continuam
sendo transcritos normalmente, apenas sem o refinamento de timestamp por palavra.

## Executar

```bash
docker run -d --gpus all -p 8000:8000 -v transcritor-dados:/data transcritor-api:whisper-cuda
```

O volume em `/data` guarda os jobs e as transcrições. Sem ele, tudo se perde quando o
container é recriado.

```bash
docker compose up -d
```

Os modelos levam alguns minutos para carregar na subida; `GET /health` só responde
quando estiverem prontos.

## Rota síncrona

`POST /transcribe` — `multipart/form-data`. Devolve o JSON pronto. Boa para áudios
curtos; para arquivos longos use a fila.

| Campo | Padrão | Descrição |
|---|---|---|
| `file` | **obrigatório** | Áudio ou vídeo (mp3, wav, m4a, ogg, flac, mp4, mkv…) |
| `language` | auto | Código ISO (`pt`, `en`…). Vazio = detecção automática |
| `task` | `transcribe` | `transcribe` ou `translate` (traduz para inglês) |
| `diarization` | `true` | Separar por falante |
| `alignment` | `true` | Alinhar timestamps por palavra |
| `num_speakers` | auto | Número exato de falantes, se conhecido |
| `min_speakers` / `max_speakers` | auto | Limites da estimativa automática |

```bash
curl -X POST http://localhost:8000/transcribe -F "file=@reuniao.mp3" -F "language=pt"
```

## Fila: áudios longos, progresso e cache

`POST /jobs` aceita os mesmos campos e devolve na hora um identificador.

```bash
curl -X POST http://localhost:8000/jobs -F "file=@reuniao.mp3" -F "num_speakers=3"
```

```json
{
  "job_id": "3f9c…",
  "file_sha256": "a71e…",
  "status": "queued",
  "progress_percent": 0.0,
  "cached": false,
  "status_url": "/jobs/3f9c…",
  "expires_in_seconds": 21600
}
```

**O id é derivado do conteúdo do arquivo** (SHA-256) combinado com os parâmetros que
influenciam o resultado. Reenviar o mesmo áudio com as mesmas opções responde
imediatamente com `"cached": true` e HTTP 200, sem reprocessar. Mudar `num_speakers` ou
o idioma gera um id diferente e reprocessa — se o id fosse apenas o hash do arquivo, o
cache devolveria um resultado calculado sob outros parâmetros. O hash puro do arquivo
fica disponível em `file_sha256`.

### Acompanhar

```bash
curl http://localhost:8000/jobs/3f9c…
```

```json
{
  "status": "running",
  "stage": "aligning",
  "stage_label": "alinhando palavras",
  "progress_percent": 68.4,
  "elapsed_seconds": 412.7,
  "eta_seconds": 190.2,
  "expires_in_seconds": 19830
}
```

Estágios: `queued` → `decoding` → `transcribing` → `aligning` → `diarizing` → `done`.
O progresso é real, reportado pelas próprias etapas do WhisperX — inclusive a
diarização, que informa segmentação e embeddings separadamente. O `eta_seconds` aparece
a partir de 10% de progresso, onde a extrapolação passa a fazer sentido.

### Baixar

Quantas vezes quiser, enquanto o job não expirar:

```bash
curl -OJ "http://localhost:8000/jobs/3f9c…/download?formato=txt"
```

| `formato` | Conteúdo |
|---|---|
| `json` (padrão) | Resultado completo: segmentos, palavras, turnos, `by_speaker` |
| `txt` | Diálogo rotulado por falante, ou texto corrido sem diarização |
| `srt` / `vtt` | Legendas com o falante entre colchetes |

### Rotas da fila

| Rota | O que faz |
|---|---|
| `POST /jobs` | Enfileira e devolve o id (202, ou 200 se já estiver em cache) |
| `GET /jobs/{id}` | Status, progresso e ETA (`?incluir_resultado=true` embute o JSON) |
| `GET /jobs/{id}/download` | Baixa o resultado no formato escolhido |
| `GET /jobs` | Lista os jobs |
| `DELETE /jobs/{id}` | Remove o job e seus arquivos |

### Validade e persistência

Jobs e áudios vivem em `/data` e sobrevivem ao reinício do container. Passadas
`JOB_TTL_SECONDS` (6 h por padrão), **o áudio e a transcrição são apagados juntos** por
uma rotina que roda a cada 10 minutos.

Se o container for reiniciado no meio de uma transcrição, o job volta para a fila
automaticamente em vez de ficar preso — o trabalho não se perde.

Só uma transcrição roda por vez; as demais aguardam na fila, já que os modelos são
serializados de qualquer forma.

## Formato do resultado

```json
{
  "language": "pt",
  "duration": 5306.2,
  "alignment": true,
  "diarization": true,
  "num_speakers": 3,
  "speakers": ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02"],
  "text": "transcrição completa ...",
  "turns": [
    { "speaker": "SPEAKER_00", "start": 0.0, "end": 5.2, "text": "Bom dia pessoal..." }
  ],
  "by_speaker": {
    "SPEAKER_00": { "total_time": 18.4, "turns": 5, "text": "tudo que essa pessoa disse" }
  },
  "segments": [
    {
      "id": 0, "start": 0.0, "end": 5.2, "speaker": "SPEAKER_00",
      "text": "Bom dia pessoal...",
      "words": [{ "start": 0.0, "end": 0.32, "word": "Bom", "score": 0.98, "speaker": "SPEAKER_00" }]
    }
  ]
}
```

`turns` é a separação por falante já pronta e costuma ser o campo mais útil.
`by_speaker` agrega tudo que cada pessoa falou. Os turnos são cortados a partir dos
timestamps de palavra, então uma troca de falante no meio da frase é dividida no ponto
certo.

## Variáveis de ambiente

| Variável | Padrão | Descrição |
|---|---|---|
| `COMPUTE_TYPE` | `float16` | `float16` (nativo da GPU) ou `int8_float16`/`int8`, que economizam memória de vídeo em placas de 6-8 GB |
| `CPU_THREADS` | `8` | Threads do CTranslate2 |
| `OMP_NUM_THREADS` | `8` | Threads do OpenMP/torch |
| `BATCH_SIZE` | `16` | Trechos transcritos por lote. Maior é mais rápido, até faltar memória de vídeo |
| `MAX_UPLOAD_MB` | `1024` | Limite de upload |
| `JOB_TTL_SECONDS` | `21600` | Validade do job (6 h) antes da limpeza |
| `JOBS_DIR` | `/data/jobs` | Onde ficam áudios e resultados |
| `ALIGN_LANGUAGES` | `pt,en` | Idiomas com alinhamento (precisa estar embutido no build) |
| `DEVICE` | `cuda` | `cpu` roda sem placa, mas aí a variante [cpu](../cpu/README.md) é menor |

## Notas

- As bibliotecas CUDA (cuDNN 9, cuBLAS) vêm como pacotes pip junto com o PyTorch; o
  driver vem da máquina. Por isso a base continua `python:slim`, sem a imagem
  `nvidia/cuda`.
- Sem `--gpus all` (ou com o toolkit ausente) o container não enxerga a placa: ele
  avisa no log e segue na CPU, bem mais devagar. `GET /health` diz onde está rodando
  (`"device": "cuda"` ou `"cpu"`).
- Passar `num_speakers` quando você souber o número continua sendo o caminho mais
  confiável, em qualquer motor de diarização.
- O `beam_size` não é ajustável por requisição: no WhisperX ele é definido ao carregar
  o modelo.
- O alinhamento depende do `punkt` do NLTK, que normalmente é baixado na primeira
  execução. Ele vem embutido na imagem; sem isso o alinhamento falharia em silêncio
  num container sem rede, caindo nos timestamps menos precisos do Whisper.
- Na variante CPU o processamento fica em torno de 1x o tempo real com `large-v3`, e a
  separação de falantes é o gargalo. Aqui as três etapas vão para a GPU; o ganho depende
  da placa e ainda não foi medido neste projeto.

## Licença

Este projeto é distribuído sob a **GNU General Public License v3.0 ou posterior**
(veja [LICENSE](../../../LICENSE)). Uso interno — rodar e modificar dentro da sua organização —
não dispara nenhuma obrigação da GPL; ela só se aplica ao **distribuir** o software ou
um produto que o embuta, caso em que o código-fonte deve ser fornecido sob os mesmos
termos.

As dependências têm licenças próprias, todas compatíveis com a GPLv3:

| Componente | Licença |
|---|---|
| WhisperX | BSD-2-Clause |
| faster-whisper, CTranslate2, modelos Whisper, FastAPI | MIT |
| pyannote.audio | MIT |
| `pyannote/speaker-diarization-community-1` | CC-BY-4.0 |
| `jonatasgrosman/wav2vec2-large-xlsr-53-portuguese` | Apache-2.0 |
| PyTorch, torchaudio | BSD-3-Clause |
| Bibliotecas CUDA (cuDNN, cuBLAS), pacotes pip da NVIDIA | [licença NVIDIA](https://docs.nvidia.com/cuda/eula/) |
| FFmpeg | LGPL/GPL — invocado como processo externo, não vinculado ao código |

Os modelos de diarização e alinhamento embutidos na imagem são redistribuídos sob
CC-BY-4.0 e Apache-2.0, com atribuição a
[pyannote](https://huggingface.co/pyannote/speaker-diarization-community-1) e a
[jonatasgrosman](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-portuguese).
