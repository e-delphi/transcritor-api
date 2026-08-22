# Transcritor API

Container autocontido de transcrição com **separação por falante**.
Sem token, sem autenticação, sem download de modelo em runtime: os modelos são
gravados dentro da imagem no build e o container roda **totalmente offline**.

## Como funciona

| Etapa | Componente |
|---|---|
| Decodificação de áudio/vídeo | FFmpeg — qualquer formato vira PCM mono 16 kHz |
| Transcrição | `faster-whisper` (CTranslate2) com `large-v3` embutido na imagem |
| Impressão digital de voz | ECAPA-TDNN via SpeechBrain (`spkrec-ecapa-voxceleb`) |
| Agrupamento de falantes | clustering aglomerativo (scikit-learn) |
| API | FastAPI + Uvicorn |

Nenhum dos modelos é *gated*: não há token HuggingFace envolvido, nem no build nem
em runtime. Depois de construída, a imagem funciona com `--network none`.

> **Por que não `openai-whisper` nem `pyannote.audio`:** o primeiro baixa o modelo no
> uso inicial e o segundo exige um token HuggingFace (o repositório é *gated*). Os dois
> inviabilizariam uma imagem autocontida que roda offline.

## Estrutura

```
app/
  main.py         API HTTP e validação de entrada
  pipeline.py     orquestra transcrição + diarização, monta o JSON
  diarization.py  embeddings ECAPA e agrupamento de falantes
  audio_io.py     decodificação via FFmpeg
docker/
  download_models.py  grava os modelos na imagem durante o build
  fix_execstack.py    corrige .so com stack executável (ver Notas)
Dockerfile
docker-compose.yml
requirements.txt
```

## Build

```bash
docker build -t transcritor-api:latest .
```

O modelo padrão é **`large-v3`**, o mais preciso disponível. Para trocar
(`tiny`, `base`, `small`, `medium`, `large-v2`, `large-v3`):

```bash
docker build --build-arg WHISPER_MODEL=small -t transcritor-api:small .
```

| Modelo | Peso do modelo | Tamanho em disco | Velocidade (CPU, 16 núcleos) |
|---|---|---|---|
| `tiny` | 75 MB | ~2,2 GB | muito rápida |
| `base` | 145 MB | ~2,3 GB | muito rápida |
| `small` | 480 MB | **2,6 GB** (medido) | **~4x tempo real** (medido) |
| `medium` | 1,5 GB | ~3,6 GB | ~2,5x tempo real |
| `large-v3` (padrão) | 2,9 GB | **5,0 GB** (medido) | **~1,5x tempo real** (medido) |

Medições feitas nesta máquina com um áudio de 35,6 s: `small` levou 9 s, `large-v3`
levou 24 s. Ambos são mais rápidos que o tempo real. As linhas `tiny`/`base`/`medium`
são estimadas a partir do peso do modelo.

> **Sobre o tamanho:** `docker images` reporta 8,94 GB para a `large-v3`, mas o
> sistema de arquivos real do container tem 5,0 GB — o image store do containerd
> contabiliza blobs comprimidos e descomprimidos. O número que vale para disco e
> transferência é o menor.

O piso de ~2,1 GB vem das dependências (torch CPU, ffmpeg, ctranslate2, onnxruntime).

## Executar

```bash
docker run -d --name transcritor-api -p 8000:8000 transcritor-api:latest
```

Ou com compose:

```bash
docker compose up -d --build
```

O primeiro `GET /health` responde `ok` assim que os modelos terminam de carregar (alguns segundos).

## Rota principal

`POST /transcribe` — `multipart/form-data`

| Campo | Tipo | Padrão | Descrição |
|---|---|---|---|
| `file` | arquivo | **obrigatório** | Áudio ou vídeo (mp3, wav, m4a, ogg, flac, mp4, mkv…) |
| `language` | string | auto | Código ISO (`pt`, `en`…). Vazio = detecção automática |
| `task` | string | `transcribe` | `transcribe` ou `translate` (traduz para inglês) |
| `diarization` | bool | `true` | Separar por falante |
| `num_speakers` | int | auto | Número exato de falantes, se conhecido — melhora bastante o resultado |
| `min_speakers` | int | `1` | Piso da busca automática |
| `max_speakers` | int | `8` | Teto da busca automática |
| `beam_size` | int | `5` | Beam search do Whisper |
| `initial_prompt` | string | — | Contexto/vocabulário para orientar a transcrição |

### Exemplo

```bash
curl -X POST http://localhost:8000/transcribe -F "file=@reuniao.mp3" -F "language=pt"
```

Com número de falantes conhecido:

```bash
curl -X POST http://localhost:8000/transcribe -F "file=@reuniao.mp3" -F "num_speakers=2"
```

### Resposta

```json
{
  "language": "pt",
  "language_probability": 0.99,
  "duration": 42.31,
  "diarization": true,
  "num_speakers": 2,
  "speakers": ["SPEAKER_00", "SPEAKER_01"],
  "text": "transcrição completa ...",
  "turns": [
    { "speaker": "SPEAKER_00", "start": 0.0,  "end": 5.2,  "text": "Bom dia pessoal..." },
    { "speaker": "SPEAKER_01", "start": 5.4,  "end": 11.8, "text": "Terminei a primeira versão..." }
  ],
  "by_speaker": {
    "SPEAKER_00": { "total_time": 18.4, "turns": 5, "text": "tudo que o falante 0 disse" },
    "SPEAKER_01": { "total_time": 21.9, "turns": 4, "text": "tudo que o falante 1 disse" }
  },
  "segments": [
    {
      "id": 0, "start": 0.0, "end": 5.2, "speaker": "SPEAKER_00",
      "text": "Bom dia pessoal...",
      "avg_logprob": -0.21, "no_speech_prob": 0.01,
      "words": [{ "start": 0.0, "end": 0.32, "word": " Bom", "probability": 0.98, "speaker": "SPEAKER_00" }]
    }
  ]
}
```

- **`turns`** é a separação por falante já pronta, em ordem cronológica — normalmente é o campo que você quer consumir.
- **`by_speaker`** agrega todo o texto de cada pessoa.
- **`segments`** traz o detalhe do Whisper com timestamps por palavra.

Turnos são construídos a partir dos timestamps de palavra, então uma troca de falante no meio de uma frase é dividida corretamente.

## Outras rotas

- `GET /health` — status
- `GET /docs` — Swagger UI interativo (dá para enviar o arquivo pelo navegador)

## Variáveis de ambiente

| Variável | Padrão | Descrição |
|---|---|---|
| `COMPUTE_TYPE` | `int8` | `int8`, `int8_float32`, `float32`. `float32` é mais preciso e mais lento |
| `CPU_THREADS` | `8` | Threads do CTranslate2 |
| `OMP_NUM_THREADS` | `8` | Threads do OpenMP/torch |
| `MAX_UPLOAD_MB` | `512` | Limite de upload |
| `DEVICE` | `cpu` | A imagem é CPU-only |
| `DIARIZATION_THRESHOLD` | `0.65` | Distância cosseno do corte de clusters. Menor = mais falantes; maior = menos |

Exemplo:

```bash
docker run -d -p 8000:8000 -e COMPUTE_TYPE=float32 -e OMP_NUM_THREADS=8 transcritor-api:latest
```

## Desempenho

Medido com `large-v3`, 16 núcleos, `COMPUTE_TYPE=int8`:

| | |
|---|---|
| Carga dos modelos (startup) | 37 s — o `/health` só responde depois disso |
| RAM em uso | ~2,0 GB |
| Transcrição | ~1,5x mais rápido que o tempo real |

Reduzir `beam_size` de 5 para 1 economiza apenas ~8% (23 s → 21 s), porque o gargalo
está no encoder, não na busca — não vale a perda de precisão. Se precisar de mais
velocidade, troque o modelo, não o `beam_size`.

## Notas

- Requisições são serializadas dentro do container (os modelos não são thread-safe). Para paralelismo, suba várias réplicas atrás de um balanceador.
- A diarização automática decide o número de falantes cortando o dendrograma pela
  distância `DIARIZATION_THRESHOLD`. Esse valor foi calibrado com áudio sintético de
  duas vozes, não com um benchmark de fala real — trate-o como ponto de partida e
  ajuste se o resultado dividir ou fundir falantes no seu material. **Passe
  `num_speakers` sempre que souber o número**: é o que dá o melhor resultado.
- Trechos muito curtos (menos de ~3 janelas de fala) são absorvidos pelo falante mais
  parecido, então interjeições isoladas podem ser atribuídas ao interlocutor.
- A imagem é CPU-only de propósito (portabilidade e tamanho). Para GPU seria preciso uma base CUDA e `DEVICE=cuda`.

## Licença

Este projeto é distribuído sob a **GNU General Public License v3.0 ou posterior**
(veja [LICENSE](LICENSE)). Uso interno — rodar e modificar dentro da sua organização —
não dispara nenhuma obrigação da GPL; ela só se aplica ao **distribuir** o software ou
um produto que o embuta, caso em que o código-fonte deve ser fornecido sob os mesmos
termos.

As dependências têm licenças próprias, todas permissivas e compatíveis com a GPLv3:

| Componente | Licença |
|---|---|
| faster-whisper, CTranslate2, modelos Whisper, FastAPI | MIT |
| SpeechBrain, `spkrec-ecapa-voxceleb` | Apache-2.0 |
| PyTorch, scikit-learn | BSD-3-Clause |
| FFmpeg | LGPL/GPL — invocado como processo externo, não vinculado ao código |
