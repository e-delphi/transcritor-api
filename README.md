# Transcritor API

Container autocontido de transcrição com **separação por falante**, construído sobre
[WhisperX](https://github.com/m-bain/whisperX). Os modelos são gravados dentro da
imagem durante o build, então o container roda **offline, sem token e sem
autenticação** — verificado com `--network none`, inclusive o alinhamento.

A imagem ocupa **7,9 GB** em disco (`docker images` reporta um número bem maior porque
o image store do containerd conta blobs comprimidos e descomprimidos).

Há também uma **instalação nativa para Windows com GPU** (AMD, NVIDIA ou Intel, via
Vulkan), que troca o motor por Qwen3-ASR e Qwen3-ForcedAligner rodando no
[audio.cpp](https://github.com/0xShug0/audio.cpp). Ela não separa falantes, mas marca
cada palavra com mais precisão e é cerca de 6× mais rápida — veja
[Motor nativo com GPU](#motor-nativo-com-gpu-windows).

| | Docker (WhisperX, CPU) | Nativo (Qwen3 + audio.cpp, GPU) |
|---|---|---|
| Separação por falante | sim | não |
| Tradução para inglês | sim | não |
| Erro mediano no início da palavra | ~65 ms (viés de +63 ms) | ~22 ms (viés de −4 ms) |
| Palavras com erro até 100 ms | 83–86% | ~98% |
| Reunião de 88 min | ~36 min | ~6 min |
| Números por extenso | como o modelo escreveu | algarismos acima de dez (pt) |

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
docker build --secret id=hf_token,src=$HOME/hf_token.txt -t transcritor-api:latest .
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
docker run -d -p 8000:8000 -v transcritor-dados:/data transcritor-api:latest
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
| `srt` / `vtt` | Legendas em frases curtas, com o falante entre colchetes quando houver |

As legendas são montadas a partir do instante de cada palavra. Por padrão cada bloco tem
**até 2 linhas de 42 caracteres** e no máximo 7 segundos, e é quebrado de preferência
no fim de frase, numa pausa da fala ou depois de vírgula — nunca deixando uma preposição
ou artigo sozinho no fim da linha. Blocos muito curtos são unidos ao seguinte, cada um
permanece meio segundo após a última palavra e nunca se sobrepõe ao próximo. Os limites
podem ser ajustados por download:

```bash
curl -OJ "http://localhost:8000/jobs/3f9c…/download?formato=srt&max_caracteres=37&max_linhas=1"
```

| Parâmetro | Padrão | Faixa |
|---|---|---|
| `max_caracteres` | `42` | 10–120 |
| `max_linhas` | `2` | 1–4 |
| `max_segundos` | `7` | 1–20 |

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
| `COMPUTE_TYPE` | `int8` | `int8`, `int8_float32`, `float32`. Mais preciso e mais lento nessa ordem |
| `CPU_THREADS` | `8` | Threads do CTranslate2 |
| `OMP_NUM_THREADS` | `8` | Threads do OpenMP/torch |
| `BATCH_SIZE` | `8` | Trechos transcritos por lote. Maior usa mais RAM |
| `MAX_UPLOAD_MB` | `1024` | Limite de upload |
| `JOB_TTL_SECONDS` | `21600` | Validade do job (6 h) antes da limpeza |
| `JOBS_DIR` | `/data/jobs` | Onde ficam áudios e resultados |
| `ALIGN_LANGUAGES` | `pt,en` | Idiomas com alinhamento (precisa estar embutido no build) |
| `DEVICE` | `cpu` | A imagem é CPU-only |
| `LEGENDA_MAX_CARACTERES` | `42` | Caracteres por linha de legenda |
| `LEGENDA_MAX_LINHAS` | `2` | Linhas por bloco de legenda |
| `LEGENDA_MAX_SEGUNDOS` | `7` | Duração máxima de um bloco |
| `LEGENDA_MIN_SEGUNDOS` | `1` | Blocos mais curtos são unidos ao seguinte, se couber |

## Motor nativo com GPU (Windows)

O Docker Desktop no Windows não repassa GPUs AMD para containers, então o motor com GPU
é instalado direto no sistema. Ele usa o servidor do audio.cpp com backend Vulkan, que
funciona com placas AMD, NVIDIA e Intel sem CUDA nem ROCm.

| Etapa | Componente |
|---|---|
| Decodificação | PyAV (FFmpeg embutido no pacote) |
| Divisão do áudio | Blocos de até 60 s, cortados no trecho de menor energia; silêncio é descartado |
| Transcrição | Qwen3-ASR 1.7B (q8_0) |
| Alinhamento de palavras | Qwen3-ForcedAligner 0.6B (q8_0) |
| Números | Por extenso → algarismos, em português |

Requisitos: Windows 10 ou 11, Python 3.11 ou mais novo e um driver de vídeo com Vulkan.
A instalação baixa ~3,7 GB (servidor e dois modelos) e confere o SHA-256 de cada arquivo.

Abra **`native\windows\instalar.bat`** com duplo clique, ou pelo terminal:

```bat
native\windows\instalar.bat
```

Tudo fica em `%LOCALAPPDATA%\TranscritorAPI` (use `-Destino` para mudar). Se os `.gguf`
já estiverem baixados, `-ModelosDe C:\pasta\dos\modelos` os aproveita sem copiar. Rodar
de novo é seguro: o que já está instalado só é conferido. Para iniciar, abra
**`native\windows\iniciar.bat`**:

```bat
native\windows\iniciar.bat -Porta 8000
```

Ele sobe o audio.cpp na porta 8081, espera ele responder e inicia a API em
`http://127.0.0.1:8000` (`-Porta` e `-Endereco` mudam isso). Ctrl+C encerra os
dois. Os `.bat` apenas chamam `install.ps1` e `start.ps1` liberando a
política de execução do PowerShell para esses scripts. As
rotas e o formato do resultado são os mesmos da versão Docker, com duas diferenças:

- `diarization` é `false` por padrão, e pedir `true` devolve HTTP 400 — o motor não
  separa falantes. `task=translate` também é recusado.
- O resultado traz `"engine": "audiocpp:qwen3"`, e `GET /health` informa o motor ativo.

**Números.** Em português, números por extenso maiores que dez viram algarismos, como
pede a convenção de legendagem: "dois mil e vinte e seis" → `2026`, "trinta e cinco por
cento" → `35%`, "um milhão e duzentos mil" → `1.200.000`. Até dez continuam por
extenso ("três pessoas"), e enumerações como "dois e três" não são somadas. O limite é
ajustável com `NUMEROS_POR_EXTENSO_ATE`.

| Variável | Padrão | Descrição |
|---|---|---|
| `ENGINE` | `whisperx` | `audiocpp` ativa este motor (o `start.ps1` já define) |
| `AUDIOCPP_URL` | `http://127.0.0.1:8081` | Endereço do servidor audio.cpp |
| `AUDIOCPP_ASR_MODEL` / `AUDIOCPP_ALIGN_MODEL` | `qwen3-asr` / `qwen3-align` | Ids dos modelos no servidor |
| `AUDIOCPP_TIMEOUT` | `900` | Tempo máximo de uma chamada ao servidor, em segundos |
| `BLOCO_MAX_SEGUNDOS` | `60` | Tamanho máximo de cada bloco enviado ao ASR |
| `NUMEROS_POR_EXTENSO_ATE` | `10` | Números até este valor ficam por extenso |

O alinhador cobre português, inglês, espanhol, francês, alemão, italiano, russo,
japonês, coreano, chinês e cantonês. Em outros idiomas o texto sai normalmente, mas o
tempo de cada palavra é estimado pela posição dentro do bloco.

## Notas

- A imagem é CPU-only de propósito, por portabilidade e tamanho. Para GPU seria preciso
  uma base CUDA e `DEVICE=cuda` — o WhisperX ganha bastante com processamento em lote
  nesse cenário.
- Passar `num_speakers` quando você souber o número continua sendo o caminho mais
  confiável, em qualquer motor de diarização.
- O `beam_size` não é ajustável por requisição: no WhisperX ele é definido ao carregar
  o modelo.
- O alinhamento depende do `punkt` do NLTK, que normalmente é baixado na primeira
  execução. Ele vem embutido na imagem; sem isso o alinhamento falharia em silêncio
  num container sem rede, caindo nos timestamps menos precisos do Whisper.
- Em CPU o processamento fica em torno de 1x o tempo real com `large-v3` — mais lento
  que a versão anterior, que não fazia alinhamento forçado nem usava o pyannote.

## Licença

Este projeto é distribuído sob a **GNU General Public License v3.0 ou posterior**
(veja [LICENSE](LICENSE)). Uso interno — rodar e modificar dentro da sua organização —
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
| FFmpeg | LGPL/GPL — invocado como processo externo, não vinculado ao código |
| audio.cpp (instalação nativa) | Apache-2.0 |
| Qwen3-ASR, Qwen3-ForcedAligner (instalação nativa) | Apache-2.0 |
| PyAV (instalação nativa) | BSD-3-Clause |

Os modelos de diarização e alinhamento embutidos na imagem são redistribuídos sob
CC-BY-4.0 e Apache-2.0, com atribuição a
[pyannote](https://huggingface.co/pyannote/speaker-diarization-community-1) e a
[jonatasgrosman](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-portuguese).
