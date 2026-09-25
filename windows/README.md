# Transcritor API — Windows nativo (Qwen3 + audio.cpp)

API de transcrição e **legendas** que roda direto no Windows, usando a GPU por Vulkan —
placas AMD, NVIDIA e Intel, sem CUDA nem ROCm. O motor é o Qwen3-ASR com o
Qwen3-ForcedAligner, servidos pelo [audio.cpp](https://github.com/0xShug0/audio.cpp).

O foco é o **instante de cada palavra**: nos testes, o início das palavras errou ~22 ms
na mediana, e ~98% ficaram a menos de 100 ms. Uma reunião de 88 min leva cerca de 7 min
numa RX 9070 XT. Não há separação por falante — para isso, use o
[projeto Docker](../README.md), na raiz deste repositório, que é independente deste.

## Como funciona

| Etapa | Componente |
|---|---|
| Decodificação | PyAV (FFmpeg embutido no pacote) — qualquer áudio ou vídeo |
| Divisão do áudio | Blocos de até 60 s, cortados no trecho de menor energia; silêncio é descartado |
| Transcrição | Qwen3-ASR 1.7B (q8_0) |
| Alinhamento de palavras | Qwen3-ForcedAligner 0.6B (q8_0) |
| Números | Por extenso → algarismos, em português |
| API e fila | FastAPI + Uvicorn |

A inferência roda no `audiocpp_server.exe`, um programa nativo em C++. A API é Python,
num ambiente isolado dentro da pasta de instalação, sem PyTorch.

## Instalar

Requisitos: Windows 10 ou 11, **Python 3.11 ou mais novo** e um driver de vídeo com
Vulkan. A instalação baixa ~3,7 GB (servidor e dois modelos) e confere o SHA-256 de
cada arquivo; se algum não bater, ela para.

Abra **`instalar.bat`** com duplo clique, ou pelo terminal:

```bat
instalar.bat
```

Tudo fica em `%LOCALAPPDATA%\TranscritorAPI`:

| Pasta | Conteúdo |
|---|---|
| `audiocpp\` | Servidor audio.cpp (Vulkan) |
| `models\` | Qwen3-ASR e Qwen3-ForcedAligner (`.gguf`) |
| `venv\` | Python isolado com as dependências da API |
| `data\jobs\` | Áudios e transcrições da fila |
| `logs\` | Registros do audio.cpp |

O código da API não é copiado: ele roda desta pasta (`windows\app`), cujo caminho fica
gravado em `repo.txt`. Se mover o repositório, rode `instalar.bat` de novo. Rodar de
novo é sempre seguro — o que já está instalado só é conferido.

| Parâmetro | Descrição |
|---|---|
| `-Destino C:\pasta` | Instala em outro lugar |
| `-ModelosDe C:\pasta` | Aproveita `.gguf` já baixados, sem copiar (vínculo no mesmo disco) |

## Iniciar

Abra **`iniciar.bat`**. Ele sobe o audio.cpp na porta 8081, espera ele responder e
inicia a API em `http://127.0.0.1:8000`. Ctrl+C encerra os dois.

```bat
iniciar.bat -Porta 8000 -Endereco 127.0.0.1
```

Por padrão a API só aceita conexões desta máquina. Para uma aplicação em outra máquina
ou num container, use `-Endereco 0.0.0.0` (de dentro de um container Docker, o endereço
é `http://host.docker.internal:8000`). A documentação interativa fica em `/docs`.

Os `.bat` só chamam `install.ps1` e `start.ps1` liberando a política de execução do
PowerShell para esses scripts.

## Rotas

| Rota | O que faz |
|---|---|
| `POST /transcribe` | Transcreve e devolve o JSON na hora. Boa para áudios curtos |
| `POST /jobs` | Enfileira e devolve o id (202, ou 200 se já estiver em cache) |
| `GET /jobs/{id}` | Status, progresso e ETA (`?incluir_resultado=true` embute o JSON) |
| `GET /jobs/{id}/download` | Baixa o resultado: `formato=json`, `txt`, `srt` ou `vtt` |
| `GET /jobs` | Lista os jobs |
| `DELETE /jobs/{id}` | Remove o job e seus arquivos |
| `GET /health` | `{"status": "ok", "engine": "audiocpp"}` |

Campos de `POST /transcribe` e `POST /jobs` (`multipart/form-data`):

| Campo | Padrão | Descrição |
|---|---|---|
| `file` | **obrigatório** | Áudio ou vídeo (mp3, wav, m4a, ogg, flac, mp4, mkv…) |
| `language` | auto | Código ISO (`pt`, `en`…). Vazio = detecção automática |
| `alignment` | `true` | Marcar o instante de cada palavra |
| `diarization` | `false` | Não suportado: `true` devolve HTTP 400 |
| `task` | `transcribe` | `translate` não é suportado e devolve HTTP 400 |

```bash
curl -X POST http://localhost:8000/jobs -F "file=@reuniao.m4a" -F "language=pt"
curl http://localhost:8000/jobs/3f9c…
curl -OJ "http://localhost:8000/jobs/3f9c…/download?formato=srt"
```

O id do job é o SHA-256 do arquivo combinado com as opções: reenviar o mesmo áudio
responde na hora com `"cached": true`. Estágios: `queued` → `decoding` →
`transcribing` → `aligning` → `done`. Passadas 6 h (`JOB_TTL_SECONDS`), o áudio e a
transcrição são apagados juntos. Só uma transcrição roda por vez.

## Legendas

As legendas `srt` e `vtt` são montadas a partir do instante de cada palavra. Por padrão
cada bloco tem **até 2 linhas de 42 caracteres** e no máximo 7 segundos, e é quebrado de
preferência no fim de frase, numa pausa da fala ou depois de vírgula — nunca deixando
uma preposição ou artigo sozinho no fim da linha. Blocos muito curtos são unidos ao
seguinte, cada um permanece meio segundo após a última palavra e nunca se sobrepõe ao
próximo.

```bash
curl -OJ "http://localhost:8000/jobs/3f9c…/download?formato=srt&max_caracteres=37&max_linhas=1"
```

| Parâmetro | Padrão | Faixa |
|---|---|---|
| `max_caracteres` | `42` | 10–120 |
| `max_linhas` | `2` | 1–4 |
| `max_segundos` | `7` | 1–20 |

## Números

Em português, números por extenso maiores que dez viram algarismos, como pede a
convenção de legendagem: "dois mil e vinte e seis" → `2026`, "trinta e cinco por cento"
→ `35%`, "um milhão e duzentos mil" → `1.200.000`. Até dez continuam por extenso ("três
pessoas"), e enumerações como "dois e três" não são somadas.

## Formato do resultado

```json
{
  "engine": "audiocpp:qwen3",
  "language": "pt",
  "duration": 5306.2,
  "alignment": true,
  "diarization": false,
  "num_speakers": 0,
  "speakers": [],
  "turns": [],
  "by_speaker": {},
  "text": "transcrição completa ...",
  "segments": [
    {
      "id": 0, "start": 0.0, "end": 5.2, "speaker": null, "aligned": true,
      "text": "Bom dia, pessoal.",
      "words": [{ "start": 0.0, "end": 0.32, "word": "Bom" }]
    }
  ]
}
```

É o mesmo formato do projeto Docker com `diarization=false`; os campos de falante
existem, mas vêm vazios. O alinhador cobre português, inglês, espanhol, francês, alemão,
italiano, russo, japonês, coreano, chinês e cantonês. Em outros idiomas o texto sai
normalmente, mas o instante de cada palavra é estimado pela posição dentro do bloco.

## Variáveis de ambiente

O `iniciar.bat` já define as necessárias; estas servem para ajuste fino.

| Variável | Padrão | Descrição |
|---|---|---|
| `AUDIOCPP_URL` | `http://127.0.0.1:8081` | Endereço do servidor audio.cpp |
| `AUDIOCPP_ASR_MODEL` / `AUDIOCPP_ALIGN_MODEL` | `qwen3-asr` / `qwen3-align` | Ids dos modelos no servidor |
| `AUDIOCPP_TIMEOUT` | `900` | Tempo máximo de uma chamada ao servidor, em segundos |
| `BLOCO_MAX_SEGUNDOS` | `60` | Tamanho máximo de cada bloco enviado ao ASR |
| `NUMEROS_POR_EXTENSO_ATE` | `10` | Números até este valor ficam por extenso |
| `LEGENDA_MAX_CARACTERES` | `42` | Caracteres por linha de legenda |
| `LEGENDA_MAX_LINHAS` | `2` | Linhas por bloco de legenda |
| `LEGENDA_MAX_SEGUNDOS` | `7` | Duração máxima de um bloco |
| `LEGENDA_MIN_SEGUNDOS` | `1` | Blocos mais curtos são unidos ao seguinte, se couber |
| `JOB_TTL_SECONDS` | `21600` | Validade do job (6 h) antes da limpeza |
| `MAX_UPLOAD_MB` | `1024` | Limite de upload |

## Licença

GPL-3.0 ou posterior — veja o [LICENSE](../LICENSE) na raiz do repositório.
As dependências têm licenças próprias, todas compatíveis com a GPLv3:

| Componente | Licença |
|---|---|
| audio.cpp | Apache-2.0 |
| Qwen3-ASR, Qwen3-ForcedAligner | Apache-2.0 |
| FastAPI, Uvicorn | MIT / BSD-3-Clause |
| PyAV | BSD-3-Clause |
| NumPy | BSD-3-Clause |
