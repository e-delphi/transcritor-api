#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Eduardo

"""Remove a marcação de "stack executável" das bibliotecas nativas instaladas.

Algumas wheels (notadamente ctranslate2) embarcam objetos em assembly sem a nota
`.note.GNU-stack`, e o linker acaba marcando o segmento PT_GNU_STACK como
executável. Kernels recentes com hardening — incluindo o do WSL2 usado pelo
Docker Desktop — recusam carregar essas bibliotecas com:

    ImportError: libctranslate2-*.so: cannot enable executable stack
                 as shared object requires: Invalid argument

Nenhuma dessas bibliotecas realmente precisa de stack executável (isso só é
necessário para trampolines de funções aninhadas em C), então limpar o bit PF_X
é seguro e equivale ao que o utilitário `execstack -c` faria — que não é mais
empacotado no Debian.
"""

import os
import struct
import sys

ELF_MAGIC = b"\x7fELF"
PT_GNU_STACK = 0x6474E551
PF_X = 0x1


def clear_execstack(path: str) -> bool:
    """Zera PF_X no PT_GNU_STACK do ELF. Retorna True se algo foi alterado."""
    with open(path, "r+b") as elf:
        header = elf.read(64)
        if len(header) < 64 or header[:4] != ELF_MAGIC:
            return False
        if header[4] != 2:  # somente ELF64
            return False

        little = header[5] == 1
        endian = "<" if little else ">"

        e_phoff = struct.unpack_from(f"{endian}Q", header, 0x20)[0]
        e_phentsize = struct.unpack_from(f"{endian}H", header, 0x36)[0]
        e_phnum = struct.unpack_from(f"{endian}H", header, 0x38)[0]
        if not e_phoff or not e_phentsize:
            return False

        for i in range(e_phnum):
            entry = e_phoff + i * e_phentsize
            elf.seek(entry)
            raw = elf.read(8)
            if len(raw) < 8:
                break
            p_type, p_flags = struct.unpack(f"{endian}II", raw)
            if p_type != PT_GNU_STACK or not p_flags & PF_X:
                continue
            elf.seek(entry + 4)
            elf.write(struct.pack(f"{endian}I", p_flags & ~PF_X))
            return True
    return False


def main(roots) -> int:
    patched = []
    for root in roots:
        for dirpath, _, filenames in os.walk(root):
            for name in filenames:
                if ".so" not in name:
                    continue
                path = os.path.join(dirpath, name)
                if os.path.islink(path):
                    continue
                try:
                    if clear_execstack(path):
                        patched.append(path)
                except (OSError, struct.error) as exc:
                    print(f"aviso: {path}: {exc}", file=sys.stderr)

    for path in patched:
        print(f"execstack removido: {path}")
    print(f"==> {len(patched)} biblioteca(s) ajustada(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or ["/opt/venv"]))
