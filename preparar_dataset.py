from collections import defaultdict
from pathlib import Path
import random
import shutil

RAIZ = Path(__file__).resolve().parent
ORIGEM = RAIZ / "archive"
DESTINO = RAIZ / "datasets" / "sentinela"
SEMENTE = 42


def ler_classes():
    linhas = (ORIGEM / "classes.txt").read_text(encoding="utf-8").splitlines()
    return [linha.strip() for linha in linhas if linha.strip()]


def ler_caixas(caminho):
    caixas = []
    for linha in caminho.read_text(encoding="utf-8").splitlines():
        if not linha.strip():
            continue
        classe, x, y, largura, altura = linha.split()
        valores = [max(0.0, min(1.0, float(valor))) for valor in (x, y, largura, altura)]
        caixas.append((int(classe), *valores))
    return caixas


def carregar_imagens():
    imagens = {
        caminho.stem: caminho
        for caminho in (ORIGEM / "images").rglob("*")
        if caminho.suffix.lower() in {".jpg", ".jpeg", ".png"}
    }
    rotulos = {caminho.stem: caminho for caminho in (ORIGEM / "labels").rglob("*.txt")}
    itens = []
    for nome, imagem in imagens.items():
        caixas = ler_caixas(rotulos[nome])
        classes = {caixa[0] for caixa in caixas}
        itens.append((nome, imagem, caixas, classes))
    return itens


def dividir(itens):
    contagem = defaultdict(int)
    for _, _, _, classes in itens:
        for classe in classes:
            contagem[classe] += 1

    grupos = defaultdict(list)
    for item in itens:
        classe_mais_rara = min(item[3], key=lambda classe: (contagem[classe], classe))
        grupos[classe_mais_rara].append(item)

    aleatorio = random.Random(SEMENTE)
    treino, validacao, teste = [], [], []
    for grupo in grupos.values():
        aleatorio.shuffle(grupo)
        total = len(grupo)
        n_teste = 1 if total >= 8 else 0
        n_validacao = 1 if total >= 4 else 0
        restante = total - n_teste - n_validacao
        n_teste += round(restante * 0.10)
        n_validacao += round(restante * 0.20)
        if n_teste + n_validacao >= total:
            n_teste = 1 if total >= 8 else 0
            n_validacao = 1 if total >= 4 else 0
        teste.extend(grupo[:n_teste])
        validacao.extend(grupo[n_teste:n_teste + n_validacao])
        treino.extend(grupo[n_teste + n_validacao:])
    return treino, validacao, teste


def gravar(partes, classes):
    if DESTINO.exists():
        shutil.rmtree(DESTINO)
    for nome_parte, itens in partes.items():
        pasta_imagens = DESTINO / "images" / nome_parte
        pasta_rotulos = DESTINO / "labels" / nome_parte
        pasta_imagens.mkdir(parents=True)
        pasta_rotulos.mkdir(parents=True)
        for nome, imagem, caixas, _ in itens:
            shutil.copy2(imagem, pasta_imagens / imagem.name)
            linhas = [
                f"{classe} {x:.6f} {y:.6f} {largura:.6f} {altura:.6f}"
                for classe, x, y, largura, altura in caixas
            ]
            (pasta_rotulos / f"{nome}.txt").write_text("\n".join(linhas) + "\n", encoding="utf-8")

    nomes = "\n".join(f"  {indice}: {nome}" for indice, nome in enumerate(classes))
    conteudo = (
        "path: datasets/sentinela\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        f"nc: {len(classes)}\n"
        "names:\n"
        f"{nomes}\n"
    )
    (RAIZ / "data.yaml").write_text(conteudo, encoding="utf-8")


def resumo(partes, classes):
    print(f"{'classe':<14} {'treino':>7} {'val':>7} {'teste':>7}")
    for indice, nome in enumerate(classes):
        valores = []
        for itens in partes.values():
            valores.append(sum(1 for item in itens if indice in item[3]))
        print(f"{nome:<14} {valores[0]:>7} {valores[1]:>7} {valores[2]:>7}")
    for nome, itens in partes.items():
        print(nome, len(itens))


def main():
    classes = ler_classes()
    treino, validacao, teste = dividir(carregar_imagens())
    partes = {"train": treino, "val": validacao, "test": teste}
    gravar(partes, classes)
    resumo(partes, classes)


if __name__ == "__main__":
    main()
