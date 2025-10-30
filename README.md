# FaceExtractor

Este projeto Python utiliza as bibliotecas `OpenCV` e `MediaPipe` para detectar rostos em imagens, recortá-los e salvá-los em um diretório separado. Ele é otimizado para performance e qualidade, suportando caminhos de arquivo com caracteres Unicode.

## Funcionalidades

*   **Detecção de Rostos:** Identifica múltiplos rostos em uma imagem usando o MediaPipe.
*   **Recorte Inteligente:** Recorta os rostos detectados com um fator de expansão configurável para incluir mais contexto (ombros, cabelo) e um ajuste vertical para melhor centralização.
*   **Redimensionamento:** Redimensiona os rostos recortados para um tamanho uniforme especificado.
*   **Processamento em Lote:** Processa todas as imagens suportadas em um diretório de entrada, incluindo subdiretórios.
*   **Suporte a Unicode:** Lida corretamente com caminhos de arquivo que contêm caracteres especiais ou acentuados (especialmente útil no Windows).

## Pré-requisitos

Certifique-se de ter o Python 3.8 ou superior instalado em seu sistema (ele foi criado usando como base o Python 3.12).

## Instalação

Siga os passos abaixo para configurar e instalar as dependências do projeto:

1.  **Crie e Ative um Ambiente Virtual (Recomendado):**
    É uma boa prática isolar as dependências do projeto.

    ```bash
    python -m venv .venv
    # No Windows:
    .\.venv\Scripts\activate
    # No macOS/Linux:
    source ./.venv/bin/activate
    ```

2.  **Instale as Dependências:**
    Com o ambiente virtual ativado, instale as bibliotecas necessárias usando o `requirements.txt`:

    ```bash
    pip install -r requirements.txt
    ```

## Como Usar

1.  **Prepare suas Imagens:**
    Coloque as imagens (formatos suportados: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`) das quais você deseja extrair rostos na pasta `FacesToExtract/`. Você pode criar subpastas dentro dela se desejar organizar suas imagens.

2.  **Configure o Script (Opcional):**
    Abra o arquivo `FaceExtractor.py` e ajuste as variáveis na seção `--- Configurações ---` dentro da função `main()` conforme suas necessidades:

    *   `INPUT_DIR`: Diretório de entrada das imagens (padrão: `FacesToExtract`).
    *   `OUTPUT_DIR`: Diretório onde os rostos recortados serão salvos (padrão: `FacesExtracted`).
    *   `CROP_SIZE`: Tamanho final (largura, altura) dos rostos recortados (padrão: `(400, 400)`).
    *   `EXPANSION_FACTOR`: Fator de "zoom" ao redor do rosto. Valores menores (ex: `1.8`) dão mais zoom no rosto; valores maiores (ex: `2.5`) incluem mais contexto.
    *   `Y_OFFSET_FACTOR`: Ajuste vertical do centro do recorte. Um valor positivo (ex: `0.2`) move o recorte para cima.
    *   `RESIZE_WIDTH_FOR_DETECTION`: Largura para redimensionar a imagem *apenas para a detecção*. Valores menores são mais rápidos, mas podem reduzir a precisão em imagens muito grandes.

3.  **Execute o Script:**

    ```bash
    # Certifique-se de que seu ambiente virtual está ativado
    python FaceExtractor.py
    ```
    Ou, se você estiver no Windows, pode usar o arquivo `run.bat` (certifique-se de que ele ativa o ambiente virtual e executa o script Python).

4.  **Verifique os Resultados:**
    Os rostos detectados e recortados serão salvos na pasta `FacesExtracted/` (ou no diretório que você configurou em `OUTPUT_DIR`). Cada rosto será salvo como um arquivo separado, com um nome indicando a imagem original e o número do rosto.

## Estrutura do Projeto

```
.
├── FaceExtractor.py          # Script principal para detecção e recorte de rostos
├── requirements.txt          # Lista de dependências do Python
├── run.bat                   # Script de execução para Windows (opcional)
├── .venv/                    # Ambiente virtual (ignorado pelo Git)
├── FacesExtracted/           # Diretório de saída para os rostos recortados
└── FacesToExtract/           # Diretório de entrada para as imagens a serem processadas
```
