# py -3.12 -m venv .venv
# .\.venv\Scripts\Activate.ps1
# pip install opencv-python mediapipe numpy
# pip freeze > requirements.txt
# deactivate
# pip install -r requirements.txt

import cv2 # Biblioteca OpenCV para processamento de imagens
import mediapipe as mp # Biblioteca MediaPipe para detecção de rostos
import numpy as np # Para manipulação de arrays e imagens
from pathlib import Path # Para manipulação de caminhos
import sys # Para mensagens de erro no stderr
from typing import Tuple, Union # Tipos para anotações

# Define um alias de tipo para clareza
Image = np.ndarray # Imagem representada como um array NumPy
PathLike = Union[str, Path] # Caminho como string ou Path
CropSize = Tuple[int, int] # Tamanho do recorte (largura, altura)

class FaceDetectorCropper:
    """
    Detecta rostos em imagens usando MediaPipe, aplica um recorte quadrado
    centralizado (com ajustes) e salva as imagens recortadas.

    Otimizado para:
    - Performance (detecção em imagem reduzida).
    - Qualidade (recorte da imagem original em alta resolução).
    - Robustez (suporte a caminhos com caracteres Unicode).
    """

    def __init__(
        self,
        resize_for_detection: int = 800,
        model_selection: int = 0,
        min_confidence: float = 0.5,
    ):
        """
        Inicializa o detector de rostos.

        Args:
            resize_for_detection: Largura para redimensionar a imagem antes da detecção.
                                Valores menores = mais rápido.
            model_selection: 0 para curto alcance (<2m), 1 para longo alcance (<5m).
            min_confidence: Confiança mínima da detecção (0.0 a 1.0).
        """
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=model_selection,
            min_detection_confidence=min_confidence,
        )
        self.resize_for_detection = resize_for_detection

    def _load_image_unicode(self, image_path: Path) -> Image | None:
        """Carrega uma imagem de um caminho (suporta Unicode no Windows)."""
        if not image_path.exists():
            print(
                f"Erro: Arquivo não encontrado {image_path}", file=sys.stderr
            )
            return None
        try:
            # Lê os bytes do arquivo e decodifica com o OpenCV
            with open(image_path, "rb") as f:
                file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if image is None:
                print(
                    f"Erro: Não foi possível decodificar a imagem {image_path}",
                    file=sys.stderr,
                )
            return image
        except Exception as e:
            print(
                f"Erro ao ler a imagem {image_path}: {e}", file=sys.stderr
            )
            return None

    def _save_image_unicode(self, image: Image, output_path: Path) -> bool:
        """Salva a imagem em um caminho (suporta Unicode no Windows)."""
        try:
            # Encoda a imagem para o formato (ex: .jpg) e salva os bytes
            ext = output_path.suffix
            is_success, buffer = cv2.imencode(ext, image)
            if not is_success:
                print(
                    f"Erro: Falha ao encodar imagem para {ext}",
                    file=sys.stderr,
                )
                return False

            with open(output_path, "wb") as f:
                f.write(buffer)
            return True
        except Exception as e:
            print(
                f"Erro ao salvar imagem {output_path}: {e}", file=sys.stderr
            )
            return False

    def _get_interpolation_method(
        self, current_width: int, target_width: int
    ) -> int:
        """Escolhe o método de interpolação ideal."""
        if current_width > target_width:
            # Diminuindo (Downscaling): INTER_AREA é melhor para evitar serrilhado
            return cv2.INTER_AREA
        elif current_width < target_width:
            # Aumentando (Upscaling): INTER_CUBIC é suave
            return cv2.INTER_CUBIC
        else:
            # Tamanho exato
            return cv2.INTER_LINEAR

    def resize_image_for_processing(
        self, image: Image
    ) -> tuple[Image, float]:
        """Reduz a imagem para detecção, mantendo a proporção."""
        height, width = image.shape[:2]

        if width <= self.resize_for_detection:
            return image, 1.0

        scale_factor = self.resize_for_detection / width
        new_width = self.resize_for_detection
        new_height = int(height * scale_factor)

        resized_image = cv2.resize(
            image, (new_width, new_height), interpolation=cv2.INTER_AREA
        )

        print(
            f"   - Imagem reduzida de {width}x{height} para {new_width}x{new_height}"
        )
        return resized_image, scale_factor

    def detect_and_crop_face(
        self,
        image_path: Path,
        output_dir: Path,
        crop_size: CropSize = (400, 400),
        expansion_factor: float = 1.8,
        y_offset_factor: float = 0.10,
    ) -> bool:
        """
        Detecta o rosto principal, recorta e salva a imagem.

        Args:
            image_path: Caminho (Path) para a imagem de entrada.
            output_dir: Caminho (Path) para o diretório de saída.
            crop_size: Tamanho final do recorte (largura, altura).
            expansion_factor: Fator de zoom. Menor = mais zoom (ex: 1.8). Maior = mais contexto (ex: 2.5).
            y_offset_factor: Ajuste vertical. Positivo = move o recorte para cima (ex: 0.10 = 10% da altura do rosto).
        
        Returns:
            True se um ou mais rostos foram salvos com sucesso.
        """

        original_image = self._load_image_unicode(image_path)
        if original_image is None:
            return False

        # 1. Redimensiona para detecção rápida
        detection_image, scale_factor = self.resize_image_for_processing(
            original_image
        )

        # 2. Detecta rostos (MediaPipe usa RGB)
        rgb_image = cv2.cvtColor(detection_image, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_image)

        if not results.detections:
            print(f"   - Nenhum rosto detectado em {image_path.name}")
            return False

        output_dir.mkdir(exist_ok=True)

        orig_height, orig_width = original_image.shape[:2]
        det_height, det_width = detection_image.shape[:2]

        save_count = 0
        for i, detection in enumerate(results.detections):
            # 3. Obtém coordenadas relativas (0.0 a 1.0) da imagem DE DETECÇÃO
            bbox = detection.location_data.relative_bounding_box

            # 4. Converte para coordenadas absolutas na imagem DE DETECÇÃO
            x_det = int(bbox.xmin * det_width)
            y_det = int(bbox.ymin * det_height)
            w_det = int(bbox.width * det_width)
            h_det = int(bbox.height * det_height)

            # 5. Escala coordenadas para a imagem ORIGINAL (alta resolução)
            x = int(x_det / scale_factor)
            y = int(y_det / scale_factor)
            w = int(w_det / scale_factor)
            h = int(h_det / scale_factor)

            # 6. Calcula centro do rosto (com ajuste vertical)
            center_x = x + w // 2
            center_y = y + h // 2
            center_y = int(
                center_y - h * y_offset_factor
            )  # Ajuste vertical

            # 7. Define o tamanho do recorte (quadrado)
            base_dim = max(w, h)
            final_dim = int(base_dim * expansion_factor)

            # 8. Calcula coordenadas ideais do recorte
            ideal_x1 = center_x - final_dim // 2
            ideal_y1 = center_y - final_dim // 2
            ideal_x2 = ideal_x1 + final_dim
            ideal_y2 = ideal_y1 + final_dim

            # 9. Ajusta coordenadas para os limites da imagem
            crop_x1 = max(0, ideal_x1)
            crop_y1 = max(0, ideal_y1)
            crop_x2 = min(orig_width, ideal_x2)
            crop_y2 = min(orig_height, ideal_y2)

            # 10. Recorta da imagem ORIGINAL
            cropped_region = original_image[crop_y1:crop_y2, crop_x1:crop_x2]

            if cropped_region.size == 0:
                print(
                    f"   - Erro: Recorte inválido (tamanho zero) para rosto {i+1}",
                    file=sys.stderr,
                )
                continue

            # 11. Adiciona padding (barras pretas) se o recorte saiu dos limites
            pad_left = max(0, -ideal_x1)
            pad_top = max(0, -ideal_y1)
            pad_right = max(0, ideal_x2 - orig_width)
            pad_bottom = max(0, ideal_y2 - orig_height)

            if any([pad_left, pad_top, pad_right, pad_bottom]):
                cropped_face = cv2.copyMakeBorder(
                    cropped_region,
                    pad_top,
                    pad_bottom,
                    pad_left,
                    pad_right,
                    cv2.BORDER_CONSTANT,
                    value=[0, 0, 0],
                )
            else:
                cropped_face = cropped_region

            # 12. Redimensionamento inteligente para o tamanho final
            interpolation = self._get_interpolation_method(
                cropped_face.shape[1], crop_size[0]
            )
            final_image = cv2.resize(
                cropped_face, crop_size, interpolation=interpolation
            )

            # 13. Salvar
            output_path = (
                output_dir / f"{image_path.stem}_face_{i+1}{image_path.suffix}"
            )
            if self._save_image_unicode(final_image, output_path):
                print(f"   - Rosto {i+1} salvo em: {output_path}")
                print(f"     - Confiança: {detection.score[0]:.2f}")
                save_count += 1
            else:
                print(
                    f"   - Erro ao salvar rosto {i+1} para {output_path}",
                    file=sys.stderr,
                )

        return save_count > 0

    def process_multiple_images(
        self,
        input_dir: PathLike,
        output_dir: PathLike = "cropped_faces",
        crop_size: CropSize = (400, 400),
        expansion_factor: float = 1.8,
        y_offset_factor: float = 0.10,
    ):
        """Processa todas as imagens suportadas em um diretório (recursivamente)."""
        input_path = Path(input_dir)
        output_path = Path(output_dir)

        if not input_path.is_dir():
            print(
                f"Erro: Diretório de entrada não encontrado {input_path}",
                file=sys.stderr,
            )
            return

        supported_formats = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

        # Busca recursiva (rglob) por todos os formatos, case-insensitive
        image_files = [
            f
            for f in input_path.rglob("*")
            if f.suffix.lower() in supported_formats
        ]

        if not image_files:
            print(
                f"Nenhuma imagem encontrada em {input_path} (formatos: {supported_formats})"
            )
            return

        print(f"Encontradas {len(image_files)} imagens. Processando...")

        successful = 0
        for image_file in image_files:
            # Imprime o caminho relativo para logs mais limpos
            print(f"\nProcessando: {image_file.relative_to(input_path)}")
            if self.detect_and_crop_face(
                image_file,
                output_path,
                crop_size,
                expansion_factor,
                y_offset_factor,
            ):
                successful += 1

        print(
            f"\nProcessamento concluído: {successful}/{len(image_files)} imagens com rostos salvos."
        )

    def visualize_detection(self, image_path: PathLike):
        """Mostra a imagem REDUZIDA com as detecções (para debug)."""

        img_path = Path(image_path)
        original_image = self._load_image_unicode(img_path)
        if original_image is None:
            return

        detection_image, _ = self.resize_image_for_processing(original_image)

        rgb_image = cv2.cvtColor(detection_image, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_image)

        display_image = detection_image.copy()
        if results.detections:
            for detection in results.detections:
                self.mp_drawing.draw_detection(display_image, detection)

        cv2.imshow(f"Detecção em {img_path.name} (Reduzida)", display_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    r"""
    Função principal para executar o processador de rostos.
    
    Instruções (exemplo):
    1. Ative seu ambiente virtual: .\.venv\Scripts\Activate.ps1
    2. Instale as dependências: pip install opencv-python mediapipe numpy
    3. Coloque suas imagens em uma pasta (ex: "fotos_turma")
    4. Ajuste as "Configurações" abaixo.
    5. Execute o script.
    """
    print("=== Detector e Recortador de Rostos (Otimizado) ===")

    # --- Configurações ---

    # Caminho das imagens originais
    INPUT_DIR = Path("FacesToExtract")

    # Onde salvar as imagens recortadas
    OUTPUT_DIR = INPUT_DIR / "..\\FacesExtracted"

    # Tamanho final do recorte (largura, altura)
    CROP_SIZE = (400, 400)

    # Fator de expansão (zoom).
    # Menor = mais zoom no rosto (ex: 1.8)
    # Maior = mais contexto/ombros (ex: 2.5)
    EXPANSION_FACTOR = 1.8

    # Ajuste vertical do centro.
    # Positivo = move o recorte para cima (ex: 0.10 = 10% da altura do rosto)
    # 0.0 = perfeitamente centrado
    Y_OFFSET_FACTOR = 0.2

    # Qualidade da detecção vs. Velocidade
    # 600 = rápido, 800 = bom, 1200 = melhor qualidade
    RESIZE_WIDTH_FOR_DETECTION = 800

    # --- Fim das Configurações ---

    detector = FaceDetectorCropper(
        resize_for_detection=RESIZE_WIDTH_FOR_DETECTION, model_selection=0, min_confidence=0.5
    )

    detector.process_multiple_images(
        INPUT_DIR,
        OUTPUT_DIR,
        crop_size=CROP_SIZE,
        expansion_factor=EXPANSION_FACTOR,
        y_offset_factor=Y_OFFSET_FACTOR,
    )

    # --- Exemplo de Debug ---
    # Para visualizar as detecções em uma imagem específica (descomente):
    # foto_teste = INPUT_DIR / "antonio.jpg"
    # if foto_teste.exists():
    #     detector.visualize_detection(foto_teste)
    # else:
    #     print(f"Arquivo de visualização não encontrado: {foto_teste}")


if __name__ == "__main__":
    main()
