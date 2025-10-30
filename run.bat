@echo off
	
REM Criar... precisa ser com python 3.12, acima disso o mediapipe parece ainda não funcionar
REM py -3.12 -m venv .venv

REM Instalar bibliotecas
REM pip install -r requirements.txt

REM Se adicionada nova biblioteca, atualizar requirements
REM pip freeze > requirements.txt

REM Carregar
call .venv\Scripts\activate

REM Rodar script
python FaceExtractor.py

REM Desativar ambiente
deactivate

REM (ou rodar tudo no ambiente sem precisar ativar, rodar, desativar .venv\Scripts\python.exe FaceExtractor.py)