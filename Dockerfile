# Python 3.12 イメージを使用
FROM python:3.12

# 作業ディレクトリの設定
WORKDIR /code

# 依存関係ファイルのコピーとインストール
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# ユーザー設定 (Hugging Face Spacesはroot権限ではなくuser権限を推奨)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

# 残りのファイルをコピー
WORKDIR $HOME/app
COPY --chown=user . $HOME/app

# 実行モード環境変数の設定 (GUIモード強制)
ENV RUN_MODE=GUI

# ポート設定 (Hugging Face 基本ポート)
EXPOSE 7860

# アプリ実行
CMD ["python", "app.py"]
