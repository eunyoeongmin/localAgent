# Python 3.12 이미지 사용
FROM python:3.12

# 작업 디렉토리 설정
WORKDIR /code

# 의존성 파일 복사 및 설치
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 사용자 설정 (허깅페이스 스페이스는 root 권한이 아닌 user 권한을 권장)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

# 나머지 파일 복사
WORKDIR $HOME/app
COPY --chown=user . $HOME/app

# 실행 모드 환경 변수 설정 (GUI 모드 강제)
ENV RUN_MODE=GUI

# 포트 설정 (허깅페이스 기본 포트)
EXPOSE 7860

# 앱 실행
CMD ["python", "app.py"]
