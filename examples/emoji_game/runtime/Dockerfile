FROM python:3.12-slim

WORKDIR /opt/emoji_game
RUN python -m pip install --no-cache-dir --upgrade trajectory-sdk emoji
RUN python -c "import emoji, trajectory"
COPY emoji_game_harness.py /opt/emoji_game/emoji_game_harness.py
COPY tasks /opt/emoji_game/tasks

CMD ["sleep", "infinity"]
