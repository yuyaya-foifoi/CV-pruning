SRC := src

style:
	poetry run isort $(SRC)
	poetry run black $(SRC) --line-length 79

activate: 
	source ~/.bashrc

tmux_start:
	tmux new-session -s train

tmux_list:
	tmux list-sessions

train:
	poetry run python src/tools/train.py