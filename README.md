# corpus-explorer

## Installation for development

### Ubuntu


#### 1. Install Poetry

In the terminal, run the following command in Python
(from [https://poetry.eustace.io/docs/](https://poetry.eustace.io/docs/)):

`curl -sSL https://raw.githubusercontent.com/sdispater/poetry/master/get-poetry.py | python`

Try `poetry --version`. If there is an error, add poetry to path.
That is, add `export PATH=~/.poetry/bin:$PATH` to your .bashrc file.


#### 2. Install pyenv

a) Follow steps 1-5 in Basic GitHub Checkout steps from
[https://github.com/pyenv/pyenv](https://github.com/pyenv/pyenv)

b) Before step 6 (running `pyenv install 3.7.4`), run the following command
(from [https://github.com/pyenv/pyenv/wiki](https://github.com/pyenv/pyenv/wiki))
to ensure zlib, zlibbz2, and other dependencies are installed:

`sudo apt-get update; sudo apt-get install --no-install-recommends make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev`

c) Remove libssl-dev and install libssl1.0-dev, or else ou will get a "missing openssl" error
(from [https://github.com/pyenv/pyenv/issues/950](https://github.com/pyenv/pyenv/issues/950)).

In the terminal, run `sudo apt-get remove libssl-dev` and `sudo apt-get install libssl1.0-dev`.

d) Proceed with step 6 of the Basic GitHub Checkout steps. That is, run `pyenv install 3.7.4`.

**Poetry and pyenv should now be installed!**


#### 3. Set project-specific Python version

Use pyenv to specify which version of Python should be used when invoking
`python` from the command line while in the repo folder.

`pyenv local 3.7.4`


#### 4. Install the project dependencies

`poetry install`

**You should be good to go.** You can now drop into the virtualenv by running
`poetry shell`, or run an arbitrary command in the virtualenv without dropping
into it by using `poetry run <your_command>`, for example `poetry run python
some_script.py`.


#### Want to add dependencies?

Simply run `poetry add <package_name>`. Don't forget to commit the resulting
changes to `pyproject.toml` and `poetry.lock`!


### MacOS

Get wrecked.

*TODO: add MacOS instructions*


## Testing the code

`make test`


## Architecture

![image](https://user-images.githubusercontent.com/5240492/68536204-082ada00-0304-11ea-979c-052883fa2484.png)
