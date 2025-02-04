# Docugent

## Set Up

1. Create a python virtual environment (venv)
```
python3 -m venv venv
```
2. Activate the venv
```
source venv/bin/activate
```
3. Install all dependencies in venv
```
pip install -r requirements.txt
```
4. Run the main file
```
python3 -m main chat
```


## Interactions
```
# prepare all newly added PDFs
python3 -m main prepare 

# enter chat mode
python3 -m main chat

# enter chat mode with streaming response
python3 -m main chat --stream

# enter chat mode with non-streaming response
python3 -m main chat
```