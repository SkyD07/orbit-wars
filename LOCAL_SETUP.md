# Local Setup

This project uses a dedicated Conda environment named `orbit-wars`.

## Activate

```powershell
& "C:\tools\Anaconda3\Scripts\conda.exe" activate orbit-wars
```

If activation is not configured for the current shell, use `conda run`:

```powershell
& "C:\tools\Anaconda3\Scripts\conda.exe" run -n orbit-wars python --version
```

## Kaggle CLI

The Kaggle token used for this project is in:

```text
C:\Users\LENOVO\Project\Tes Skill\kaggle.json
```

Use it by setting `KAGGLE_CONFIG_DIR` to that folder:

```powershell
$env:KAGGLE_CONFIG_DIR = "C:\Users\LENOVO\Project\Tes Skill"
```

## Download Competition Files

```powershell
$env:KAGGLE_CONFIG_DIR = "C:\Users\LENOVO\Project\Tes Skill"
& "C:\tools\Anaconda3\Scripts\conda.exe" run -n orbit-wars kaggle competitions download -c orbit-wars -p . --force
Expand-Archive -LiteralPath ".\orbit-wars.zip" -DestinationPath "." -Force
```

## Smoke Test

```powershell
& "C:\tools\Anaconda3\Scripts\conda.exe" run -n orbit-wars python -c "from kaggle_environments import make; env=make('orbit_wars', configuration={'seed':42}, debug=True); env.run(['main.py','random']); print([(i, s.reward, s.status) for i, s in enumerate(env.steps[-1])])"
```

Expected result for the starter agent at seed 42:

```text
[(0, 1, 'DONE'), (1, -1, 'DONE')]
```

## Play In Browser

Run the local browser player:

```powershell
& "C:\tools\Anaconda3\Scripts\conda.exe" run -n orbit-wars python play_server.py
```

Then open:

```text
http://127.0.0.1:5000
```

Spectator controls:

- The 4-player lineup is `Public 1224`, `Public 1000`, `Public 1060`, and local `Medium`.
- Click `Step` to advance exactly one turn.
- The bots only act when you click `Step`; there is no automatic progression.
