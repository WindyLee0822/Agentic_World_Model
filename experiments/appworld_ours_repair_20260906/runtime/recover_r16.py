from status import client
from pathlib import Path
root=Path(__file__).resolve().parent.parent/'snapshots'/'r16'
client._code.download('3b165da4-75f7-4c5d-bbcc-3f7eefd5fe3d','1',str(root))
print('Recovered R16 source to',root,flush=True)
