"""Publish complete append-only benchmark records without losing prior evidence."""
import json
import os
import tempfile


def publish(path, content, minimum_count=0):
    rows = [json.loads(line) for line in content.splitlines() if line.strip()]
    if not rows or len(rows) < minimum_count:
        raise ValueError("Remote records are empty or shorter than the audited snapshot")
    old = path.read_bytes() if path.exists() else b""
    old_rows = [json.loads(line) for line in old.splitlines() if line.strip()]
    if rows[:len(old_rows)] != old_rows:
        raise ValueError("Remote records do not preserve the previously downloaded prefix")
    if content == old:
        return len(rows), False
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return len(rows), True
