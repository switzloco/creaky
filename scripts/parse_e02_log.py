import json
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

log_file = "checkpoints/e02_trained/training-book.log"
with open(log_file, "r", encoding="utf-8") as f:

    text = f.read()

print(f"Total log length: {len(text)} characters")

try:
    events = json.loads(text)
    print(f"Total events: {len(events)}")
    # Print the last 50 events
    for e in events[-50:]:
        d = e.get("data", "")
        if d.strip():
            print(f"[{e.get('stream_name')}] {d.strip()}")
except Exception as ex:
    print("Error parsing json log:", ex)
    # Print last 2000 chars
    print(text[-2000:])
