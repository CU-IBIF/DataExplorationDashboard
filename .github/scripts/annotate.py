"""Turn test failures / log tails into GitHub error annotations (visible on the run summary page
and through the API, without downloading logs).

    python .github/scripts/annotate.py junit junit.xml
    python .github/scripts/annotate.py log selftest.log [title]
"""
import sys
import xml.etree.ElementTree as ET


def emit(title, text, limit=6000):
    text = text[-limit:].replace("%", "%25").replace("\r", "").replace("\n", "%0A")
    print(f"::error title={title}::{text}")


def main():
    kind, path = sys.argv[1], sys.argv[2]
    if kind == "junit":
        n = 0
        for case in ET.parse(path).iter("testcase"):
            for bad in list(case.findall("failure")) + list(case.findall("error")):
                emit(f"{case.get('classname')}.{case.get('name')}", (bad.get("message") or "") + "\n" + (bad.text or ""))
                n += 1
                if n >= 9:
                    return
    else:
        with open(path, encoding="utf-8", errors="replace") as fh:
            lines = fh.read().splitlines()
        emit(sys.argv[3] if len(sys.argv) > 3 else "log tail", "\n".join(lines[-60:]))


if __name__ == "__main__":
    main()
