import argparse
from pathlib import Path

from openai import OpenAI

parser = argparse.ArgumentParser()
parser.add_argument("prompt", help="Question or task for Astra")
parser.add_argument("--file", help="Optional file to include")
args = parser.parse_args()

message = args.prompt
if args.file:
    path = Path(args.file)
    message += f"\n\nFile: {path}\n\n{path.read_text(encoding='utf-8')}"

response = OpenAI().responses.create(
    model="gpt-6-astra",
    input=message,
)
print(response.output_text)