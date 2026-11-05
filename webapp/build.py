"""Build the static demo from this checkout's unmodified analysis modules."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import zipfile

HERE = Path(__file__).resolve().parent
MODULES = ('__init__.py', 'api.py', 'CompactionFunctions.py', 'StructureTensor.py',
           'utilities.py', 'plotting.py', 'distance.py')


def build(repo, output):
    output.mkdir(parents=True, exist_ok=True)
    for source in (HERE / 'src').iterdir():
        if source.name == '__pycache__':
            continue
        if source.is_file():
            shutil.copy2(source, output / source.name)
        elif source.is_dir():
            shutil.copytree(source, output / source.name, dirs_exist_ok=True)
    shutil.copytree(HERE / 'examples', output / 'examples', dirs_exist_ok=True)
    hashes = {}
    with zipfile.ZipFile(output / 'analysis.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in MODULES:
            data = (repo / 'CompactionAnalyzer' / name).read_bytes()
            archive.writestr('CompactionAnalyzer/' + name, data)
            hashes[name] = hashlib.sha256(data).hexdigest()
        archive.write(repo / 'LICENSE', 'LICENSE')
    shutil.copy2(repo / 'LICENSE', output / 'LICENSE.txt')
    (output / '.nojekyll').touch()
    (output / 'build-info.json').write_text(json.dumps({'modules': hashes,
        'pyodide': '0.27.7', 'calculation': 'CompactionAnalyzer.analyze_image'}, indent=2))
    print(f'Built static demo: {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, default=HERE.parent)
    parser.add_argument('--output', type=Path, default=HERE / 'dist')
    args = parser.parse_args()
    build(args.repo.resolve(), args.output.resolve())
