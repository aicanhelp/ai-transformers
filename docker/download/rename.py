from ai_harness.fileutils import DirNavigator, join_path
import os
from box import Box
import sys


class ModelFileRname():
    def __init__(self, dir):
        self._dir = dir

    def _handle_file(self, json_file, dir):
        print('Handling ' + join_path(dir, json_file))
        file_base_name = os.path.basename(json_file).replace('.json', '')
        lock_file = join_path(self._dir, dir, file_base_name + '.lock')
        if os.path.exists(lock_file): os.remove(lock_file)

        json_file = join_path(self._dir, dir, json_file)
        obj = Box.from_json(filename=json_file)
        target_name = join_path(self._dir, dir, os.path.basename(obj.url))
        os.rename(join_path(self._dir, dir, file_base_name), target_name)

        os.remove(json_file)

    def run(self):
        dirNavigator = DirNavigator(file_pattern="*.json")
        dirNavigator.on_item(onFile=self._handle_file)
        dirNavigator.nav(self._dir)


if __name__ == '__main__':
    print('Scan dir: ' + sys.argv[1])
    ModelFileRname(sys.argv[1]).run()
