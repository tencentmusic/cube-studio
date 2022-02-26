
import os
import glob
import shutil
import traceback
from .context import KFJobContext


class DataPathCleaner(object):
    def __init__(self, base_path=None):
        self.base_path = (base_path or '').strip() or (KFJobContext.get_context().get_user_path() or '').strip()
        if self.base_path and not os.path.isabs(self.base_path):
            raise RuntimeError("base_path must be a absolute path, got '{}'".format(self.base_path))
        elif self.base_path:
            self.base_path = os.path.normpath(self.base_path)
        print("{}: set base_path='{}'".format(self, self.base_path))

    def clean(self, file_pattern):
        if not file_pattern:
            print("file_pattern not specified, will not do clean")
            return
        if not isinstance(file_pattern, list):
            file_pattern = [file_pattern]

        for fp in file_pattern:
            if os.path.isabs(fp):
                file_to_clean = os.path.abspath(fp)
            else:
                file_to_clean = os.path.abspath(os.path.join(KFJobContext.get_context().export_path, fp))

            if self.base_path and not file_to_clean.startswith(self.base_path):
                print("file_pattern '{}' go out of base path '{}', ignore it".format(fp, self.base_path))
                continue
            for ftc in glob.glob(file_to_clean):
                try:
                    if os.path.isdir(ftc):
                        shutil.rmtree(ftc, ignore_errors=True)
                        print("removed dir '{}' derived from '{}', pattern='{}'".format(ftc, file_to_clean, fp))
                    else:
                        os.remove(ftc)
                        print("removed file '{}' derived from '{}', pattern='{}'".format(ftc, file_to_clean, fp))
                except Exception as e:
                    print("clean '{}' dervied from '{}', pattern='{}' error: {}\n{}"
                          .format(ftc, file_to_clean, fp, e, traceback.format_exc()))