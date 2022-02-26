
import os
from .context import KFJobContext
import traceback
import random
import glob


class Archiver(object):
    def __init__(self, base_path=None):
        ctx = KFJobContext.get_context()
        self.archive_base_path = (base_path or '').strip() or (ctx.archive_base_path or '').strip()
        if self.archive_base_path and not os.path.isabs(self.archive_base_path):
            raise RuntimeError("archive_base_path must be a absolute path, got '{}'"
                               .format(self.archive_base_path))
        elif self.archive_base_path:
            pipeline = "-".join([ctx.pipeline_id, ctx.pipeline_name])
            self.archive_base_path = os.path.normpath(os.path.join(self.archive_base_path, ctx.creator or '', pipeline))
            print("{}: set archive_base_path='{}'".format(self, self.archive_base_path))

    @staticmethod
    def __random_file_name(length=10):
        return ''.join(random.choices('1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ', k=length))

    @staticmethod
    def __extract_source_files(source_files):
        if not source_files:
            return []
        if not isinstance(source_files, list):
            source_files = [source_files]

        extracted = []
        for source in source_files:
            try:
                for f in glob.glob(source):
                    extracted.append(os.path.normpath(f))
            except Exception as e:
                print("walk through source path '{}' error: {}\n{}".format(source, e, traceback.format_exc()))
                continue
        return extracted

    def archive(self, source_files, path_name, compressfile=None):
        if not self.archive_base_path:
            print("archive_base_path not set, will not archive '{}' to '{}'".format(source_files, path_name))
            return None

        extract_source_files = self.__extract_source_files(source_files)
        if not extract_source_files:
            print("found no source file from '{}' to archive".format(source_files))
            return []

        print("extracted source files to be archived: {}".format(extract_source_files))

        ctx = KFJobContext.get_context()
        dest_path = os.path.join(self.archive_base_path, (ctx.run_id or '').strip(), (path_name or '').strip())
        dest_path = os.path.abspath(dest_path)
        if not dest_path.startswith(self.archive_base_path):
            raise RuntimeError("path_name '{}' go out of base path '{}'".format(path_name, self.archive_base_path))
        if not os.path.isdir(dest_path):
            try:
                os.makedirs(dest_path, exist_ok=True)
                print("create archive dir '{}' for source files '{}'".format(dest_path, source_files))
            except Exception as e:
                print("create archive dir '{}' for source files '{}' error: {}\n{}"
                      .format(dest_path, source_files, e, traceback.format_exc()))
                return None

        if isinstance(compressfile, str):
            compressfile = compressfile.strip()
        elif compressfile:
            compressfile = self.__random_file_name()

        if compressfile:
            if not compressfile.endswith(".tar.gz"):
                compressfile += ".tar.gz"
            tarfile_name = os.path.join(dest_path, compressfile)
            cur_dir_temp = os.getcwd()
            try:
                import tarfile
                f = tarfile.open(tarfile_name, "w:gz")
                for s in extract_source_files:
                    s_path, s_name = os.path.split(s)
                    os.chdir(s_path)
                    f.add(s_name)
                f.close()
                print("archived '{}' into tar file '{}'".format(extract_source_files, tarfile_name))
                return tarfile_name
            except Exception as e:
                print("archive '{}' into tar file '{}' error: {}\n{}".format(extract_source_files, tarfile_name,
                                                                             e, traceback.format_exc()))
                if os.path.isfile(tarfile_name):
                    os.remove(tarfile_name)
                return None
            finally:
                os.chdir(cur_dir_temp)
        else:
            import shutil
            archived = []
            for s in extract_source_files:
                try:
                    if os.path.isdir(s):
                        dir_basename = os.path.basename(s)
                        dest_s = os.path.join(dest_path, dir_basename)
                        if os.path.isdir(dest_s):
                            print("destination dir '{}' of source '{}' exists, will remove it first".format(s, dest_s))
                            shutil.rmtree(dest_s, ignore_errors=True)
                        shutil.copytree(s, os.path.join(dest_path, dir_basename))
                    else:
                        dest_s = shutil.copy2(s, dest_path)
                    archived.append((s, dest_s))
                    print("arhived '{}' into dir '{}'".format(s, dest_path))
                except Exception as e:
                    print("archive '{}' into dir '{}' error: {}\n{}".format(s, dest_path, e, traceback.format_exc()))
                    for _, copied in archived:
                        if os.path.isfile(copied):
                            os.remove(copied)
                        elif os.path.isdir(copied):
                            shutil.rmtree(copied, ignore_errors=True)
                    return None
            return archived

    def find_tf_model_paths(self):
        def __is_tf_model_path(path):
            if not os.path.isdir(path):
                return False
            variables_path = os.path.join(path, 'variables')
            pb_path = os.path.join(path, 'saved_model.pb')
            return os.path.isdir(variables_path) and os.path.isfile(pb_path)

        search_queue = [self.archive_base_path]
        while len(search_queue) > 0:
            sp = search_queue.pop(0)
            if __is_tf_model_path(sp):
                yield sp

            for sub in glob.glob(os.path.join(sp, '*')):
                if not os.path.isdir(sub):
                    continue
                search_queue.append(sub)
