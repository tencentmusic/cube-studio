"""Regenerate tests."""

import fire
import logging
import os
import shutil
import subprocess

class Generator:
  @staticmethod
  def write_gcp_kpt(kpt="kpt"):
    """Create test data based on running the kpt commands.

    This will allow us to see any diffs if we refactor the commands
    """
    repo_root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"])
    repo_root = repo_root.decode()
    repo_root = repo_root.strip()

    data_dir = os.path.join(repo_root, "tests", "testdata_gcp_kpt")
    if os.path.exists(data_dir):
      shutil.rmtree(data_dir)

    logging.info("Removing directory %s", data_dir)

    subdirs = ["v2", "Kptfile"]

    for s in subdirs:
      src = os.path.join(repo_root, "gcp", s)
      if not os.path.exists(src):
        continue
      if os.path.isdir(src):
        shutil.copytree(src, os.path.join(data_dir, "gcp", s))
      else:
        shutil.copyfile(src, os.path.join(data_dir, "gcp", s))

    # Run a bunch of kpt commands. We want to change all the setters
    # unique values so we can see how the substitutions play out
    setters = {
      "gcloud.core.project": "customerProject",
      "gcloud.project.projectNumber": "999911112222",
      "gcloud.compute.zone": "testZone",
      "gcloud.compute.region": "testRegion",
      "location": "testLocation",
      "name": "testKptName",
      "log-firewalls": "true",
      "mgmt-name": "testMgmtName",
    }

    for k, v in setters.items():
      command = [kpt, "cfg", "set", ".", k, v]
      logging.info("Executing:\n%s", " ".join(command))
      subprocess.check_call(command, cwd=os.path.join(data_dir, "gcp"))

if __name__ == "__main__":

  logging.basicConfig(
      level=logging.INFO,
      format=('%(levelname)s|%(asctime)s'
              '|%(pathname)s|%(lineno)d| %(message)s'),
      datefmt='%Y-%m-%dT%H:%M:%S',
  )
  logging.getLogger().setLevel(logging.INFO)
  fire.Fire(Generator)
