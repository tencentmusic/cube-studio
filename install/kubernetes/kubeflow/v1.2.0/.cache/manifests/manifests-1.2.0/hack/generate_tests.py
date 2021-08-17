"""Regenerate tests for only the files that have changed."""

import argparse
import jinja2
import logging
import os
import shutil
import subprocess

# Search dirs should be directories to search for kustomization packages
# that we want to test. These should be kustomization's that are doing
# non-trivial transformations (e.g. combining multiple packages, applying
# patches) etc... The point of the unittests is to make it easy for reviewers
# to verify that the expected output is correct and verify the actual output
# matches the expected output.
SEARCH_DIRS = [
  "stacks",
  # TODO(https://github.com/kubeflow/manifests/issues/1052): Remove this
  # after the move to v3 is done.
  "tests/legacy_kustomizations",
  "katib/installs",
  ]

# The subdirectory to store the expected manifests in
# We use a subdirectory of test_data because we could potentially
# have more than one version of a manifest.
KUSTOMIZE_OUTPUT_DIR = "test_data/expected"

TEST_NAME = "kustomize_test.go"

def generate_test_path(repo_root, kustomize_rpath):
  """Generate the full path of the  test.go file for a particular package

  Args:
    repo_root: Root of the repository
    kustomize_rpath: The relative path (relative to repo root) of the
      kustomize package to generate the test for.
  """

  test_path = os.path.join(repo_root, "tests", kustomize_rpath,
                           TEST_NAME)
  return test_path

def run_kustomize_build(repo_root, package_dir):
  """Run kustomize build and store the output in the test directory."""

  rpath = os.path.relpath(package_dir, repo_root)

  output_dir = os.path.join(repo_root, "tests", rpath, KUSTOMIZE_OUTPUT_DIR)

  if os.path.exists(output_dir):
    # Remove any previous version of the directory so that we ensure
    # that all files in that directory are from the new run
    # of kustomize build -o
    logging.info("Removing directory %s", output_dir)
    shutil.rmtree(output_dir)

  logging.info("Creating directory %s", output_dir)
  os.makedirs(output_dir)

  subprocess.check_call([os.environ.get("KUSTOMIZE_BIN", "kustomize"), "build", "--load_restrictor", "none",
                         "-o", output_dir], cwd=os.path.join(repo_root,
                                                             package_dir))
def find_kustomize_dirs(search_dirs):
  """Find all kustomization directories in search_dirs.

  Args:
    search_dirs: A list of directories to recursively search for
     kustomization.yaml files which will be used to
       1. generate expected output
       2. generate tests
  """

  changed_dirs = set()

  for s in search_dirs:
    for child, _, files in os.walk(s):
      for f in files:
        if f == "kustomization.yaml":
          changed_dirs.add(child)

  return changed_dirs

def write_go_test(test_path, package_name, package_dir):
  """Write the go test file.

  Args:
    test_path: Path for the go file
    package_name: The name for the go package the test should live in
    package_dir: The path to the kustomize package being tested; this
      should be the relative path to the kustomize directory.
  """
  test_contents = template.render({"package": package_name,
                                   "package_dir":package_dir})


  logging.info("Writing file: %s", test_path)
  with open(test_path, "w") as test_file:
    test_file.write(test_contents)

if __name__ == "__main__":

  logging.basicConfig(
      level=logging.INFO,
      format=('%(levelname)s|%(asctime)s'
              '|%(pathname)s|%(lineno)d| %(message)s'),
      datefmt='%Y-%m-%dT%H:%M:%S',
  )
  logging.getLogger().setLevel(logging.INFO)

  parser = argparse.ArgumentParser()

  parser.add_argument(
      "--all",
      dest = "all_tests",
      action = "store_true",
      help="(Deprecated) this parameter has no effect")

  parser.set_defaults(all_tests=False)

  args = parser.parse_args()

  repo_root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"])
  repo_root = repo_root.decode()
  repo_root = repo_root.strip()

  # Get a list of package directories
  full_search_dirs = [os.path.join(repo_root, s) for s in SEARCH_DIRS]
  package_dirs = find_kustomize_dirs(full_search_dirs)

  changed_dirs = package_dirs

  this_dir = os.path.dirname(__file__)
  loader = jinja2.FileSystemLoader(searchpath=os.path.join(
    this_dir, "templates"))
  env = jinja2.Environment(loader=loader)
  template = env.get_template("kustomize_test.go.template")

  for full_dir in changed_dirs:
    # Get the relative path of the kustomize directory.
    # This is the path relative to the repo root.
    rpath = os.path.relpath(full_dir, repo_root)

    test_path = generate_test_path(repo_root, rpath)
    logging.info("Regenerating test %s for %s ", test_path, full_dir)

    # Generate the kustomize output
    run_kustomize_build(repo_root, full_dir)

    # Create the go test file.
    # TODO(jlewi): We really shouldn't need to redo this if it already
    # exists.

    # The go package name will be the final directory in the path
    package_name = os.path.basename(full_dir)
    # Go package names replace hyphens with underscores
    package_name = package_name.replace("-", "_")

    # We need to construct the path relative to the _test.go file of
    # the kustomize package. This path with consist of ".." entries repeated
    # enough times to get to the root of the repo. We then add the relative
    # path to the kustomize package.
    pieces = rpath.split(os.path.sep)

    p = [".."] * len(pieces)
    p.append("..")
    p.append(rpath)
    package_dir = os.path.join(*p)

    write_go_test(test_path, package_name, package_dir)
