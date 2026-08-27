import pathlib
import os
import shutil
import uuid
import pytest

try:
    from snakemake import snakemake
except ImportError:  # pragma: no cover
    pytest.skip("snakemake not installed", allow_module_level=True)


# Previously used parents[2] which resolved to the transitcdp package directory, leading to
# duplicated 'transitcdp/transitcdp' in constructed paths. We need the repository root.
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]


@pytest.fixture(autouse=True)
def _chdir_project_root(monkeypatch):
    old = os.getcwd()
    monkeypatch.chdir(PROJECT_ROOT)
    yield
    os.chdir(old)

# Helper to print reproducible command with trailing slashes
def _print_cmd(dryrun: bool, wrkspdir: pathlib.Path):
    srcdir = (PROJECT_ROOT / 'transitcdp').as_posix().rstrip('/') + '/'
    wdir = wrkspdir.as_posix().rstrip('/') + '/'
    parts = [
        "python -m snakemake",
        "-s transitcdp/workflow/main.smk",
        "--cores 1",
        "--configfile transitcdp/workflow/config/main_integration_test.yaml",
        f"--config srcdir={srcdir} wrkspdir={wdir}",
    ]
    if dryrun:
        parts.append("--dry-run")
    cmd = " \\\n  ".join(parts)
    print(f"\n[Snakemake command]\n{cmd}\n")


def test_pipeline_dag_builds():
    snakefile = PROJECT_ROOT / "transitcdp" / "workflow" / "main.smk"
    configfile = PROJECT_ROOT / "transitcdp" / "workflow" / "config" / "main_integration_test.yaml"
    assert snakefile.is_file(), f"Missing snakefile: {snakefile}"
    assert configfile.is_file(), f"Missing config: {configfile}"

    # Create ephemeral workspace dir (not strictly needed for dry-run but keeps parity)
    wrkspdir = PROJECT_ROOT / ".pytest_workspaces" / f"dag_{uuid.uuid4().hex[:8]}"
    wrkspdir.mkdir(parents=True, exist_ok=True)

    _print_cmd(dryrun=True, wrkspdir=wrkspdir)

    success = snakemake(
        snakefile=str(snakefile),
        configfiles=[str(configfile)],
        cores=1,
        dryrun=True,
        printshellcmds=False,
        keepgoing=False,
        config={
            "srcdir": (PROJECT_ROOT / "transitcdp").as_posix().rstrip('/') + '/',
            "wrkspdir": wrkspdir.as_posix().rstrip('/') + '/',
        },
    )
    assert success, "Snakemake dry-run failed (DAG did not build)."
    # Clean only on success
    shutil.rmtree(wrkspdir, ignore_errors=True)


@pytest.mark.slow
def test_pipeline_executes_minimally(tmp_path):
    snakefile = PROJECT_ROOT / "transitcdp" / "workflow" / "main.smk"
    configfile = PROJECT_ROOT / "transitcdp" / "workflow" / "config" / "main_integration_test.yaml"

    wrkspdir = tmp_path / f"run_{uuid.uuid4().hex[:8]}"
    wrkspdir.mkdir(parents=True, exist_ok=True)

    _print_cmd(dryrun=False, wrkspdir=wrkspdir)

    success = snakemake(
        snakefile=str(snakefile),
        configfiles=[str(configfile)],
        cores=1,
        dryrun=False,
        printshellcmds=False,
        keepgoing=False,
        config={
            "srcdir": (PROJECT_ROOT / "transitcdp").as_posix().rstrip('/') + '/',
            "wrkspdir": wrkspdir.as_posix().rstrip('/') + '/',
        },
    )
    assert success, "Snakemake execution failed."

    # Remove workspace only if successful to aid debugging on failure
    #shutil.rmtree(wrkspdir, ignore_errors=True)
