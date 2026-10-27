import pytest

from aqg import config


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """An empty repository folder with no secrets in the environment."""
    for name in config.SECRET_NAMES + ("AQG_ENV_FILE",):
        monkeypatch.delenv(name, raising=False)
    root = tmp_path / "repo"
    root.mkdir()
    return root


def write_outside(repo, text):
    folder = repo.parent / ".secrets"
    folder.mkdir()
    (folder / "aqg.env").write_text(text)


def test_environment_wins_over_files(repo, monkeypatch):
    (repo / ".env").write_text("COHERE_API_KEY=from-file\n")
    monkeypatch.setenv("COHERE_API_KEY", "from-env")
    assert config.get_secret("COHERE_API_KEY", repo) == "from-env"


def test_explicit_file_comes_before_the_repo_env(repo, tmp_path, monkeypatch):
    explicit = tmp_path / "custom.env"
    explicit.write_text("MISTRAL_API_KEY=explicit\n")
    (repo / ".env").write_text("MISTRAL_API_KEY=repo\n")
    monkeypatch.setenv("AQG_ENV_FILE", str(explicit))
    assert config.get_secret("MISTRAL_API_KEY", repo) == "explicit"


def test_repo_env_comes_before_the_outside_folder(repo):
    (repo / ".env").write_text("COHERE_API_KEY=repo\n")
    write_outside(repo, "COHERE_API_KEY=outside\n")
    assert config.get_secret("COHERE_API_KEY", repo) == "repo"


def test_secrets_folder_outside_the_repo_is_used(repo):
    write_outside(repo, "OPENROUTER_API_KEY=outside\n")
    assert config.get_secret("OPENROUTER_API_KEY", repo) == "outside"


def test_missing_secret_names_where_to_put_it(repo):
    with pytest.raises(RuntimeError, match=r"\.secrets"):
        config.require_secret("OPENALEX_API_KEY", repo)


def test_describe_reports_presence_only(repo):
    (repo / ".env").write_text("COHERE_API_KEY=abc123\n")
    report = config.describe(repo)
    assert report["COHERE_API_KEY"] is True
    assert report["OPENROUTER_API_KEY"] is False
    assert "abc123" not in repr(report)
