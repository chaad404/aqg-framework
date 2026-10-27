from tools import check_staged


def reader(contents):
    return lambda path: contents[path]


def test_blocks_files_that_must_stay_local():
    files = [".env", "config/prod.env", ".secrets/aqg.env", "data/app.db", "notes/book.pdf",
             "outputs/run.csv", "source_review/review.csv", "index.faiss"]
    found = check_staged.problems(files, read=lambda path: "", secrets={})
    assert len(found) == len(files)


def test_allows_ordinary_source_files():
    files = ["src/aqg/config.py", ".env.example", "src/aqg/data_model.py", "docs/protocol.md"]
    assert check_staged.problems(files, read=lambda path: "print('hello')", secrets={}) == []


def test_blocks_a_known_secret_value_without_showing_it():
    secret = "abcdefghijklmnop1234"
    found = check_staged.problems(["src/x.py"], read=reader({"src/x.py": f"KEY = '{secret}'"}),
                                  secrets={"COHERE_API_KEY": secret})
    assert found == ["src/x.py: contains the value of COHERE_API_KEY"]


def test_blocks_openrouter_shaped_keys():
    fake_key = "sk-or-v1-" + "ab" * 32
    found = check_staged.problems(["README.md"], read=reader({"README.md": fake_key}), secrets={})
    assert found == ["README.md: contains an OpenRouter key"]
