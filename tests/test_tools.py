import os
import tempfile
from src.tools.file_tools import read_file, write_file, list_dir
from src.tools.system_tools import get_time, run_shell


class TestFileTools:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_read_file(self):
        path = os.path.join(self.tmpdir, "test.txt")
        with open(path, "w") as f:
            f.write("hello world")
        result = read_file.invoke({"path": path})
        assert "hello world" in result

    def test_read_nonexistent_file(self):
        result = read_file.invoke({"path": "/nonexistent/path.txt"})
        assert "Error" in result or "not found" in result.lower()

    def test_write_file(self):
        path = os.path.join(self.tmpdir, "output.txt")
        result = write_file.invoke({"path": path, "content": "test content"})
        assert "success" in result.lower() or "wrote" in result.lower()
        assert os.path.exists(path)

    def test_list_dir(self):
        os.makedirs(os.path.join(self.tmpdir, "subdir"))
        with open(os.path.join(self.tmpdir, "a.txt"), "w") as f:
            f.write("")
        result = list_dir.invoke({"path": self.tmpdir})
        assert "a.txt" in result


class TestSystemTools:
    def test_get_time(self):
        result = get_time.invoke({})
        assert len(result) > 0

    def test_run_shell_harmless(self):
        result = run_shell.invoke({"command": "echo hello"})
        assert "hello" in result