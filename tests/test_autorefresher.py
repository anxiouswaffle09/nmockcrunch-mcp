"""Tests for AutoRefresher and call_tool path guard behavior."""

import json
import os
import sys
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from jcodemunch_mcp.server import AutoRefresher, MAX_WATCHED_PATHS, call_tool


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def ar(tmp_path):
    """Fresh AutoRefresher instance with CONFIG_PATH redirected to tmp_path."""
    cfg = str(tmp_path / "autorefresh.json")
    with patch.object(AutoRefresher, "CONFIG_PATH", cfg):
        yield AutoRefresher()


# ---------------------------------------------------------------------------
# is_path_watched
# ---------------------------------------------------------------------------

class TestIsPathWatched:
    def test_empty_watchlist_allows_all(self, ar):
        assert ar.is_path_watched("/any/random/path") is True

    def test_exact_match(self, ar, tmp_path):
        watched = str(tmp_path / "project")
        ar._paths = {watched}
        assert ar.is_path_watched(watched) is True

    def test_subdirectory_match(self, ar, tmp_path):
        watched = str(tmp_path / "project")
        ar._paths = {watched}
        assert ar.is_path_watched(str(tmp_path / "project" / "src" / "main.py")) is True

    def test_unrelated_path_blocked(self, ar, tmp_path):
        ar._paths = {str(tmp_path / "project")}
        assert ar.is_path_watched("/completely/different/path") is False

    def test_prefix_collision(self, ar, tmp_path):
        """/watched/project must not match /watched/projectfoo."""
        watched = str(tmp_path / "project")
        sibling = str(tmp_path / "projectfoo")
        ar._paths = {watched}
        assert ar.is_path_watched(sibling) is False

    @pytest.mark.skipif(sys.platform == "win32", reason="Symlinks unreliable on Windows")
    def test_resolves_symlinks(self, ar, tmp_path):
        real = tmp_path / "real"
        real.mkdir()
        link = tmp_path / "link"
        link.symlink_to(real)
        ar._paths = {str(real)}
        assert ar.is_path_watched(str(link)) is True

    def test_non_watched_path_blocked(self, ar, tmp_path):
        ar._paths = {str(tmp_path / "project")}
        assert ar.is_path_watched(str(tmp_path / "other")) is False


# ---------------------------------------------------------------------------
# register_path
# ---------------------------------------------------------------------------

class TestRegisterPath:
    def test_returns_true_when_newly_added(self, ar, tmp_path):
        target = tmp_path / "project"
        target.mkdir()
        assert ar.register_path(str(target)) is True
        assert os.path.realpath(str(target)) in ar._paths

    def test_returns_true_when_already_present(self, ar, tmp_path):
        target = tmp_path / "project"
        target.mkdir()
        ar.register_path(str(target))
        assert ar.register_path(str(target)) is True

    def test_returns_false_when_cap_hit(self, ar, tmp_path):
        target = tmp_path / "project"
        target.mkdir()
        for i in range(MAX_WATCHED_PATHS):
            ar._paths.add(f"/fake/path/{i}")
        assert ar.register_path(str(target)) is False
        assert os.path.realpath(str(target)) not in ar._paths

    def test_persists_to_disk(self, ar, tmp_path):
        target = tmp_path / "project"
        target.mkdir()
        ar.register_path(str(target))
        cfg = json.loads(Path(AutoRefresher.CONFIG_PATH).read_text())
        assert str(target.resolve()) in cfg["paths"]

    def test_already_present_does_not_write_disk(self, ar, tmp_path):
        target = tmp_path / "project"
        target.mkdir()
        ar.register_path(str(target))
        mtime_after_first = Path(AutoRefresher.CONFIG_PATH).stat().st_mtime
        ar.register_path(str(target))
        mtime_after_second = Path(AutoRefresher.CONFIG_PATH).stat().st_mtime
        assert mtime_after_first == mtime_after_second


# ---------------------------------------------------------------------------
# remove_path
# ---------------------------------------------------------------------------

class TestRemovePath:
    def test_returns_true_and_removes_from_memory(self, ar, tmp_path):
        target = tmp_path / "project"
        target.mkdir()
        ar.register_path(str(target))
        resolved = os.path.realpath(str(target))
        assert ar.remove_path(str(target)) is True
        assert resolved not in ar._paths

    def test_returns_false_when_not_present(self, ar, tmp_path):
        assert ar.remove_path(str(tmp_path / "nonexistent")) is False

    def test_persists_removal_to_disk(self, ar, tmp_path):
        target = tmp_path / "project"
        target.mkdir()
        ar.register_path(str(target))
        ar.remove_path(str(target))
        cfg = json.loads(Path(AutoRefresher.CONFIG_PATH).read_text())
        assert str(target.resolve()) not in cfg["paths"]

    def test_disk_unchanged_when_not_present(self, ar, tmp_path):
        other = tmp_path / "other"
        other.mkdir()
        ar.register_path(str(other))
        before = json.loads(Path(AutoRefresher.CONFIG_PATH).read_text())
        ar.remove_path(str(tmp_path / "nonexistent"))
        after = json.loads(Path(AutoRefresher.CONFIG_PATH).read_text())
        assert before == after


# ---------------------------------------------------------------------------
# get_watched_paths
# ---------------------------------------------------------------------------

class TestGetWatchedPaths:
    def test_returns_empty_when_no_paths(self, ar):
        assert ar.get_watched_paths() == []

    def test_returns_sorted_list(self, ar):
        ar._paths = {"/z/path", "/a/path", "/m/path"}
        paths = ar.get_watched_paths()
        assert paths == sorted(paths)

    def test_picks_up_disk_changes(self, ar, tmp_path):
        cfg_path = Path(AutoRefresher.CONFIG_PATH)
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(json.dumps({"paths": ["/new/external/path"]}))
        ar._cfg_mtime = None  # force reload
        paths = ar.get_watched_paths()
        assert "/new/external/path" in paths


# ---------------------------------------------------------------------------
# call_tool path guard behavior
# ---------------------------------------------------------------------------

class TestPathGuard:
    """Tests for path guard in call_tool. Mocks auto_refresher to avoid
    real filesystem and index operations."""

    @pytest.mark.asyncio
    async def test_refresh_tool_blocked_returns_guard_error(self):
        mock_ar = MagicMock()
        mock_ar.is_path_watched.return_value = False

        with patch("jcodemunch_mcp.server.auto_refresher", mock_ar):
            result = await call_tool("get_file_outline", {"repo": "foo", "file_path": "bar.py"})

        data = json.loads(result[0].text)
        assert data["error"] == "PATH_GUARD_BLOCKED"

    @pytest.mark.asyncio
    async def test_refresh_tool_blocked_skips_maybe_refresh(self):
        """After the ordering fix, refresh must not run when blocked."""
        mock_ar = MagicMock()
        mock_ar.is_path_watched.return_value = False

        with patch("jcodemunch_mcp.server.auto_refresher", mock_ar):
            await call_tool("search_symbols", {"repo": "foo", "query": "bar"})

        mock_ar.maybe_refresh.assert_not_called()

    @pytest.mark.asyncio
    async def test_refresh_tool_allowed_runs_maybe_refresh(self):
        mock_ar = MagicMock()
        mock_ar.is_path_watched.return_value = True
        mock_ar.maybe_refresh.return_value = None

        with patch("jcodemunch_mcp.server.auto_refresher", mock_ar), \
             patch("jcodemunch_mcp.server.get_file_outline", return_value={"symbols": []}):
            result = await call_tool("get_file_outline", {"repo": "foo", "file_path": "bar.py"})

        mock_ar.maybe_refresh.assert_called_once()
        data = json.loads(result[0].text)
        assert data.get("error") != "PATH_GUARD_BLOCKED"

    @pytest.mark.asyncio
    async def test_empty_watchlist_allows_refresh_tools(self):
        """Empty watchlist = default allow."""
        mock_ar = MagicMock()
        mock_ar.is_path_watched.return_value = True  # empty → True
        mock_ar.maybe_refresh.return_value = None

        with patch("jcodemunch_mcp.server.auto_refresher", mock_ar), \
             patch("jcodemunch_mcp.server.get_file_outline", return_value={"symbols": []}):
            result = await call_tool("get_file_outline", {"repo": "foo", "file_path": "bar.py"})

        data = json.loads(result[0].text)
        assert data.get("error") != "PATH_GUARD_BLOCKED"

    @pytest.mark.asyncio
    async def test_index_folder_blocked_when_path_not_watched(self):
        mock_ar = MagicMock()
        mock_ar.is_path_watched.return_value = False

        with patch("jcodemunch_mcp.server.auto_refresher", mock_ar):
            result = await call_tool("index_folder", {"path": "/some/project"})

        data = json.loads(result[0].text)
        assert data["error"] == "PATH_GUARD_BLOCKED"

    @pytest.mark.asyncio
    async def test_index_folder_allowed_when_path_watched(self, tmp_path):
        mock_ar = MagicMock()
        mock_ar.is_path_watched.return_value = True
        mock_ar.register_path.return_value = True

        with patch("jcodemunch_mcp.server.auto_refresher", mock_ar), \
             patch("jcodemunch_mcp.server.index_folder", return_value={"success": True, "indexed": 0}):
            result = await call_tool("index_folder", {"path": str(tmp_path)})

        data = json.loads(result[0].text)
        assert data.get("error") != "PATH_GUARD_BLOCKED"

    @pytest.mark.asyncio
    async def test_add_to_watchlist_success(self, tmp_path):
        mock_ar = MagicMock()
        mock_ar.register_path.return_value = True
        mock_ar.get_watched_paths.return_value = [str(tmp_path)]

        with patch("jcodemunch_mcp.server.auto_refresher", mock_ar):
            result = await call_tool("add_to_watchlist", {"path": str(tmp_path)})

        data = json.loads(result[0].text)
        assert data["success"] is True
        assert "message" not in data

    @pytest.mark.asyncio
    async def test_add_to_watchlist_cap_hit_reports_failure(self, tmp_path):
        mock_ar = MagicMock()
        mock_ar.register_path.return_value = False
        mock_ar.get_watched_paths.return_value = []

        with patch("jcodemunch_mcp.server.auto_refresher", mock_ar):
            result = await call_tool("add_to_watchlist", {"path": str(tmp_path)})

        data = json.loads(result[0].text)
        assert data["success"] is False
        assert "message" in data

    @pytest.mark.asyncio
    async def test_remove_from_watchlist_present(self, tmp_path):
        mock_ar = MagicMock()
        mock_ar.remove_path.return_value = True
        mock_ar.get_watched_paths.return_value = []

        with patch("jcodemunch_mcp.server.auto_refresher", mock_ar):
            result = await call_tool("remove_from_watchlist", {"path": str(tmp_path)})

        data = json.loads(result[0].text)
        assert data["success"] is True

    @pytest.mark.asyncio
    async def test_remove_from_watchlist_not_present(self, tmp_path):
        mock_ar = MagicMock()
        mock_ar.remove_path.return_value = False
        mock_ar.get_watched_paths.return_value = []

        with patch("jcodemunch_mcp.server.auto_refresher", mock_ar):
            result = await call_tool("remove_from_watchlist", {"path": str(tmp_path)})

        data = json.loads(result[0].text)
        assert data["success"] is False

    @pytest.mark.asyncio
    async def test_list_watched_paths(self, tmp_path):
        mock_ar = MagicMock()
        mock_ar.get_watched_paths.return_value = [str(tmp_path)]

        with patch("jcodemunch_mcp.server.auto_refresher", mock_ar):
            result = await call_tool("list_watched_paths", {})

        data = json.loads(result[0].text)
        assert "watched_paths" in data
        assert str(tmp_path) in data["watched_paths"]
