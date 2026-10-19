"""Recording publication preserves fresh destinations and cleans up failures."""

from types import SimpleNamespace

import pytest

import scene_recall.visualization.recording as recording_io


@pytest.fixture
def recording(monkeypatch):
    class Recording:
        def __init__(self):
            self.events = []
            self.fail = None

        def save(self, path):
            self.events.append("save")
            path.write_bytes(b"complete recording")

        def send_recording_name(self, name):
            self.events.append("name")

        def flush(self):
            self.events.append("flush")
            if self.fail == "flush":
                raise RuntimeError("flush failed")

        def disconnect(self):
            self.events.append("disconnect")
            if self.fail == "disconnect":
                raise RuntimeError("disconnect failed")

    instance = Recording()
    monkeypatch.setattr(
        recording_io,
        "require_rerun",
        lambda: SimpleNamespace(RecordingStream=lambda name: instance),
    )
    return instance


def test_recording_is_published_only_after_disconnect(tmp_path, recording):
    output = tmp_path / "result.rrd"
    with recording_io.recording_stream(output) as stream:
        assert stream is recording
        assert not output.exists()
        assert recording.events == ["save", "name"]
    assert recording.events == ["save", "name", "flush", "disconnect"]
    assert output.read_bytes() == b"complete recording"
    assert not list(tmp_path.glob(".recording-*"))


def test_destination_appearing_during_recording_is_preserved(tmp_path, recording):
    output = tmp_path / "result.rrd"
    with pytest.raises(FileExistsError), recording_io.recording_stream(output):
        output.write_bytes(b"preserve")
    assert output.read_bytes() == b"preserve"
    assert recording.events[-1] == "disconnect"
    assert not list(tmp_path.glob(".recording-*"))


@pytest.mark.parametrize("failure", ["body", "flush", "disconnect"])
def test_failed_recording_leaves_no_output(tmp_path, recording, failure):
    output = tmp_path / "result.rrd"
    recording.fail = failure
    with (
        pytest.raises(RuntimeError, match="failed"),
        recording_io.recording_stream(output),
    ):
        if failure == "body":
            raise RuntimeError("body failed")
    assert recording.events[-1] == "disconnect"
    assert not output.exists()
    assert not list(tmp_path.glob(".recording-*"))
