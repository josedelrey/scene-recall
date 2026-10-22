"""Revision-aware bounded voxel chunks built from camera-local observations."""

from dataclasses import dataclass, field

from scene_recall.mapping.voxel import MappingConfig, VoxelMap, sample_observation


@dataclass
class ReconstructionChunk:
    segment: int
    frames: list = field(default_factory=list)
    component: int = -1
    version: int = -1
    cloud: object = None


class RevisionReconstruction:
    """Reintegrate changed chunks atomically from original camera-frame samples.

    Chunks never cross local tracking failures. Stored samples grow with admitted
    history, while each fusion operation obeys the configured chunk capacity.
    A snapshot exposes only current clouds after successful synchronization.
    """

    def __init__(self, calibration, config=None):
        self.calibration = calibration
        self.config = config or MappingConfig()
        self.samples, self.chunks, self.membership = {}, [], {}
        self.version = -1
        self.ready = False
        self.dirty_chunks = set()

    def admit(self, observation, segment):
        frame = observation.frame_index
        if frame in self.samples:
            raise ValueError("reconstruction observation already admitted")
        if frame % self.config.keyframe_stride:
            return False
        samples = sample_observation(observation, self.calibration, self.config)
        if not len(samples.points):
            return False
        if (
            not self.chunks
            or self.chunks[-1].segment != segment
            or len(self.chunks[-1].frames) >= self.config.max_keyframes_per_submap
        ):
            self.chunks.append(ReconstructionChunk(segment))
        self.chunks[-1].frames.append(frame)
        self.membership[frame] = len(self.chunks) - 1
        self.samples[frame] = samples
        self.ready = False
        return True

    def synchronize(self, poses, components, version, dirty):
        self.ready = False
        self.dirty_chunks.update(
            self.membership[i] for i in dirty if i in self.membership
        )
        self.dirty_chunks.update(
            i for i, c in enumerate(self.chunks) if c.cloud is None
        )
        affected = self.dirty_chunks.copy()
        rebuilt = {}
        for index in sorted(affected):
            chunk = self.chunks[index]
            roots = {components[i] for i in chunk.frames}
            if len(roots) != 1:
                raise ValueError("one reconstruction chunk spans disconnected gauges")
            grid = VoxelMap(self.config.voxel_size_m, self.config.max_voxels)
            for frame in chunk.frames:
                grid.integrate(self.samples[frame], poses[frame], frame)
            rebuilt[index] = grid.cloud(), int(next(iter(roots)))
        # Failed fusion leaves the previous clouds intact and marks them stale.
        for index, (cloud, component) in rebuilt.items():
            self.chunks[index].cloud = cloud
            self.chunks[index].component = component
        for chunk in self.chunks:
            chunk.version = version
        self.version, self.ready = version, True
        self.dirty_chunks.clear()
        return sorted(affected)

    def snapshot(self):
        if not self.ready:
            raise RuntimeError("reconstruction has not synchronized current revisions")
        return tuple(self.chunks)
