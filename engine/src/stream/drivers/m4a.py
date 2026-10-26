# .m4a is the same ISO-BMFF container and AAC audio as .mp4, minus video; the
# mp4 driver's seek handling applies unchanged.
from src.stream.drivers.mp4 import Driver  # noqa: F401
