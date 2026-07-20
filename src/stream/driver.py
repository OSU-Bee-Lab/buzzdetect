from abc import ABC, abstractmethod

class AudioDriver(ABC):
    def __init__(self, path_audio):
        self.path_audio = path_audio
        self.samplerate = None
        self.channels = None
        self.frames = None

    @abstractmethod
    def read(self):
        pass

    @abstractmethod
    def seek(self, sample):
        pass

    def tell(self):
        pass

    def close(self):
        pass

