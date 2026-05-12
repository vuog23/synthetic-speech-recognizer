class MelSpecConfig:
    def __init__(
        self,
        supported_extensions=None,
        silence_threshold=0.01,
        db_floor=-80.0,
        db_ref=1.0,
    ):
        self.SUPPORTED_EXTENSIONS = supported_extensions or {".wav", ".mp3"}
        self.SILENCE_THRESHOLD = silence_threshold
        self.DB_FLOOR = db_floor
        self.DB_REF = db_ref

    def __repr__(self):
        return (
            f"AudioPreprocessConfig("
            f"extensions={self.SUPPORTED_EXTENSIONS}, "
            f"silence_threshold={self.SILENCE_THRESHOLD}, "
            f"db_floor={self.DB_FLOOR}, "
            f"db_ref={self.DB_REF})"
        )