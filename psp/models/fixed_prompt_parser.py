from .semantic_parser import SemmanticParser

class PromptParser(SemmanticParser):
    def __init__(self, pretrained: str):
        super().__init__(pretrained=pretrained)

    