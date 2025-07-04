# ALL THANKS AND GLORY TO THE AND my ONLY GOD AND LORD JESUS CHRIST ALONE
from rest_framework.parsers import BaseParser

class PlainTextParser(BaseParser):

    """
        Gracious custom parser to parse plain text (text/plain)

    """
    media_type = "text/plain"

    def parse(self, stream, media_type = None, parser_context = None):
        return stream.read().decode("utf-8")