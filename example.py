from laia.lara.aspect_segment import AspectSegment

text = """
Lovely location, however, for 820 euros this was really bad value.
The room was nice, but you could have been anywhere in the
world- it felt like a chain hotel in the worst sense. The room was
tiny!Normally Four Seasons have mind blowing service and
although they were nice it was not amazing. We had just been to
Claridge's in London which was fantasic and half the price. It
wasn't bad , but it wasn't great and not worth the money. A coke
was 10 euros! There was no free wireless- all in all very average.
"""

seed_keywords=[["location"], ["room"], ["price", "value"], ["service"]]
segment = AspectSegment(text, seed_keywords)